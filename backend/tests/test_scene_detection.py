"""Phase 4.2b - real scene boundary detector.

detect_scene_boundaries() is an I/O boundary (ffmpeg subprocess), not a pure
function - app.core.media.run() is mocked throughout so these tests never
invoke a real subprocess. See the "REAL SMOKE" section of the Slice report
for the one genuine ffmpeg run performed manually, outside the test suite.
"""

from __future__ import annotations

import subprocess

import pytest

from app.core import media
from app.video_intelligence import scene_detection
from app.video_intelligence.chunking import SceneBoundary
from app.video_intelligence.scene_detection import (
    DEFAULT_SCENE_CHANGE_THRESHOLD,
    MAX_SCENE_BOUNDARIES,
    detect_scene_boundaries,
)


def _fake_run(stderr: str):
    def _fn(cmd, timeout=None, cwd=None):
        return subprocess.CompletedProcess(cmd, 0, "", stderr)

    return _fn


class TestParsing:
    def test_a_single_boundary(self, monkeypatch):
        monkeypatch.setattr(
            media, "run", _fake_run("... n:0 pts_time:5.2 iskey:1 ...")
        )
        result = detect_scene_boundaries("fake.mp4", 0, 20_000_000)
        assert result == [SceneBoundary(timestamp_us=5_200_000)]

    def test_b_multiple_boundaries(self, monkeypatch):
        stderr = "pts_time:2.0\n" "pts_time:8.5\n" "pts_time:15.25\n"
        monkeypatch.setattr(media, "run", _fake_run(stderr))
        result = detect_scene_boundaries("fake.mp4", 0, 20_000_000)
        assert [b.timestamp_us for b in result] == [2_000_000, 8_500_000, 15_250_000]

    def test_c_unordered_input_sorted_output(self, monkeypatch):
        stderr = "pts_time:15.0\npts_time:2.0\npts_time:8.0\n"
        monkeypatch.setattr(media, "run", _fake_run(stderr))
        result = detect_scene_boundaries("fake.mp4", 0, 20_000_000)
        timestamps = [b.timestamp_us for b in result]
        assert timestamps == sorted(timestamps)

    def test_d_duplicates_deduplicated(self, monkeypatch):
        stderr = "pts_time:5.0\npts_time:5.0\npts_time:5.0\npts_time:9.0\n"
        monkeypatch.setattr(media, "run", _fake_run(stderr))
        result = detect_scene_boundaries("fake.mp4", 0, 20_000_000)
        assert [b.timestamp_us for b in result] == [5_000_000, 9_000_000]

    def test_e_malformed_lines_ignored(self, monkeypatch):
        stderr = (
            "garbage line with no timestamp\n"
            "pts_time:corrupt_not_a_number\n"  # regex requires digits, won't match
            "pts_time:5.0\n"
            "another random ffmpeg log line\n"
        )
        monkeypatch.setattr(media, "run", _fake_run(stderr))
        result = detect_scene_boundaries("fake.mp4", 0, 20_000_000)
        assert [b.timestamp_us for b in result] == [5_000_000]

    def test_f_zero_boundaries(self, monkeypatch):
        monkeypatch.setattr(media, "run", _fake_run("no scene changes here"))
        result = detect_scene_boundaries("fake.mp4", 0, 20_000_000)
        assert result == []


class TestFailureSemantics:
    def test_g_subprocess_failure_returns_empty(self, monkeypatch):
        def _raise(cmd, timeout=None, cwd=None):
            raise media.FFmpegError("boom")

        monkeypatch.setattr(media, "run", _raise)
        assert detect_scene_boundaries("fake.mp4", 0, 20_000_000) == []

    def test_h_timeout_returns_empty(self, monkeypatch):
        def _raise(cmd, timeout=None, cwd=None):
            raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=600)

        monkeypatch.setattr(media, "run", _raise)
        assert detect_scene_boundaries("fake.mp4", 0, 20_000_000) == []

    def test_oserror_returns_empty(self, monkeypatch):
        def _raise(cmd, timeout=None, cwd=None):
            raise OSError("ffmpeg executable not found")

        monkeypatch.setattr(media, "run", _raise)
        assert detect_scene_boundaries("fake.mp4", 0, 20_000_000) == []

    def test_p_cancellation_propagates_not_swallowed(self, monkeypatch):
        def _raise(cmd, timeout=None, cwd=None):
            raise media.FFmpegCanceledError("job canceled")

        monkeypatch.setattr(media, "run", _raise)
        with pytest.raises(media.FFmpegCanceledError):
            detect_scene_boundaries("fake.mp4", 0, 20_000_000)


class TestRangeFiltering:
    def test_i_boundary_outside_range_excluded(self, monkeypatch):
        # -t limits ffmpeg's own window, but a defensive filter still
        # excludes anything ffmpeg might report past it.
        monkeypatch.setattr(media, "run", _fake_run("pts_time:15.0\n"))
        result = detect_scene_boundaries("fake.mp4", 0, 10_000_000)
        assert result == []

    def test_j_boundary_exactly_at_start_excluded(self, monkeypatch):
        monkeypatch.setattr(media, "run", _fake_run("pts_time:0\n"))
        result = detect_scene_boundaries("fake.mp4", 0, 10_000_000)
        assert result == []

    def test_k_boundary_exactly_at_end_excluded(self, monkeypatch):
        monkeypatch.setattr(media, "run", _fake_run("pts_time:10.0\n"))  # == duration
        result = detect_scene_boundaries("fake.mp4", 0, 10_000_000)
        assert result == []

    def test_l_nonzero_start_absolute_conversion(self, monkeypatch):
        # Matches the design gate's own worked example: start_us=120_000_000,
        # ffmpeg (input-seeked) reports pts_time=5.2 relative to the seek ->
        # absolute SceneBoundary must be 125_200_000.
        monkeypatch.setattr(media, "run", _fake_run("pts_time:5.2\n"))
        result = detect_scene_boundaries("fake.mp4", 120_000_000, 180_000_000)
        assert result == [SceneBoundary(timestamp_us=125_200_000)]


class TestCap:
    def test_m_more_than_cap_boundaries_capped_deterministically(self, monkeypatch):
        # 600 one-second-apart boundaries across a 700s range.
        stderr = "".join(f"pts_time:{t}.0\n" for t in range(1, 601))
        monkeypatch.setattr(media, "run", _fake_run(stderr))
        result = detect_scene_boundaries("fake.mp4", 0, 700_000_000)
        assert len(result) == MAX_SCENE_BOUNDARIES
        # Sorted ascending, first 500 kept - not "500 strongest" (no score
        # exists to rank by).
        assert result[0].timestamp_us == 1_000_000
        assert result[-1].timestamp_us == 500_000_000


class TestCommandConstruction:
    def test_n_threshold_interpolated_into_command(self, monkeypatch):
        captured = {}

        def _capture(cmd, timeout=None, cwd=None):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(media, "run", _capture)
        detect_scene_boundaries("fake.mp4", 0, 20_000_000, threshold=0.5)

        vf_arg = captured["cmd"][captured["cmd"].index("-vf") + 1]
        assert "gt(scene,0.5)" in vf_arg

    def test_default_threshold_used_when_unspecified(self, monkeypatch):
        captured = {}

        def _capture(cmd, timeout=None, cwd=None):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(media, "run", _capture)
        detect_scene_boundaries("fake.mp4", 0, 20_000_000)

        vf_arg = captured["cmd"][captured["cmd"].index("-vf") + 1]
        assert f"gt(scene,{DEFAULT_SCENE_CHANGE_THRESHOLD})" in vf_arg


class TestThresholdValidation:
    def test_o_threshold_zero_rejected_before_subprocess(self, monkeypatch):
        calls = {"n": 0}

        def _count(cmd, timeout=None, cwd=None):
            calls["n"] += 1
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(media, "run", _count)
        with pytest.raises(ValueError):
            detect_scene_boundaries("fake.mp4", 0, 20_000_000, threshold=0.0)
        assert calls["n"] == 0

    def test_threshold_above_one_rejected(self, monkeypatch):
        calls = {"n": 0}

        def _count(cmd, timeout=None, cwd=None):
            calls["n"] += 1
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(media, "run", _count)
        with pytest.raises(ValueError):
            detect_scene_boundaries("fake.mp4", 0, 20_000_000, threshold=1.5)
        assert calls["n"] == 0

    def test_inverted_range_rejected_before_subprocess(self, monkeypatch):
        calls = {"n": 0}

        def _count(cmd, timeout=None, cwd=None):
            calls["n"] += 1
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(media, "run", _count)
        with pytest.raises(ValueError):
            detect_scene_boundaries("fake.mp4", 20_000_000, 10_000_000)
        assert calls["n"] == 0


class TestNoEditorialLeakage:
    def test_editorial_plan_schema_has_no_detector_vocabulary(self):
        import json

        from app.editor.agent import EditorialPlan

        schema = json.dumps(EditorialPlan.model_json_schema())
        for term in ("detect_scene_boundaries", "DEFAULT_SCENE_CHANGE_THRESHOLD"):
            assert term not in schema

    def test_evidence_request_schema_has_no_detector_vocabulary(self):
        import json

        from app.editor.evidence_request import EvidenceRequest

        schema = json.dumps(EvidenceRequest.model_json_schema())
        for term in ("detect_scene_boundaries", "DEFAULT_SCENE_CHANGE_THRESHOLD"):
            assert term not in schema


def test_module_does_not_import_pipeline():
    """Section 1: video_intelligence must not depend on app.pipeline."""
    import ast
    import inspect

    source = inspect.getsource(scene_detection)
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    parts = {part for module in imported_modules for part in module.split(".")}
    assert "pipeline" not in parts
    assert "cv2" not in parts
    assert "scenedetect" not in parts
