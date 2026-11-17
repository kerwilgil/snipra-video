"""Phase 4.2C - scene-aware service composition wiring tests.

Verifies VideoIntelligenceService composes plan_analysis() ->
detect_scene_boundaries_detailed() -> refine_plan_with_scenes() -> provider
dispatch for COARSE_SCAN only, with FULL_SCAN/TARGETED_SCAN/PLAN_INFEASIBLE
paying zero detector calls and every detector failure mode falling back to
the base plan rather than blocking VIA. No real ffmpeg subprocess or Claude
API calls: detect_scene_boundaries_detailed is monkeypatched at the
app.video_intelligence.service import site (that is the name service.py
actually calls - patching scene_detection.detect_scene_boundaries_detailed
would not affect it), and _extract_frames/_call_vlm_with_frames are
monkeypatched exactly like test_planner_wiring.py already does. Since Phase
4.2D, every request here also passes through the (empty-by-default, real DB)
scene cache - see test_scene_cache.py for cache-specific coverage. Real
ffmpeg+detector+refiner composition is validated by a manual smoke run
outside the test suite (see the Phase 4.2C/4.2D slice reports), matching
Phase 4.2b's own precedent.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from sqlmodel import Session

from app.core import media
from app.core.db import engine, init_db
from app.editor import service as editor_service
from app.video_intelligence import service as via_service_module
from app.video_intelligence.chunking import (
    MAX_CHUNK_DURATION_S,
    MAX_CHUNKS_PER_REQUEST,
    AnalysisStrategy,
    SceneBoundary,
    plan_analysis,
)
from app.video_intelligence.models import (
    ExtractedFrame,
    VideoIntelligenceError,
    VideoIntelligenceErrorCode,
    VideoIntelligenceRequest,
)
from app.video_intelligence.providers import ExistingSnipraVLMProvider
from app.video_intelligence.registry import get_registry
from app.video_intelligence.scene_detection import SceneDetectionResult
from app.video_intelligence.service import VideoIntelligenceService


def _fake_extracted_frames(
    start_us: int, end_us: int, sampling_fps: float = 1.0
) -> list[ExtractedFrame]:
    """VIA v2 contract migration: `_extract_frames` returns
    ``list[ExtractedFrame]`` (frame_id + timestamp_us + data), not raw
    ``[b"frame"] * n``. Block 02's parsers build a ``frame_id_to_ts`` map from
    it for evidence provenance, so a bytes list raises ``'bytes' object has no
    attribute 'frame_id'`` and fails every chunk. Mirrors
    ``ExistingSnipraVLMProvider._extract_frames_blocking``'s deterministic
    ``frame_{idx:04d}`` id / ``start_us + idx/sampling_fps`` timestamp scheme.
    """
    duration_s = (end_us - start_us) / 1_000_000
    count = max(1, round(duration_s * sampling_fps))
    frames: list[ExtractedFrame] = []
    for idx in range(count):
        ts = (
            start_us + int((idx / sampling_fps) * 1_000_000)
            if sampling_fps > 0
            else start_us
        )
        frames.append(
            ExtractedFrame(frame_id=f"frame_{idx:04d}", timestamp_us=ts, data=b"frame")
        )
    return frames


@pytest.fixture(autouse=True)
def _via_available(monkeypatch):
    """Tras N.1, VIA es fail-closed sin proveedor visual configurado. Estos
    tests ejercitan wiring de planner/scene, no la política de proveedores;
    fijar un proveedor visual válido con clave FALSA mantiene el test igual,
    porque _call_vlm_with_frames está mockeado en todos ellos."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "llm_provider", "claude")
    monkeypatch.setattr(settings, "llm_model", "claude-haiku-4-5")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")


def _request(
    asset_id: str,
    duration_s: float,
    mode: str = "summarize_video",
    start_s: float = 0.0,
) -> VideoIntelligenceRequest:
    kwargs = dict(
        asset_id=asset_id,
        start_us=int(start_s * 1_000_000),
        end_us=int((start_s + duration_s) * 1_000_000),
        mode=mode,
    )
    if mode == "ask_video":
        kwargs["question"] = "what happens?"
    else:
        kwargs["objective"] = "Summarize"
    return VideoIntelligenceRequest(**kwargs)


def _real_asset(tmp_path, name: str) -> str:
    """Creates a real EditorAsset row via the canonical app.editor.service
    helpers (the same ones test_editor_agent_e2e.py etc. already use) so
    scene_cache.resolve_scene_cache_lookup resolves a real original_path/
    fingerprint/asset_metadata exactly like production does - not a
    hand-rolled DB insert. Each call gets a distinct project (via name), so
    asset_metadata always starts empty - a guaranteed cache miss on first
    use, which is what most of these wiring tests want."""
    init_db()
    with Session(engine) as session:
        project = editor_service.create_project(session, f"scene-aware-{name}")
        source = tmp_path / f"{name}.mp4"
        # Unique bytes per asset: add_asset fingerprints file content (sha256),
        # and the VIA result cache keys on that fingerprint. Identical bytes
        # across tests made every asset share one cache key, so a prior test's
        # cached summarize_video result was served here and the provider spy
        # never fired. (Latent since the helper's "distinct asset per test"
        # intent; unmasked once _extract_frames fakes stopped crashing.)
        source.write_bytes(f"fake video bytes {name}".encode())
        asset = editor_service.add_asset(session, project, source, duration_us=90_000_000)
        return asset.id


def _install_always_succeeding_fakes(monkeypatch, provider):
    async def _fake_extract_frames(asset_id, start_us, end_us, sampling_fps=1.0):
        return _fake_extracted_frames(start_us, end_us, sampling_fps)

    monkeypatch.setattr(provider, "_extract_frames", _fake_extract_frames)
    monkeypatch.setattr(
        provider,
        "_call_vlm_with_frames",
        lambda system, prompt, frames: json.dumps(
            {
                "answer": "ok",
                "confidence": 0.9,
                "observations": [],
                "events": [],
                "scenes": [],
                "objects": [],
            }
        ),
    )


def _counting_detector(monkeypatch, boundaries=None, cacheable=True):
    """Patches the detect_scene_boundaries_detailed name as bound into
    app.video_intelligence.service (module-level `from ... import
    detect_scene_boundaries_detailed`), not the scene_detection module
    itself. Only called on a real cache miss - every asset here starts with
    empty asset_metadata (see _real_asset), so that's every call in this
    file unless a test seeds the cache itself."""
    calls: list[tuple[str, int, int]] = []

    def _fake(asset_path, start_us, end_us, **kwargs):
        calls.append((asset_path, start_us, end_us))
        return SceneDetectionResult(
            boundaries=list(boundaries) if boundaries else [], cacheable=cacheable
        )

    monkeypatch.setattr(via_service_module, "detect_scene_boundaries_detailed", _fake)
    return calls


class TestFullScanNeverCallsDetector:
    """Item A."""

    def test_full_scan_detector_calls_zero(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "full-scan")
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)
        calls = _counting_detector(monkeypatch)

        request = _request(asset_id, duration_s=10)
        assert plan_analysis(request).strategy == AnalysisStrategy.FULL_SCAN

        asyncio.run(service.summarize_video(request))
        assert len(calls) == 0


class TestTargetedScanNeverCallsDetector:
    """Item B."""

    def test_targeted_scan_detector_calls_zero(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "targeted-scan")
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)
        calls = _counting_detector(monkeypatch)

        request = _request(asset_id, duration_s=90, mode="ask_video")
        assert plan_analysis(request).strategy == AnalysisStrategy.TARGETED_SCAN

        asyncio.run(service.ask_video(request))
        assert len(calls) == 0


class TestCoarseScanCallsDetectorAtMostOnce:
    """Item C, and section 9's "1 detection / request, never per chunk"."""

    def test_coarse_scan_multichunk_detector_called_once(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "coarse-scan")
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)
        calls = _counting_detector(monkeypatch, boundaries=[])

        request = _request(asset_id, duration_s=65)
        base_plan = plan_analysis(request)
        assert base_plan.strategy == AnalysisStrategy.COARSE_SCAN
        assert len(base_plan.chunks) > 1

        asyncio.run(service.summarize_video(request))
        assert len(calls) == 1


class TestEmptyBoundariesFallBackToBasePlan:
    """Items D and H: whether boundaries=[] came from "no real cuts" or from
    an environmental detector failure, detect_scene_boundaries() already
    normalizes both to [] (Phase 4.2b) - the service cannot and should not
    distinguish them. Either way the provider is still called, with the
    unmodified base plan."""

    def test_provider_still_called_with_base_plan(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "empty-boundaries")
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)
        _counting_detector(monkeypatch, boundaries=[])

        received = {}
        real_summarize = ExistingSnipraVLMProvider.summarize_video

        async def _spy(self, request, analysis_plan=None):
            received["plan"] = analysis_plan
            received["called"] = received.get("called", 0) + 1
            return await real_summarize(self, request, analysis_plan)

        monkeypatch.setattr(ExistingSnipraVLMProvider, "summarize_video", _spy)

        request = _request(asset_id, duration_s=65)
        base_plan = plan_analysis(request)

        result = asyncio.run(service.summarize_video(request))

        assert received["called"] == 1
        assert received["plan"] == base_plan
        assert result.metadata.partial is False


class TestRelevantBoundariesRefinePlan:
    """Items E, F, G, and section 14's deterministic improvement proof."""

    def test_provider_receives_refined_plan_closer_to_scene_cut(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "relevant-boundaries")
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)

        request = _request(asset_id, duration_s=65)
        base_plan = plan_analysis(request)
        # base internal boundary sits at 30s; the "real" cut is 1.5s earlier,
        # well inside the 2.5s snap tolerance.
        scene_boundary_us = base_plan.chunks[0].end_us - 1_500_000
        _counting_detector(
            monkeypatch, boundaries=[SceneBoundary(timestamp_us=scene_boundary_us)]
        )

        received = {}
        real_summarize = ExistingSnipraVLMProvider.summarize_video

        async def _spy(self, request, analysis_plan=None):
            received["plan"] = analysis_plan
            return await real_summarize(self, request, analysis_plan)

        monkeypatch.setattr(ExistingSnipraVLMProvider, "summarize_video", _spy)

        asyncio.run(service.summarize_video(request))
        refined_plan = received["plan"]

        # E: refined, not base, plan reached the provider.
        assert refined_plan != base_plan
        # F: scene-awareness never changes chunk/call count.
        assert len(refined_plan.chunks) == len(base_plan.chunks)
        # G: refined boundary strictly closer to (here: exactly on) the real
        # scene cut than the base plan's technical boundary was.
        base_distance = abs(base_plan.chunks[0].end_us - scene_boundary_us)
        refined_distance = abs(refined_plan.chunks[0].end_us - scene_boundary_us)
        assert refined_distance < base_distance
        assert refined_distance == 0
        assert refined_plan.chunks[0].end_us == scene_boundary_us


class TestUnresolvableAssetPathFallsBack:
    """Section 8/6: asset lookup failure is an environmental failure like any
    other - detector is never invoked, base plan reaches the provider."""

    def test_missing_asset_skips_detector_without_failing_request(self, monkeypatch):
        init_db()
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)
        calls = _counting_detector(monkeypatch)

        request = _request("does-not-exist-in-db", duration_s=65)
        result = asyncio.run(service.summarize_video(request))

        assert len(calls) == 0
        assert result is not None
        assert result.metadata.partial is False


class TestPlanInfeasibleSkipsDetectorAndProvider:
    """Item I: PLAN_INFEASIBLE fails fast before any scene detection or
    provider dispatch, mirroring TestPlanInfeasibleFailsFast in
    test_planner_wiring.py."""

    def test_infeasible_range_never_reaches_detector_or_provider(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "plan-infeasible")
        service = VideoIntelligenceService()
        detector_calls = _counting_detector(monkeypatch)
        provider_calls = {"n": 0}

        async def _counting_summarize(self, request, analysis_plan=None):
            provider_calls["n"] += 1
            raise AssertionError("provider.summarize_video must not be called")

        monkeypatch.setattr(ExistingSnipraVLMProvider, "summarize_video", _counting_summarize)

        boundary_s = MAX_CHUNK_DURATION_S * MAX_CHUNKS_PER_REQUEST
        request = _request(asset_id, duration_s=boundary_s + 3600)

        with pytest.raises(VideoIntelligenceError) as exc_info:
            asyncio.run(service.summarize_video(request))

        assert exc_info.value.code == VideoIntelligenceErrorCode.PLAN_INFEASIBLE
        assert len(detector_calls) == 0
        assert provider_calls["n"] == 0


class TestCancellationPropagates:
    """Item J: a real FFmpegCanceledError from the detector must not be
    swallowed into a silent base-plan fallback."""

    def test_detector_cancellation_is_not_swallowed(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "cancellation")
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)

        def _raise(asset_path, start_us, end_us, **kwargs):
            raise media.FFmpegCanceledError("job canceled")

        monkeypatch.setattr(via_service_module, "detect_scene_boundaries_detailed", _raise)

        request = _request(asset_id, duration_s=65)

        with pytest.raises(media.FFmpegCanceledError):
            asyncio.run(service.summarize_video(request))
