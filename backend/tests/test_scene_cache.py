"""Phase 4.2D - persistent scene boundary cache tests.

Two layers: unit tests against app.video_intelligence.scene_cache's pure
read/write functions directly (identity matching, revalidation, metadata
preservation, eviction - fast, no service/provider involved), and
integration tests through VideoIntelligenceService (cache miss -> write ->
hit, non-cacheable failures, cancellation, write-failure resilience, and the
COARSE_SCAN-only runtime scope) using the same monkeypatch style
test_scene_aware_wiring.py already established. No real ffmpeg subprocess or
Claude API calls in this file - see the Phase 4.2D slice report for the one
real ffmpeg smoke run performed manually, outside the test suite.
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
    VideoIntelligenceRequest,
)
from app.video_intelligence.providers import ExistingSnipraVLMProvider
from app.video_intelligence.registry import get_registry
from app.video_intelligence.scene_cache import (
    MAX_SCENE_CACHE_ENTRIES_PER_ASSET,
    SCENE_DETECTOR_CACHE_VERSION,
    read_cached_boundaries,
    resolve_scene_cache_lookup,
    write_cached_boundaries,
)
from app.video_intelligence.scene_detection import (
    DEFAULT_SCENE_CHANGE_THRESHOLD,
    SceneDetectionResult,
)
from app.video_intelligence.service import VideoIntelligenceService

THRESHOLD = DEFAULT_SCENE_CHANGE_THRESHOLD


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
    """Tras N.1, VIA es fail-closed sin proveedor visual. Estos tests cubren
    el cache de escenas, no la política de proveedores: disponibilidad real
    con credencial FALSA, así las rutas de cache son las que se ejercitan."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "llm_provider", "claude")
    monkeypatch.setattr(settings, "llm_model", "claude-haiku-4-5")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")


def _real_asset(tmp_path, name: str, asset_metadata: dict | None = None) -> str:
    """Real EditorAsset row via the canonical app.editor.service helpers, so
    fingerprint/original_path/asset_metadata all resolve exactly like
    production. Each name gets its own project, so metadata starts empty
    unless asset_metadata is supplied."""
    init_db()
    with Session(engine) as session:
        project = editor_service.create_project(session, f"scene-cache-{name}")
        source = tmp_path / f"{name}.mp4"
        source.write_bytes(f"fake video bytes for {name}".encode())
        asset = editor_service.add_asset(
            session, project, source, duration_us=90_000_000, asset_metadata=asset_metadata
        )
        return asset.id


def _boundaries(*timestamps_us: int) -> list[SceneBoundary]:
    return [SceneBoundary(timestamp_us=ts) for ts in timestamps_us]


def _request(
    asset_id: str, duration_s: float, mode: str = "summarize_video", objective: str = "Summarize"
) -> VideoIntelligenceRequest:
    kwargs = dict(asset_id=asset_id, start_us=0, end_us=int(duration_s * 1_000_000), mode=mode)
    if mode == "ask_video":
        kwargs["question"] = "what happens?"
    else:
        kwargs["objective"] = objective
    return VideoIntelligenceRequest(**kwargs)


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
    """Patches detect_scene_boundaries_detailed as bound into
    app.video_intelligence.service. Raises if invoked more than once by
    default is NOT enforced here - callers assert call count themselves."""
    calls: list[tuple[str, int, int]] = []

    def _fake(asset_path, start_us, end_us, **kwargs):
        calls.append((asset_path, start_us, end_us))
        return SceneDetectionResult(
            boundaries=list(boundaries) if boundaries else [], cacheable=cacheable
        )

    monkeypatch.setattr(via_service_module, "detect_scene_boundaries_detailed", _fake)
    return calls


def _forbidden_detector(monkeypatch):
    def _fail(asset_path, start_us, end_us, **kwargs):
        raise AssertionError("detect_scene_boundaries_detailed must not be called")

    monkeypatch.setattr(via_service_module, "detect_scene_boundaries_detailed", _fail)


def _spy_cache_lookup(monkeypatch):
    real = via_service_module.resolve_scene_cache_lookup
    calls = {"n": 0}

    def _wrapped(asset_id):
        calls["n"] += 1
        return real(asset_id)

    monkeypatch.setattr(via_service_module, "resolve_scene_cache_lookup", _wrapped)
    return calls


def _spy_write(monkeypatch):
    real = via_service_module.write_cached_boundaries
    calls = []

    def _wrapped(asset_id, **kwargs):
        calls.append((asset_id, kwargs))
        return real(asset_id, **kwargs)

    monkeypatch.setattr(via_service_module, "write_cached_boundaries", _wrapped)
    return calls


class TestServiceMissThenHit:
    """Items A and B: first request writes the cache on a genuine miss,
    the immediately following identical request hits it without touching
    the detector."""

    def test_first_request_misses_writes_second_request_hits(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "miss-then-hit")
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)

        request = _request(asset_id, duration_s=65)
        base_plan = plan_analysis(request)
        assert base_plan.strategy == AnalysisStrategy.COARSE_SCAN

        scene_ts = base_plan.chunks[0].end_us - 1_500_000
        detector_calls = _counting_detector(monkeypatch, boundaries=_boundaries(scene_ts))

        asyncio.run(service.summarize_video(request))
        assert len(detector_calls) == 1

        lookup = resolve_scene_cache_lookup(asset_id)
        cached = read_cached_boundaries(
            lookup.asset_metadata,
            fingerprint=lookup.fingerprint,
            threshold=THRESHOLD,
            start_us=request.start_us,
            end_us=request.end_us,
        )
        assert cached == _boundaries(scene_ts)

        # Second request over the same asset/range/threshold, but with a
        # different objective text - deliberately NOT identical to the first
        # request, so it misses VideoIntelligenceCache's own request-level
        # result cache (a different, unrelated cache - see
        # scene_cache.py's module docstring) and actually re-enters the
        # scene-aware composition path this test wants to exercise. The
        # scene cache itself doesn't key on objective/question at all.
        _forbidden_detector(monkeypatch)
        received = {}
        real_summarize = ExistingSnipraVLMProvider.summarize_video

        async def _spy(self, req, analysis_plan=None):
            received["plan"] = analysis_plan
            return await real_summarize(self, req, analysis_plan)

        monkeypatch.setattr(ExistingSnipraVLMProvider, "summarize_video", _spy)

        second_request = _request(asset_id, duration_s=65, objective="Summarize again")
        asyncio.run(service.summarize_video(second_request))
        assert received["plan"].chunks[0].end_us == scene_ts


class TestRangeIdentity:
    """Item C."""

    def test_different_range_is_a_miss(self):
        metadata = {
            "video_intelligence": {
                "scene_detection": {
                    "version": SCENE_DETECTOR_CACHE_VERSION,
                    "entries": [
                        {
                            "fingerprint": "fp",
                            "threshold": THRESHOLD,
                            "start_us": 0,
                            "end_us": 65_000_000,
                            "boundaries_us": [30_000_000],
                        }
                    ],
                }
            }
        }
        hit = read_cached_boundaries(
            metadata, fingerprint="fp", threshold=THRESHOLD, start_us=0, end_us=90_000_000
        )
        assert hit is None


class TestThresholdIdentity:
    """Item D."""

    def test_different_threshold_is_a_miss(self):
        metadata = {
            "video_intelligence": {
                "scene_detection": {
                    "version": SCENE_DETECTOR_CACHE_VERSION,
                    "entries": [
                        {
                            "fingerprint": "fp",
                            "threshold": 0.35,
                            "start_us": 0,
                            "end_us": 65_000_000,
                            "boundaries_us": [30_000_000],
                        }
                    ],
                }
            }
        }
        hit = read_cached_boundaries(
            metadata, fingerprint="fp", threshold=0.20, start_us=0, end_us=65_000_000
        )
        assert hit is None


class TestVersionIdentity:
    """Item E."""

    def test_different_cache_version_is_a_miss(self):
        metadata = {
            "video_intelligence": {
                "scene_detection": {
                    "version": SCENE_DETECTOR_CACHE_VERSION + 999,
                    "entries": [
                        {
                            "fingerprint": "fp",
                            "threshold": THRESHOLD,
                            "start_us": 0,
                            "end_us": 65_000_000,
                            "boundaries_us": [30_000_000],
                        }
                    ],
                }
            }
        }
        hit = read_cached_boundaries(
            metadata, fingerprint="fp", threshold=THRESHOLD, start_us=0, end_us=65_000_000
        )
        assert hit is None


class TestFingerprintIdentity:
    """Item F."""

    def test_different_fingerprint_is_a_miss(self):
        metadata = {
            "video_intelligence": {
                "scene_detection": {
                    "version": SCENE_DETECTOR_CACHE_VERSION,
                    "entries": [
                        {
                            "fingerprint": "content-a",
                            "threshold": THRESHOLD,
                            "start_us": 0,
                            "end_us": 65_000_000,
                            "boundaries_us": [30_000_000],
                        }
                    ],
                }
            }
        }
        hit = read_cached_boundaries(
            metadata, fingerprint="content-b", threshold=THRESHOLD, start_us=0, end_us=65_000_000
        )
        assert hit is None


class TestBoundaryRevalidation:
    """Items G, H, I: cached boundaries are never trusted blindly."""

    def _entries(self, boundaries_us):
        return {
            "video_intelligence": {
                "scene_detection": {
                    "version": SCENE_DETECTOR_CACHE_VERSION,
                    "entries": [
                        {
                            "fingerprint": "fp",
                            "threshold": THRESHOLD,
                            "start_us": 0,
                            "end_us": 65_000_000,
                            "boundaries_us": boundaries_us,
                        }
                    ],
                }
            }
        }

    def test_g_valid_boundaries_pass_revalidation(self):
        hit = read_cached_boundaries(
            self._entries([10_000_000, 40_000_000]),
            fingerprint="fp",
            threshold=THRESHOLD,
            start_us=0,
            end_us=65_000_000,
        )
        assert hit == _boundaries(10_000_000, 40_000_000)
        assert all(isinstance(b, SceneBoundary) for b in hit)

    def test_h_corrupt_json_shape_is_a_miss(self):
        hit = read_cached_boundaries(
            self._entries("not-a-list"),
            fingerprint="fp",
            threshold=THRESHOLD,
            start_us=0,
            end_us=65_000_000,
        )
        assert hit is None

    def test_h_corrupt_non_numeric_entries_is_a_miss(self):
        hit = read_cached_boundaries(
            self._entries([10_000_000, "not-a-number"]),
            fingerprint="fp",
            threshold=THRESHOLD,
            start_us=0,
            end_us=65_000_000,
        )
        assert hit is None

    def test_i_out_of_range_boundary_is_a_miss(self):
        hit = read_cached_boundaries(
            self._entries([10_000_000, 70_000_000]),  # 70s > end_us=65s
            fingerprint="fp",
            threshold=THRESHOLD,
            start_us=0,
            end_us=65_000_000,
        )
        assert hit is None

    def test_i_boundary_at_exact_edges_is_a_miss(self):
        hit = read_cached_boundaries(
            self._entries([0, 65_000_000]),  # exactly start_us/end_us
            fingerprint="fp",
            threshold=THRESHOLD,
            start_us=0,
            end_us=65_000_000,
        )
        assert hit is None


class TestSuccessfulEmptyIsCacheable:
    """Item J: a genuinely completed run with zero cuts is a valid,
    cacheable result - distinct from "never ran"."""

    def test_empty_boundaries_roundtrip_as_a_valid_hit(self, tmp_path):
        asset_id = _real_asset(tmp_path, "empty-cacheable")
        ok = write_cached_boundaries(
            asset_id,
            fingerprint="fp",
            threshold=THRESHOLD,
            start_us=0,
            end_us=65_000_000,
            boundaries=[],
        )
        assert ok is True

        lookup = resolve_scene_cache_lookup(asset_id)
        hit = read_cached_boundaries(
            lookup.asset_metadata,
            fingerprint="fp",
            threshold=THRESHOLD,
            start_us=0,
            end_us=65_000_000,
        )
        assert hit == []
        assert hit is not None


class TestNonCacheableFailureNotPersisted:
    """Items K and L: detect_scene_boundaries_detailed already normalizes
    both ffmpeg failure and timeout to the same SceneDetectionResult
    shape (cacheable=False) - from the cache's perspective they are
    identical, so one parametrized check covers both."""

    @pytest.mark.parametrize("scenario", ["ffmpeg_failure", "timeout"])
    def test_non_cacheable_result_writes_nothing(self, monkeypatch, tmp_path, scenario):
        asset_id = _real_asset(tmp_path, f"non-cacheable-{scenario}")
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)
        _counting_detector(monkeypatch, boundaries=[], cacheable=False)
        write_calls = _spy_write(monkeypatch)

        request = _request(asset_id, duration_s=65)
        asyncio.run(service.summarize_video(request))

        assert len(write_calls) == 0
        lookup = resolve_scene_cache_lookup(asset_id)
        assert lookup.asset_metadata.get("video_intelligence") is None


class TestCancellationNotCached:
    """Item M."""

    def test_cancellation_propagates_and_writes_nothing(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "cancellation")
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)
        write_calls = _spy_write(monkeypatch)

        def _raise(asset_path, start_us, end_us, **kwargs):
            raise media.FFmpegCanceledError("job canceled")

        monkeypatch.setattr(via_service_module, "detect_scene_boundaries_detailed", _raise)

        request = _request(asset_id, duration_s=65)
        with pytest.raises(media.FFmpegCanceledError):
            asyncio.run(service.summarize_video(request))

        assert len(write_calls) == 0


class TestCacheWriteFailureDoesNotBreakRequest:
    """Item N: cache persistence is best-effort - a DB failure while writing
    must not affect the (already-refined) result the caller gets back."""

    def test_write_failure_still_returns_refined_result(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "write-failure")
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)

        request = _request(asset_id, duration_s=65)
        base_plan = plan_analysis(request)
        scene_ts = base_plan.chunks[0].end_us - 1_500_000
        _counting_detector(monkeypatch, boundaries=_boundaries(scene_ts))
        monkeypatch.setattr(via_service_module, "write_cached_boundaries", lambda *a, **k: False)

        received = {}
        real_summarize = ExistingSnipraVLMProvider.summarize_video

        async def _spy(self, req, analysis_plan=None):
            received["plan"] = analysis_plan
            return await real_summarize(self, req, analysis_plan)

        monkeypatch.setattr(ExistingSnipraVLMProvider, "summarize_video", _spy)

        result = asyncio.run(service.summarize_video(request))
        assert result.metadata.partial is False
        assert received["plan"].chunks[0].end_us == scene_ts


class TestMetadataPreservation:
    """Item O."""

    def test_existing_asset_metadata_survives_cache_write(self, tmp_path):
        asset_id = _real_asset(
            tmp_path, "preserve-metadata", asset_metadata={"existing": {"foo": "bar"}}
        )
        write_cached_boundaries(
            asset_id,
            fingerprint="fp",
            threshold=THRESHOLD,
            start_us=0,
            end_us=65_000_000,
            boundaries=_boundaries(30_000_000),
        )

        lookup = resolve_scene_cache_lookup(asset_id)
        assert lookup.asset_metadata["existing"] == {"foo": "bar"}
        assert "video_intelligence" in lookup.asset_metadata


class TestMultipleRangesCoexist:
    """Item P."""

    def test_two_distinct_ranges_both_readable(self, tmp_path):
        asset_id = _real_asset(tmp_path, "multi-range")
        write_cached_boundaries(
            asset_id,
            fingerprint="fp",
            threshold=THRESHOLD,
            start_us=0,
            end_us=65_000_000,
            boundaries=_boundaries(30_000_000),
        )
        write_cached_boundaries(
            asset_id,
            fingerprint="fp",
            threshold=THRESHOLD,
            start_us=100_000_000,
            end_us=165_000_000,
            boundaries=_boundaries(130_000_000),
        )

        lookup = resolve_scene_cache_lookup(asset_id)
        first = read_cached_boundaries(
            lookup.asset_metadata,
            fingerprint="fp",
            threshold=THRESHOLD,
            start_us=0,
            end_us=65_000_000,
        )
        second = read_cached_boundaries(
            lookup.asset_metadata,
            fingerprint="fp",
            threshold=THRESHOLD,
            start_us=100_000_000,
            end_us=165_000_000,
        )
        assert first == _boundaries(30_000_000)
        assert second == _boundaries(130_000_000)


class TestEntryCapEviction:
    """Item Q."""

    def test_cap_enforced_fifo(self, tmp_path):
        asset_id = _real_asset(tmp_path, "entry-cap")
        total = MAX_SCENE_CACHE_ENTRIES_PER_ASSET + 3
        for i in range(total):
            write_cached_boundaries(
                asset_id,
                fingerprint="fp",
                threshold=THRESHOLD,
                start_us=i * 1_000_000_000,
                end_us=i * 1_000_000_000 + 65_000_000,
                boundaries=_boundaries(i * 1_000_000_000 + 30_000_000),
            )

        lookup = resolve_scene_cache_lookup(asset_id)
        entries = lookup.asset_metadata["video_intelligence"]["scene_detection"]["entries"]
        assert len(entries) == MAX_SCENE_CACHE_ENTRIES_PER_ASSET

        # The earliest-written ranges (i=0, 1, 2) must have been evicted.
        for i in range(3):
            hit = read_cached_boundaries(
                lookup.asset_metadata,
                fingerprint="fp",
                threshold=THRESHOLD,
                start_us=i * 1_000_000_000,
                end_us=i * 1_000_000_000 + 65_000_000,
            )
            assert hit is None
        # The most recent range must still be present.
        last = total - 1
        hit = read_cached_boundaries(
            lookup.asset_metadata,
            fingerprint="fp",
            threshold=THRESHOLD,
            start_us=last * 1_000_000_000,
            end_us=last * 1_000_000_000 + 65_000_000,
        )
        assert hit is not None


class TestRuntimeScopeSkipsCacheAndDetector:
    """Items R, S, T: only COARSE_SCAN (multi-chunk) ever looks at the
    cache."""

    def test_r_full_scan_no_lookup_no_detector(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "scope-full-scan")
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)
        lookup_calls = _spy_cache_lookup(monkeypatch)
        detector_calls = _counting_detector(monkeypatch)

        request = _request(asset_id, duration_s=10)
        assert plan_analysis(request).strategy == AnalysisStrategy.FULL_SCAN
        asyncio.run(service.summarize_video(request))

        assert lookup_calls["n"] == 0
        assert len(detector_calls) == 0

    def test_s_targeted_scan_no_lookup_no_detector(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "scope-targeted-scan")
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)
        lookup_calls = _spy_cache_lookup(monkeypatch)
        detector_calls = _counting_detector(monkeypatch)

        request = _request(asset_id, duration_s=90, mode="ask_video")
        assert plan_analysis(request).strategy == AnalysisStrategy.TARGETED_SCAN
        asyncio.run(service.ask_video(request))

        assert lookup_calls["n"] == 0
        assert len(detector_calls) == 0

    def test_t_plan_infeasible_no_lookup_no_detector_no_provider(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "scope-plan-infeasible")
        service = VideoIntelligenceService()
        lookup_calls = _spy_cache_lookup(monkeypatch)
        detector_calls = _counting_detector(monkeypatch)
        provider_calls = {"n": 0}

        async def _counting_summarize(self, request, analysis_plan=None):
            provider_calls["n"] += 1
            raise AssertionError("must not be called")

        monkeypatch.setattr(ExistingSnipraVLMProvider, "summarize_video", _counting_summarize)

        boundary_s = MAX_CHUNK_DURATION_S * MAX_CHUNKS_PER_REQUEST
        request = _request(asset_id, duration_s=boundary_s + 3600)

        with pytest.raises(VideoIntelligenceError):
            asyncio.run(service.summarize_video(request))

        assert lookup_calls["n"] == 0
        assert len(detector_calls) == 0
        assert provider_calls["n"] == 0


class TestCacheHitRefinesPlan:
    """Item U."""

    def test_hit_refines_plan_using_cached_boundaries(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "hit-refines")
        request = _request(asset_id, duration_s=65)
        base_plan = plan_analysis(request)
        scene_ts = base_plan.chunks[0].end_us - 1_500_000

        write_cached_boundaries(
            asset_id,
            fingerprint=resolve_scene_cache_lookup(asset_id).fingerprint,
            threshold=THRESHOLD,
            start_us=request.start_us,
            end_us=request.end_us,
            boundaries=_boundaries(scene_ts),
        )

        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)
        _forbidden_detector(monkeypatch)

        received = {}
        real_summarize = ExistingSnipraVLMProvider.summarize_video

        async def _spy(self, req, analysis_plan=None):
            received["plan"] = analysis_plan
            return await real_summarize(self, req, analysis_plan)

        monkeypatch.setattr(ExistingSnipraVLMProvider, "summarize_video", _spy)

        asyncio.run(service.summarize_video(request))

        assert received["plan"] != base_plan
        assert received["plan"].chunks[0].end_us == scene_ts
        assert len(received["plan"].chunks) == len(base_plan.chunks)


class TestEmptyCacheHitUsesBasePlan:
    """Item V."""

    def test_empty_cached_result_skips_detector_uses_base_plan(self, monkeypatch, tmp_path):
        asset_id = _real_asset(tmp_path, "hit-empty")
        request = _request(asset_id, duration_s=65)
        base_plan = plan_analysis(request)

        write_cached_boundaries(
            asset_id,
            fingerprint=resolve_scene_cache_lookup(asset_id).fingerprint,
            threshold=THRESHOLD,
            start_us=request.start_us,
            end_us=request.end_us,
            boundaries=[],
        )

        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)
        _forbidden_detector(monkeypatch)

        received = {}
        real_summarize = ExistingSnipraVLMProvider.summarize_video

        async def _spy(self, req, analysis_plan=None):
            received["plan"] = analysis_plan
            return await real_summarize(self, req, analysis_plan)

        monkeypatch.setattr(ExistingSnipraVLMProvider, "summarize_video", _spy)

        asyncio.run(service.summarize_video(request))
        assert received["plan"] == base_plan
