"""Dedicated VIA v2 provenance contract tests.

These tests verify the VIA v2 contract requirements explicitly:
- Missing confidence → None/UNKNOWN
- Missing nature → UNKNOWN/UNKNOWN
- Explicit observation/inference separation
- Frame grounding with valid/invalid frame IDs
- Explicit offset handling (valid/malformed/missing)
- Object timestamp fallback behavior
- Cache contract version isolation
- Provider/model isolation
- Plan fingerprint isolation
"""

import json

import pytest

from app.editor.via_evidence import (
    _object_to_evidence,
)
from app.video_intelligence.chunking import AnalysisChunk
from app.video_intelligence.models import (
    ConfidenceSource,
    EvidenceNature,
    EvidenceProvenance,
    ExtractedFrame,
    NatureSource,
    TemporalSource,
    VideoAnswer,
    VideoEventCandidate,
    VideoIntelligenceRequest,
    VideoIntelligenceResult,
    VideoObjectEvidence,
    VideoObservation,
    VideoProcessingMetadata,
)
from app.video_intelligence.providers import ExistingSnipraVLMProvider

# --------------------------------------------------------------------------- #
# Real-parser integration harness                                            #
#                                                                            #
# These helpers drive the ACTUAL VIA v2 parsers                              #
# (_parse_response / _parse_ask_chunk_response / _parse_chunk_response) with #
# real raw-JSON payloads, not just _determine_temporal_source() in          #
# isolation, so temporal grounding is verified end to end and can never     #
# silently diverge between the three parsers.                               #
# --------------------------------------------------------------------------- #

FLOOR_US = 1_000_000
CEIL_US = 11_000_000
KNOWN_FRAME_TS = 2_000_000


def _mk_provider() -> ExistingSnipraVLMProvider:
    provider = ExistingSnipraVLMProvider()
    provider._vision_provider = "claude"
    provider._vision_model = "test-model"
    provider._detector_initialized = True
    provider._availability = {"available": True, "code": None, "reason": None}
    return provider


def _request() -> VideoIntelligenceRequest:
    return VideoIntelligenceRequest(
        asset_id="a",
        start_us=FLOOR_US,
        end_us=CEIL_US,
        mode="ask_video",
        question="q?",
    )


def _frames() -> list[ExtractedFrame]:
    return [ExtractedFrame(frame_id="frame_0000", timestamp_us=KNOWN_FRAME_TS, data=b"x")]


def _run_parse_response(obs_payload: dict, frames: list[ExtractedFrame]):
    raw = json.dumps({"answer": "a", "observations": [obs_payload], "events": []})
    result = _mk_provider()._parse_response(_request(), raw, 1, 1, frames)
    obs = result.observations[0]
    return (
        obs.provenance.temporal_source,
        obs.provenance.frame_ids,
        obs.start_us,
        obs.end_us,
        obs.nature_source,
    )


def _run_parse_ask_chunk_response(obs_payload: dict, frames: list[ExtractedFrame]):
    raw = json.dumps({"answer": "a", "observations": [obs_payload], "events": []})
    chunk = AnalysisChunk(
        index=0,
        start_us=FLOOR_US,
        end_us=CEIL_US,
        frames_planned=1,
        sampling_fps=1.0,
    )
    out = _mk_provider()._parse_ask_chunk_response(_request(), chunk, raw, frames)
    obs = out["observations"][0]
    return (
        obs["provenance"].temporal_source,
        obs["provenance"].frame_ids,
        obs["start_us"],
        obs["end_us"],
        obs["nature_source"],
    )


def _run_parse_chunk_response(obs_payload: dict, frames: list[ExtractedFrame]):
    raw = json.dumps(
        {"observations": [obs_payload], "events": [], "scenes": [], "objects": []}
    )
    chunk = {"start_us": FLOOR_US, "end_us": CEIL_US}
    out = _mk_provider()._parse_chunk_response(_request(), chunk, raw, frames)
    obs = out["observations"][0]
    return (
        obs["provenance"].temporal_source,
        obs["provenance"].frame_ids,
        obs["start_us"],
        obs["end_us"],
        obs["nature_source"],
    )


_REAL_PARSERS = {
    "parse_response": _run_parse_response,
    "parse_ask_chunk_response": _run_parse_ask_chunk_response,
    "parse_chunk_response": _run_parse_chunk_response,
}


class TestV2ConfidenceMissing:
    """V2-01: Missing confidence → None/UNKNOWN"""

    def test_missing_confidence_returns_none(self):
        obs = VideoObservation(
            asset_id="a",
            start_us=0,
            end_us=1_000_000,
            description="test",
            nature=EvidenceNature.OBSERVATION,
            confidence=None,
            source="p",
            provider="p",
            model="m",
        )
        assert obs.confidence is None
        assert obs.confidence_source == ConfidenceSource.UNKNOWN

    def test_missing_confidence_in_answer(self):
        ans = VideoAnswer(answer="test", confidence=None)
        assert ans.confidence is None
        assert ans.confidence_source == ConfidenceSource.UNKNOWN


class TestV2NatureMissing:
    """V2-02: Missing nature in parser input → UNKNOWN/UNKNOWN"""

    def test_missing_nature_in_parser_input(self):
        from app.video_intelligence.providers import ExistingSnipraVLMProvider

        provider = ExistingSnipraVLMProvider()
        provider._vision_provider = "claude"
        provider._vision_model = "test"
        provider._detector_initialized = True
        provider._availability = {"available": True, "code": None, "reason": None}

        # Test that missing nature in raw response becomes UNKNOWN
        nature, nature_source = provider._parse_nature(None)
        assert nature == EvidenceNature.UNKNOWN
        assert nature_source == NatureSource.UNKNOWN


class TestV2ObservationNature:
    """V2-03: Explicit observation nature survives"""

    def test_observation_nature_preserved(self):
        obs = VideoObservation(
            asset_id="a",
            start_us=0,
            end_us=1_000_000,
            description="test",
            nature=EvidenceNature.OBSERVATION,
            confidence=0.8,
            confidence_source=ConfidenceSource.MODEL_EXPLICIT,
            nature_source=NatureSource.MODEL_EXPLICIT,
            source="p",
            provider="p",
            model="m",
        )
        assert obs.nature == EvidenceNature.OBSERVATION
        assert obs.nature_source == NatureSource.MODEL_EXPLICIT


class TestV2InferenceNature:
    """V2-04: Explicit inference nature survives"""

    def test_inference_nature_preserved(self):
        ev = VideoEventCandidate(
            asset_id="a",
            start_us=0,
            end_us=1_000_000,
            type="action",
            description="test",
            confidence=0.7,
            nature=EvidenceNature.INFERENCE,
            nature_source=NatureSource.MODEL_EXPLICIT,
            provider="p",
            model="m",
        )
        assert ev.nature == EvidenceNature.INFERENCE
        assert ev.nature_source == NatureSource.MODEL_EXPLICIT


class TestV2ValidFrameGrounding:
    """V2-05: Valid frame IDs → FRAME_MAPPED"""

    def test_valid_frame_ids_frame_mapped(self):
        provider = ExistingSnipraVLMProvider()
        frame_id_to_ts = {"frame_0000": 0, "frame_0001": 1_000_000}
        temporal = provider._determine_temporal_source(
            ["frame_0000", "frame_0001"], frame_id_to_ts, False
        )
        assert temporal == TemporalSource.FRAME_MAPPED


class TestV2MixedFrameIds:
    """V2-06: Mixed valid/invalid frame IDs → filtered, CHUNK_WINDOW if no offset"""

    def test_mixed_frame_ids_filtered(self):
        provider = ExistingSnipraVLMProvider()
        frame_id_to_ts = {"frame_0000": 0}
        valid = provider._validate_frame_ids(
            ["frame_0000", "frame_9999"], frame_id_to_ts
        )
        assert valid == ["frame_0000"]
        temporal = provider._determine_temporal_source(
            ["frame_0000", "frame_9999"], frame_id_to_ts, False
        )
        assert temporal == TemporalSource.CHUNK_WINDOW


class TestV2AllInvalidFrameIds:
    """V2-07: All invalid frame IDs → CHUNK_WINDOW"""

    def test_all_invalid_frame_ids(self):
        provider = ExistingSnipraVLMProvider()
        temporal = provider._determine_temporal_source(
            ["frame_9999"], {"frame_0000": 0}, False
        )
        assert temporal == TemporalSource.CHUNK_WINDOW


class TestV2ValidExplicitOffset:
    """V2-08: Valid explicit offset → MODEL_GROUNDED"""

    def test_valid_explicit_offset(self):
        provider = ExistingSnipraVLMProvider()
        temporal = provider._determine_temporal_source(
            [], {}, True  # has_explicit_offset=True
        )
        assert temporal == TemporalSource.MODEL_GROUNDED

    def test_valid_offset_parsed(self):
        provider = ExistingSnipraVLMProvider()
        ts = provider._parse_optional_offset_us("1.5s", 0, 10_000_000)
        assert ts == 1_500_000


class TestV2MalformedExplicitOffset:
    """V2-09: Malformed explicit offset → CHUNK_WINDOW"""

    def test_malformed_offset_not_model_grounded(self):
        provider = ExistingSnipraVLMProvider()
        # Test that malformed offset is not treated as explicit
        # The _has_explicit_offset should return False for unparseable values
        assert provider._has_explicit_offset({"start": "banana"}) is False
        assert provider._has_explicit_offset({"end": ""}) is False
        assert provider._has_explicit_offset({"start": None}) is False
        assert provider._has_explicit_offset({}) is False

    def test_malformed_offset_falls_to_chunk_window(self):
        provider = ExistingSnipraVLMProvider()
        temporal = provider._determine_temporal_source(
            [], {}, False  # has_explicit_offset=False for malformed
        )
        assert temporal == TemporalSource.CHUNK_WINDOW


class TestV2MissingTemporal:
    """V2-10: Missing temporal info → CHUNK_WINDOW"""

    def test_missing_temporal(self):
        provider = ExistingSnipraVLMProvider()
        temporal = provider._determine_temporal_source(
            [], {}, False
        )
        assert temporal == TemporalSource.CHUNK_WINDOW


class TestV2AskParserParity:
    """V2-11: Ask parser parity with main parser"""

    def test_ask_parser_same_temporal_logic(self):
        provider = ExistingSnipraVLMProvider()
        frame_id_to_ts = {"frame_0000": 0, "frame_0001": 1_000_000}

        # Both parsers should use same temporal logic
        obs_frame_ids = ["frame_0000", "frame_0001"]
        ev_frame_ids = ["frame_0000"]

        obs_temporal = provider._determine_temporal_source(
            obs_frame_ids, frame_id_to_ts, False
        )
        ev_temporal = provider._determine_temporal_source(
            ev_frame_ids, frame_id_to_ts, False
        )

        assert obs_temporal == TemporalSource.FRAME_MAPPED
        assert ev_temporal == TemporalSource.FRAME_MAPPED


class TestV2TargetedAskParserParity:
    """V2-12: Targeted ask parser parity"""

    def test_targeted_ask_same_temporal_logic(self):
        provider = ExistingSnipraVLMProvider()
        frame_id_to_ts = {"frame_0000": 0, "frame_0001": 1_000_000}

        temporal = provider._determine_temporal_source(
            ["frame_0000", "frame_0001"], frame_id_to_ts, True
        )
        assert temporal == TemporalSource.FRAME_MAPPED


class TestV2SummarizeParserParity:
    """V2-13: Summarize parser parity"""

    def test_summarize_parser_same_temporal_logic(self):
        provider = ExistingSnipraVLMProvider()
        frame_id_to_ts = {"frame_0000": 0, "frame_0001": 1_000_000}

        temporal = provider._determine_temporal_source(
            ["frame_0000", "frame_0001"], frame_id_to_ts, True
        )
        assert temporal == TemporalSource.FRAME_MAPPED


class TestV2FramePriorityOverModelOffset:
    """V2-14: Valid frame IDs take priority over model offset"""

    def test_frame_priority_over_offset(self):
        provider = ExistingSnipraVLMProvider()
        frame_id_to_ts = {"frame_0000": 0, "frame_0001": 1_000_000}

        # Both valid frames AND explicit offset → FRAME_MAPPED (priority 1)
        temporal = provider._determine_temporal_source(
            ["frame_0000", "frame_0001"], frame_id_to_ts, True
        )
        assert temporal == TemporalSource.FRAME_MAPPED


class TestV2ObjectRealTimestamps:
    """V2-15: Object with real timestamps"""

    def test_object_with_real_timestamps(self):
        obj = VideoObjectEvidence(
            asset_id="a",
            label="car",
            timestamps_us=[1_000_000, 2_000_000],
            confidence=0.8,
            nature=EvidenceNature.OBSERVATION,
            provider="p",
            model="m",
            first_seen_us=1_000_000,
            last_seen_us=2_000_000,
        )

        ev = _object_to_evidence(obj, "a", "p", "m", "via_v2", False, [])
        assert ev.start_us == 1_000_000
        assert ev.end_us == 2_000_000


class TestV2ObjectProvenanceWindow:
    """V2-16: Object with provenance analysis window"""

    def test_object_provenance_window(self):
        prov = EvidenceProvenance(
            provider="p",
            model="m",
            frame_ids=[],
            frame_timestamps_us=[],
            analysis_window_start_us=0,
            analysis_window_end_us=10_000_000,
            temporal_source=TemporalSource.CHUNK_WINDOW,
            confidence_source=ConfidenceSource.UNKNOWN,
            nature_source=NatureSource.UNKNOWN,
        )
        obj = VideoObjectEvidence(
            asset_id="a",
            label="car",
            timestamps_us=[],
            confidence=None,
            nature=EvidenceNature.UNKNOWN,
            provider="p",
            model="m",
            provenance=prov,
        )

        ev = _object_to_evidence(obj, "a", "p", "m", "via_v2", False, [])
        assert ev.start_us == 0
        assert ev.end_us == 10_000_000
        assert ev.provenance_temporal_source == TemporalSource.CHUNK_WINDOW


class TestV2ObjectVIAResultWindow:
    """V2-17: Object with VIA result window fallback"""

    def test_object_via_result_fallback(self):

        obj = VideoObjectEvidence(
            asset_id="a",
            label="car",
            timestamps_us=[],
            confidence=None,
            nature=EvidenceNature.UNKNOWN,
            provider="p",
            model="m",
            provenance=None,
        )
        ev = _object_to_evidence(
            obj,
            "a",
            "p",
            "m",
            "via_v2",
            False,
            [],
            analysis_window_start_us=5_000_000,
            analysis_window_end_us=15_000_000,
        )
        assert ev.start_us == 5_000_000
        assert ev.end_us == 15_000_000
        assert ev.provenance_temporal_source == TemporalSource.CHUNK_WINDOW


class TestV2ObjectNoZeroZero:
    """V2-18: Object without timestamps never uses 0/0"""

    def test_object_no_zero_zero(self):

        # Case A: with analysis window supplied
        obj = VideoObjectEvidence(
            asset_id="a",
            label="car",
            timestamps_us=[],
            confidence=None,
            nature=EvidenceNature.UNKNOWN,
            provider="p",
            model="m",
            provenance=None,
        )
        ev = _object_to_evidence(
            obj,
            "a",
            "p",
            "m",
            "via_v2",
            False,
            [],
            analysis_window_start_us=5_000_000,
            analysis_window_end_us=15_000_000,
        )
        assert ev.start_us == 5_000_000
        assert ev.end_us == 15_000_000

        # Case B: no timestamps, no provenance, no analysis window -> should raise ValueError
        obj_no_window = VideoObjectEvidence(
            asset_id="a",
            label="car",
            timestamps_us=[],
            confidence=None,
            nature=EvidenceNature.UNKNOWN,
            provider="p",
            model="m",
            provenance=None,
        )
        with pytest.raises(ValueError, match="no temporal grounding"):
            _object_to_evidence(obj_no_window, "a", "p", "m", "via_v2", False, [])


class TestV2CacheContractV1V2Isolation:
    """V2-19: Cache v1/v2 isolation"""

    def test_v1_v2_cache_isolation(self):
        from app.video_intelligence.cache import compute_cache_key
        from app.video_intelligence.models import VideoAnalysisConfig

        req = VideoIntelligenceRequest(
            asset_id="a",
            start_us=0,
            end_us=10_000_000,
            mode="ask_video",
            question="test",
            analysis_config=VideoAnalysisConfig(),
        )
        key_v1 = compute_cache_key(req, contract_version="via_v1")
        key_v2 = compute_cache_key(req, contract_version="via_v2")
        assert key_v1 != key_v2


class TestV2CacheRoundtrip:
    """V2-20: Cache roundtrip with same key"""

    def test_cache_roundtrip(self):
        from app.video_intelligence.cache import cache_result, cached_result, compute_cache_key

        req = VideoIntelligenceRequest(
            asset_id="a",
            start_us=0,
            end_us=10_000_000,
            mode="ask_video",
            question="test",
        )
        key = compute_cache_key(
            req,
            contract_version="via_v2",
            provider="openai",
            model="gpt-4o-mini",
            frame_set_fingerprint="test",
        )
        result = VideoIntelligenceResult(
            asset_id="a",
            start_us=0,
            end_us=10_000_000,
            mode="ask_video",
            question="test",
            metadata=VideoProcessingMetadata(
                provider="openai", model="gpt-4o-mini"
            ),
            contract_version="via_v2",
        )
        cache_result(key, result.model_dump())
        cached = cached_result(key)
        assert cached is not None
        assert cached["contract_version"] == "via_v2"


class TestV2PlanFingerprintIsolation:
    """V2-21: Plan fingerprint isolation"""

    def test_different_plan_different_fingerprint(self):
        from app.video_intelligence.chunking import plan_analysis

        # Different modes produce different strategies
        req1 = VideoIntelligenceRequest(
            asset_id="a",
            start_us=0,
            end_us=60_000_000,
            mode="ask_video",
            question="what happens?",
        )
        req2 = VideoIntelligenceRequest(
            asset_id="a",
            start_us=0,
            end_us=60_000_000,
            mode="summarize_video",
            objective="test",
        )
        plan1 = plan_analysis(req1)
        plan2 = plan_analysis(req2)
        # ask_video -> TARGETED_SCAN, summarize_video -> COARSE_SCAN
        # They may have different chunking strategies
        assert plan1.strategy != plan2.strategy


class TestV2ProviderModelIsolation:
    """V2-22: Provider/model cache isolation"""

    def test_different_provider_different_key(self):
        from app.video_intelligence.cache import compute_cache_key
        from app.video_intelligence.models import (
            VideoAnalysisConfig,
        )

        req = VideoIntelligenceRequest(
            asset_id="a",
            start_us=0,
            end_us=10_000_000,
            mode="ask_video",
            question="test",
            analysis_config=VideoAnalysisConfig(),
        )
        key1 = compute_cache_key(
            req,
            contract_version="via_v2",
            provider="openai",
            model="gpt-4o-mini",
            frame_set_fingerprint="f1",
        )
        key2 = compute_cache_key(
            req,
            contract_version="via_v2",
            provider="nvidia",
            model="llama",
            frame_set_fingerprint="f1",
        )
        key3 = compute_cache_key(
            req,
            contract_version="via_v2",
            provider="openai",
            model="gpt-4o",
            frame_set_fingerprint="f1",
        )
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3


@pytest.mark.parametrize("parser_name", list(_REAL_PARSERS))
class TestV2RealParserIntegration:
    """V2-23..V2-28: the three real parsers, driven with raw JSON, must all
    apply identical VIA v2 temporal grounding (sections 12-13)."""

    def test_mixed_frame_ids_frame_mapped(self, parser_name):
        ts, frame_ids, start_us, end_us, _ = _REAL_PARSERS[parser_name](
            {
                "description": "d",
                "frame_ids": ["frame_0000", "frame_9999"],
                "nature": "observation",
            },
            _frames(),
        )
        assert ts == TemporalSource.FRAME_MAPPED
        assert frame_ids == ["frame_0000"]
        assert (start_us, end_us) == (KNOWN_FRAME_TS, KNOWN_FRAME_TS)

    def test_malformed_temporal_falls_to_full_window(self, parser_name):
        ts, _frame_ids, start_us, end_us, _ = _REAL_PARSERS[parser_name](
            {"description": "d", "start": "banana", "end": "", "nature": "observation"},
            _frames(),
        )
        assert ts == TemporalSource.CHUNK_WINDOW
        assert (start_us, end_us) == (FLOOR_US, CEIL_US)

    def test_model_grounded_uses_parsed_offsets(self, parser_name):
        ts, _frame_ids, start_us, end_us, _ = _REAL_PARSERS[parser_name](
            {"description": "d", "start": "1.5s", "end": "2.0s", "nature": "observation"},
            _frames(),
        )
        assert ts == TemporalSource.MODEL_GROUNDED
        assert (start_us, end_us) == (FLOOR_US + 1_500_000, FLOOR_US + 2_000_000)

    def test_only_one_bound_is_not_model_grounded(self, parser_name):
        ts, _frame_ids, start_us, end_us, _ = _REAL_PARSERS[parser_name](
            {"description": "d", "start": "1.5s", "nature": "observation"},
            _frames(),
        )
        assert ts == TemporalSource.CHUNK_WINDOW
        assert (start_us, end_us) == (FLOOR_US, CEIL_US)

    def test_nan_infinity_are_not_valid_grounding(self, parser_name):
        ts, _frame_ids, start_us, end_us, _ = _REAL_PARSERS[parser_name](
            {"description": "d", "start": "nan", "end": "inf", "nature": "observation"},
            _frames(),
        )
        assert ts == TemporalSource.CHUNK_WINDOW
        assert (start_us, end_us) == (FLOOR_US, CEIL_US)

    def test_missing_nature_becomes_unknown(self, parser_name):
        *_rest, nature_source = _REAL_PARSERS[parser_name](
            {"description": "d"}, _frames()
        )
        assert nature_source == NatureSource.UNKNOWN


class TestV2ScenesRealParserGrounding:
    """V2-29: scenes in _parse_chunk_response follow the same rule as
    observations - malformed model offsets fall back to the full window."""

    def test_scene_malformed_offset_full_window(self):
        raw = json.dumps(
            {
                "observations": [],
                "events": [],
                "scenes": [
                    {
                        "description": "s",
                        "start": "banana",
                        "end": "3.0s",
                        "nature": "observation",
                    }
                ],
                "objects": [],
            }
        )
        chunk = {"start_us": FLOOR_US, "end_us": CEIL_US}
        out = _mk_provider()._parse_chunk_response(_request(), chunk, raw, _frames())
        scene = out["scenes"][0]
        assert scene["provenance"].temporal_source == TemporalSource.CHUNK_WINDOW
        assert scene["start_us"] == FLOOR_US
        assert scene["end_us"] == CEIL_US


class TestV2ObjectTemporalSourcePreserved:
    """V2-30: an object carrying explicit provenance.temporal_source keeps it -
    a bare timestamps_us list must NOT be auto-promoted to FRAME_MAPPED
    (section 10)."""

    def test_model_grounded_provenance_survives_timestamps(self):
        prov = EvidenceProvenance(
            provider="p",
            model="m",
            frame_ids=[],
            frame_timestamps_us=[],
            analysis_window_start_us=0,
            analysis_window_end_us=10_000_000,
            temporal_source=TemporalSource.MODEL_GROUNDED,
            confidence_source=ConfidenceSource.UNKNOWN,
            nature_source=NatureSource.UNKNOWN,
        )
        obj = VideoObjectEvidence(
            asset_id="a",
            label="car",
            timestamps_us=[1_000_000, 2_000_000],
            confidence=None,
            nature=EvidenceNature.UNKNOWN,
            provider="p",
            model="m",
            provenance=prov,
        )
        ev = _object_to_evidence(obj, "a", "p", "m", "via_v2", False, [])
        assert ev.start_us == 1_000_000
        assert ev.end_us == 2_000_000
        assert ev.provenance_temporal_source == TemporalSource.MODEL_GROUNDED


class TestV2ParseChunkResponseObjectMalformedTimestamps:
    """V2-31: malformed / non-finite object timestamps are dropped, never
    coerced into a synthetic floor_us point."""

    def test_malformed_object_timestamps_dropped(self):
        raw = json.dumps(
            {
                "observations": [],
                "events": [],
                "scenes": [],
                "objects": [
                    {
                        "label": "car",
                        "timestamps_us": ["1.0s", "banana", "nan"],
                        "nature": "observation",
                    }
                ],
            }
        )
        chunk = {"start_us": FLOOR_US, "end_us": CEIL_US}
        out = _mk_provider()._parse_chunk_response(_request(), chunk, raw, _frames())
        obj = out["objects"][0]
        assert obj["timestamps_us"] == [FLOOR_US + 1_000_000]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])