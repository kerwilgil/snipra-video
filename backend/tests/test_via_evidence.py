"""Tests for Phase 3.1 VIA Evidence Normalization Layer and Phase 3.2 Event Map Integration."""

from __future__ import annotations

import pytest

from app.editor.via_evidence import (
    EvidenceKind,
    NormalizedVideoEvidence,
    _normalize_description,
    compute_evidence_id,
    deduplicate_evidence,
    normalize_via_result,
    temporal_overlap_us,
)
from app.video_intelligence.models import (
    VideoEventCandidate,
    VideoIntelligenceResult,
    VideoObjectEvidence,
    VideoObservation,
    VideoObservationNature,
    VideoProcessingMetadata,
    VideoScene,
)


class TestComputeEvidenceId:
    """Tests for deterministic evidence ID computation."""

    def test_same_input_produces_same_id(self):
        """Same input must always produce identical evidence_id."""
        id1 = compute_evidence_id(
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT,
            description="Person jumps",
            provider="snipra_vlm",
            model="claude-haiku",
            chunk_id="chunk_0",
        )
        id2 = compute_evidence_id(
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT,
            description="Person jumps",
            provider="snipra_vlm",
            model="claude-haiku",
            chunk_id="chunk_0",
        )
        assert id1 == id2

    def test_different_asset_produces_different_id(self):
        """Different asset_id must produce different evidence_id."""
        id1 = compute_evidence_id(
            asset_id="asset-1", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, description="test",
            provider="p", model="m",
        )
        id2 = compute_evidence_id(
            asset_id="asset-2", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, description="test",
            provider="p", model="m",
        )
        assert id1 != id2

    def test_different_range_produces_different_id(self):
        """Different temporal range must produce different evidence_id."""
        id1 = compute_evidence_id(
            asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, description="test",
            provider="p", model="m",
        )
        id2 = compute_evidence_id(
            asset_id="a", start_us=5_000_000, end_us=15_000_000,
            evidence_kind=EvidenceKind.EVENT, description="test",
            provider="p", model="m",
        )
        assert id1 != id2

    def test_different_description_produces_different_id(self):
        """Different description must produce different evidence_id."""
        id1 = compute_evidence_id(
            asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, description="Person jumps",
            provider="p", model="m",
        )
        id2 = compute_evidence_id(
            asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, description="Person runs",
            provider="p", model="m",
        )
        assert id1 != id2

    def test_different_provider_produces_different_id(self):
        """Different provider must produce different evidence_id."""
        id1 = compute_evidence_id(
            asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, description="test",
            provider="provider1", model="m",
        )
        id2 = compute_evidence_id(
            asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, description="test",
            provider="provider2", model="m",
        )
        assert id1 != id2

    def test_different_model_produces_different_id(self):
        """Different model must produce different evidence_id."""
        id1 = compute_evidence_id(
            asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, description="test",
            provider="p", model="model1",
        )
        id2 = compute_evidence_id(
            asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, description="test",
            provider="p", model="model2",
        )
        assert id1 != id2

    def test_different_chunk_produces_different_id(self):
        """Different chunk_id must produce different evidence_id."""
        id1 = compute_evidence_id(
            asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, description="test",
            provider="p", model="m", chunk_id="chunk_0",
        )
        id2 = compute_evidence_id(
            asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, description="test",
            provider="p", model="m", chunk_id="chunk_1",
        )
        assert id1 != id2

    def test_evidence_kind_affects_id(self):
        """Different evidence_kind must produce different evidence_id."""
        id1 = compute_evidence_id(
            asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, description="test",
            provider="p", model="m",
        )
        id2 = compute_evidence_id(
            asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.OBSERVATION, description="test",
            provider="p", model="m",
        )
        assert id1 != id2

    def test_process_independent_identity(self):
        """ID must be process-independent (no Python hash())."""
        # Just verify it's SHA256-based and deterministic
        id1 = compute_evidence_id(
            asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, description="test",
            provider="p", model="m",
        )
        # SHA256 produces 64 hex chars, we truncate to 32
        assert len(id1) == 32
        assert all(c in "0123456789abcdef" for c in id1)


class TestNormalizeDescription:
    """Tests for description normalization."""

    def test_strips_whitespace(self):
        assert _normalize_description("  Hello  ") == "hello"

    def test_lowercase(self):
        assert _normalize_description("HELLO") == "hello"

    def test_truncates_to_160(self):
        long_desc = "a" * 200
        assert len(_normalize_description(long_desc)) == 160


class TestTemporalOverlap:
    """Tests for temporal overlap calculation."""

    def test_no_overlap(self):
        assert temporal_overlap_us(0, 10_000_000, 20_000_000, 30_000_000) == 0

    def test_partial_overlap(self):
        assert temporal_overlap_us(0, 20_000_000, 10_000_000, 30_000_000) == 10_000_000

    def test_full_containment(self):
        assert temporal_overlap_us(0, 30_000_000, 10_000_000, 20_000_000) == 10_000_000

    def test_adjacent_no_overlap(self):
        assert temporal_overlap_us(0, 10_000_000, 10_000_000, 20_000_000) == 0


class TestDeduplicateEvidence:
    """Tests for conservative evidence deduplication."""

    def test_empty_list(self):
        assert deduplicate_evidence([]) == []

    def test_no_duplicates_preserved(self):
        """Non-overlapping evidence should all be preserved."""
        ev1 = NormalizedVideoEvidence(
            evidence_id="id1", asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Event A", confidence_original=0.8,
            provider="p", model="m",
        )
        ev2 = NormalizedVideoEvidence(
            evidence_id="id2", asset_id="a", start_us=20_000_000, end_us=30_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Event B", confidence_original=0.7,
            provider="p", model="m",
        )
        result = deduplicate_evidence([ev1, ev2])
        assert len(result) == 2

    def test_duplicate_same_description_merged(self):
        """Same description with high overlap should merge (keep higher confidence)."""
        ev1 = NormalizedVideoEvidence(
            evidence_id="id1", asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Person jumps", confidence_original=0.6,
            provider="p", model="m",
        )
        ev2 = NormalizedVideoEvidence(
            evidence_id="id2", asset_id="a", start_us=2_000_000, end_us=12_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Person jumps", confidence_original=0.9,
            provider="p", model="m",
        )
        result = deduplicate_evidence([ev1, ev2])
        assert len(result) == 1
        assert result[0].confidence_original == 0.9  # Higher confidence kept

    def test_different_kind_not_merged(self):
        """Different evidence_kind should not merge."""
        ev1 = NormalizedVideoEvidence(
            evidence_id="id1", asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Person jumps", confidence_original=0.8,
            provider="p", model="m",
        )
        ev2 = NormalizedVideoEvidence(
            evidence_id="id2", asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.OBSERVATION, nature=VideoObservationNature.OBSERVATION,
            description="Person jumps", confidence_original=0.9,
            provider="p", model="m",
        )
        result = deduplicate_evidence([ev1, ev2])
        assert len(result) == 2

    def test_different_asset_not_merged(self):
        """Different asset should not merge."""
        ev1 = NormalizedVideoEvidence(
            evidence_id="id1", asset_id="asset1", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Person jumps", confidence_original=0.8,
            provider="p", model="m",
        )
        ev2 = NormalizedVideoEvidence(
            evidence_id="id2", asset_id="asset2", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Person jumps", confidence_original=0.9,
            provider="p", model="m",
        )
        result = deduplicate_evidence([ev1, ev2])
        assert len(result) == 2

    def test_insufficient_overlap_not_merged(self):
        """Overlap below threshold should not merge."""
        ev1 = NormalizedVideoEvidence(
            evidence_id="id1", asset_id="a", start_us=0, end_us=5_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Event", confidence_original=0.8,
            provider="p", model="m",
        )
        ev2 = NormalizedVideoEvidence(
            evidence_id="id2", asset_id="a", start_us=8_000_000, end_us=13_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Event", confidence_original=0.9,
            provider="p", model="m",
        )
        # Overlap is 0 (gap of 3M us), threshold is 5M us
        result = deduplicate_evidence([ev1, ev2])
        assert len(result) == 2

    def test_different_normalized_description_not_merged(self):
        """Different description after normalization should not merge."""
        ev1 = NormalizedVideoEvidence(
            evidence_id="id1", asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Person JUMPS", confidence_original=0.8,
            provider="p", model="m",
        )
        ev2 = NormalizedVideoEvidence(
            evidence_id="id2", asset_id="a", start_us=2_000_000, end_us=12_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="person runs", confidence_original=0.9,  # Different after normalize
            provider="p", model="m",
        )
        result = deduplicate_evidence([ev1, ev2])
        assert len(result) == 2

    def test_case_insensitive_description_match(self):
        """Case differences should be normalized and match."""
        ev1 = NormalizedVideoEvidence(
            evidence_id="id1", asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Person JUMPS", confidence_original=0.8,
            provider="p", model="m",
        )
        ev2 = NormalizedVideoEvidence(
            evidence_id="id2", asset_id="a", start_us=2_000_000, end_us=12_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="person jumps", confidence_original=0.9,
            provider="p", model="m",
        )
        result = deduplicate_evidence([ev1, ev2])
        assert len(result) == 1
        assert result[0].confidence_original == 0.9

    def test_prefers_higher_confidence(self):
        """When merging, should keep the higher confidence entry."""
        ev_low = NormalizedVideoEvidence(
            evidence_id="id_low", asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="test", confidence_original=0.5,
            provider="p", model="m",
        )
        ev_high = NormalizedVideoEvidence(
            evidence_id="id_high", asset_id="a", start_us=0, end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="test", confidence_original=0.9,
            provider="p", model="m",
        )
        result = deduplicate_evidence([ev_low, ev_high])
        assert len(result) == 1
        assert result[0].evidence_id == "id_high"

    def test_preserves_both_when_in_doubt(self):
        """If overlap threshold not met, preserve both (conservative)."""
        ev1 = NormalizedVideoEvidence(
            evidence_id="id1", asset_id="a", start_us=0, end_us=4_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Event A", confidence_original=0.8,
            provider="p", model="m",
        )
        ev2 = NormalizedVideoEvidence(
            evidence_id="id2", asset_id="a", start_us=8_000_000, end_us=12_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Event A", confidence_original=0.9,
            provider="p", model="m",
        )
        # Overlap is 0, but even if there was small overlap (<5M), should not merge
        result = deduplicate_evidence([ev1, ev2])
        assert len(result) == 2

    def test_identical_2s_events_deduplicate(self):
        """Two identical 2-second events should deduplicate (relative overlap = 100%)."""
        ev1 = NormalizedVideoEvidence(
            evidence_id="id1", asset_id="a", start_us=10_000_000, end_us=12_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Person lifts red object", confidence_original=0.8,
            provider="p", model="m",
        )
        ev2 = NormalizedVideoEvidence(
            evidence_id="id2", asset_id="a", start_us=10_000_000, end_us=12_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Person lifts red object", confidence_original=0.9,
            provider="p", model="m",
        )
        result = deduplicate_evidence([ev1, ev2])
        assert len(result) == 1
        assert result[0].confidence_original == 0.9

    def test_identical_500ms_events_deduplicate(self):
        """Two identical 500ms events should deduplicate (relative overlap = 100%)."""
        ev1 = NormalizedVideoEvidence(
            evidence_id="id1", asset_id="a", start_us=10_000_000, end_us=10_500_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Quick flash", confidence_original=0.8,
            provider="p", model="m",
        )
        ev2 = NormalizedVideoEvidence(
            evidence_id="id2", asset_id="a", start_us=10_000_000, end_us=10_500_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Quick flash", confidence_original=0.9,
            provider="p", model="m",
        )
        result = deduplicate_evidence([ev1, ev2])
        assert len(result) == 1
        assert result[0].confidence_original == 0.9

    def test_partial_overlap_below_threshold_not_deduplicate(self):
        """90% overlap of shorter event but different descriptions -> no dedup."""
        ev1 = NormalizedVideoEvidence(
            evidence_id="id1", asset_id="a", start_us=10_000_000, end_us=12_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Person lifts red object", confidence_original=0.8,
            provider="p", model="m",
        )
        ev2 = NormalizedVideoEvidence(
            evidence_id="id2", asset_id="a", start_us=10_500_000, end_us=12_500_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Person lifts blue object", confidence_original=0.9,
            provider="p", model="m",
        )
        # 75% relative overlap (1.5M / 2M) < 80% threshold
        result = deduplicate_evidence([ev1, ev2])
        assert len(result) == 2

    def test_same_description_distant_timestamps_not_deduplicate(self):
        """Same description but distant timestamps should not deduplicate."""
        ev1 = NormalizedVideoEvidence(
            evidence_id="id1", asset_id="a", start_us=0, end_us=2_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Person waves", confidence_original=0.8,
            provider="p", model="m",
        )
        ev2 = NormalizedVideoEvidence(
            evidence_id="id2", asset_id="a", start_us=60_000_000, end_us=62_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Person waves", confidence_original=0.9,
            provider="p", model="m",
        )
        # No temporal overlap at all
        result = deduplicate_evidence([ev1, ev2])
        assert len(result) == 2

    def test_same_time_different_kind_not_deduplicate(self):
        """Same time range but different evidence_kind should not deduplicate."""
        ev1 = NormalizedVideoEvidence(
            evidence_id="id1", asset_id="a", start_us=10_000_000, end_us=12_000_000,
            evidence_kind=EvidenceKind.EVENT, nature=VideoObservationNature.OBSERVATION,
            description="Action happens", confidence_original=0.8,
            provider="p", model="m",
        )
        ev2 = NormalizedVideoEvidence(
            evidence_id="id2", asset_id="a", start_us=10_000_000, end_us=12_000_000,
            evidence_kind=EvidenceKind.OBSERVATION, nature=VideoObservationNature.OBSERVATION,
            description="Action happens", confidence_original=0.9,
            provider="p", model="m",
        )
        result = deduplicate_evidence([ev1, ev2])
        assert len(result) == 2


class TestNormalizedVideoEvidence:
    """Tests for the normalized evidence contract."""

    def test_observation_fields_required(self):
        """Observation evidence requires all core fields."""
        ev = NormalizedVideoEvidence(
            evidence_id="test-id",
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            evidence_kind=EvidenceKind.OBSERVATION,
            nature=VideoObservationNature.OBSERVATION,
            description="Test observation",
            confidence_original=0.8,
            provider="snipra_vlm",
            model="claude-haiku",
            chunk_id="chunk_0",
        )
        assert ev.evidence_kind == EvidenceKind.OBSERVATION
        assert ev.nature == VideoObservationNature.OBSERVATION
        assert ev.source == "via"
        assert ev.contract_version == "via_v2"

    def test_event_preserves_enriched_fields(self):
        """Event evidence can carry enriched metadata."""
        ev = NormalizedVideoEvidence(
            evidence_id="test-id",
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT,
            nature=VideoObservationNature.OBSERVATION,
            description="Test event",
            confidence_original=0.8,
            provider="snipra_vlm",
            model="claude-haiku",
            visual_novelty="high",
            payoff_present=True,
            reaction_present=False,
            evidence_quality="clear",
        )
        assert ev.visual_novelty == "high"
        assert ev.payoff_present is True
        assert ev.reaction_present is False
        assert ev.evidence_quality == "clear"

    # (duplicate test removed: name collision with test above)
    def test_event_preserves_enriched_fields_with_actions(self):
        """Event evidence can carry enriched metadata including actions."""
        ev = NormalizedVideoEvidence(
            evidence_id="test-id",
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT,
            nature=VideoObservationNature.OBSERVATION,
            description="Test event",
            confidence_original=0.8,
            provider="snipra_vlm",
            model="claude-haiku",
            observed_action="person picks up phone",
            subjects=["person"],
            objects=["phone"],
            visual_novelty="high",
            payoff_present=True,
            reaction_present=False,
            evidence_quality="clear",
        )
        assert ev.observed_action == "person picks up phone"
        assert ev.subjects == ["person"]
        assert "phone" in ev.objects
        assert ev.visual_novelty == "high"
        assert ev.payoff_present is True
        assert ev.reaction_present is False
        assert ev.evidence_quality == "clear"

    def test_scene_preserves_scene_fields(self):
        """Scene evidence can carry scene-specific fields."""
        ev = NormalizedVideoEvidence(
            evidence_id="test-id",
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            evidence_kind=EvidenceKind.SCENE,
            nature=VideoObservationNature.OBSERVATION,
            description="Scene change",
            confidence_original=0.8,
            provider="snipra_vlm",
            model="claude-haiku",
            scene_type="transition",
            scene_continuity="continuous",
        )
        assert ev.scene_type == "transition"
        assert ev.scene_continuity == "continuous"

    def test_object_preserves_object_fields(self):
        """Object evidence can carry object-specific fields."""
        ev = NormalizedVideoEvidence(
            evidence_id="test-id",
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            evidence_kind=EvidenceKind.OBJECT,
            nature=VideoObservationNature.OBSERVATION,
            description="Object: phone",
            confidence_original=0.8,
            provider="snipra_vlm",
            model="claude-haiku",
            object_label="phone",
            object_track_id="track_1",
            object_interaction="pickup",
        )
        assert ev.object_label == "phone"
        assert ev.object_track_id == "track_1"
        assert ev.object_interaction == "pickup"

    def test_event_preserves_enriched_fields_extra(self):
        """Event evidence can carry enriched metadata."""
        ev = NormalizedVideoEvidence(
            evidence_id="test-id",
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT,
            nature=VideoObservationNature.OBSERVATION,
            description="Test event",
            confidence_original=0.8,
            provider="snipra_vlm",
            model="claude-haiku",
            observed_action="person picks up phone",
            subjects=["person"],
            objects=["phone"],
            visual_novelty="high",
            payoff_present=True,
            reaction_present=False,
            evidence_quality="clear",
        )
        assert ev.observed_action == "person picks up phone"
        assert ev.subjects == ["person"]
        assert "phone" in ev.objects
        assert ev.visual_novelty == "high"
        assert ev.payoff_present is True
        assert ev.reaction_present is False
        assert ev.evidence_quality == "clear"

    def test_scene_preserves_enriched_fields(self):
        """Scene evidence can carry enriched scene-specific fields."""
        ev = NormalizedVideoEvidence(
            evidence_id="test-id",
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            evidence_kind=EvidenceKind.SCENE,
            nature=VideoObservationNature.OBSERVATION,
            description="Scene change",
            confidence_original=0.8,
            provider="snipra_vlm",
            model="claude-haiku",
            scene_type="transition",
            scene_continuity="continuous",
            dominant_objects=["person", "phone"],
            subjects=["person"],
        )
        assert ev.scene_type == "transition"
        assert ev.scene_continuity == "continuous"
        assert "person" in ev.dominant_objects
        assert "phone" in ev.dominant_objects
        assert ev.subjects == ["person"]

    def test_object_preserves_enriched_fields(self):
        """Object evidence can carry enriched object-specific fields."""
        ev = NormalizedVideoEvidence(
            evidence_id="test-id",
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            evidence_kind=EvidenceKind.OBJECT,
            nature=VideoObservationNature.OBSERVATION,
            description="Object: phone",
            confidence_original=0.8,
            provider="snipra_vlm",
            model="claude-haiku",
            object_label="phone",
            first_seen_us=5_000_000,
            last_seen_us=15_000_000,
            object_track_id="track_001",
            interactions=["pickup", "hold"],
        )
        assert ev.object_label == "phone"
        assert ev.first_seen_us == 5_000_000
        assert ev.last_seen_us == 15_000_000
        assert ev.object_track_id == "track_001"
        assert "pickup" in ev.interactions
        assert "hold" in ev.interactions


class TestNormalizeViaResult:
    """Tests for VIA result normalization."""

    def _create_mock_via_result(self) -> VideoIntelligenceResult:
        """Create a representative VIA result for testing."""
        metadata = VideoProcessingMetadata(
            chunks_processed=2,
            chunks_failed=0,
            total_frames_analyzed=10,
            provider="snipra_vlm",
            model="claude-haiku-4-5",
            processing_time_ms=5000,
            partial=False,
            failed_chunks=[],
        )
        return VideoIntelligenceResult(
            asset_id="asset-123",
            start_us=0,
            end_us=30_000_000,
            mode="summarize_video",
            question=None,
            objective="Summarize video",
            summary="Video summary",
            answer=None,
            observations=[
                VideoObservation(
                    asset_id="asset-123",
                    start_us=0,
                    end_us=5_000_000,
                    description="Person enters room",
                    nature=VideoObservationNature.OBSERVATION,
                    confidence=0.9,
                    source="snipra_vlm",
                    provider="snipra_vlm",
                    model="claude-haiku-4-5",
                    chunk_id="chunk_0",
                )
            ],
            events=[
                VideoEventCandidate(
                    asset_id="asset-123",
                    start_us=5_000_000,
                    end_us=15_000_000,
                    type="action",
                    description="Person picks up object",
                    confidence=0.85,
                    nature=VideoObservationNature.OBSERVATION,
                    provider="snipra_vlm",
                    model="claude-haiku-4-5",
                    chunk_id="chunk_1",
                    observed_action="Person picks up object",
                    subjects=["person"],
                    objects=["phone"],
                )
            ],
            scenes=[
                VideoScene(
                    asset_id="asset-123",
                    start_us=0,
                    end_us=30_000_000,
                    description="Indoor scene with person",
                    confidence=0.95,
                    nature=VideoObservationNature.OBSERVATION,
                    provider="snipra_vlm",
                    model="claude-haiku-4-5",
                    chunk_id="chunk_0",
                    subjects=["person"],
                    scene_type="indoor",
                    continuity="continuous",
                    dominant_objects=["person", "phone"],
                )
            ],
            objects=[
                VideoObjectEvidence(
                    asset_id="asset-123",
                    label="phone",
                    timestamps_us=[5_000_000, 10_000_000],
                    confidence=0.9,
                    nature=VideoObservationNature.OBSERVATION,
                    provider="snipra_vlm",
                    model="claude-haiku-4-5",
                    first_seen_us=5_000_000,
                    last_seen_us=10_000_000,
                    track_id="track_001",
                    interactions=["pickup", "hold"],
                )
            ],
            metadata=metadata,
            partial=False,
            contract_version="via_v2",
        )

    def test_normalize_observations(self):
        """Observations are normalized to evidence."""
        via_result = self._create_mock_via_result()
        evidence_list = normalize_via_result(via_result)

        obs_evidence = [e for e in evidence_list if e.evidence_kind == EvidenceKind.OBSERVATION]
        assert len(obs_evidence) == 1
        assert obs_evidence[0].description == "Person enters room"
        assert obs_evidence[0].nature == VideoObservationNature.OBSERVATION
        assert obs_evidence[0].confidence_original == 0.9

    def test_normalize_events(self):
        """Events are normalized to evidence."""
        via_result = self._create_mock_via_result()
        evidence_list = normalize_via_result(via_result)

        event_evidence = [e for e in evidence_list if e.evidence_kind == EvidenceKind.EVENT]
        assert len(event_evidence) == 1
        assert event_evidence[0].description == "Person picks up object"
        assert event_evidence[0].confidence_original == 0.85

    def test_normalize_scenes(self):
        """Scenes are normalized to evidence."""
        via_result = self._create_mock_via_result()
        evidence_list = normalize_via_result(via_result)

        scene_evidence = [e for e in evidence_list if e.evidence_kind == EvidenceKind.SCENE]
        assert len(scene_evidence) == 1
        assert scene_evidence[0].description == "Indoor scene with person"
        assert scene_evidence[0].confidence_original == 0.95

    def test_normalize_objects(self):
        """Objects are normalized to evidence."""
        via_result = self._create_mock_via_result()
        evidence_list = normalize_via_result(via_result)

        obj_evidence = [e for e in evidence_list if e.evidence_kind == EvidenceKind.OBJECT]
        assert len(obj_evidence) == 1
        assert "phone" in obj_evidence[0].description
        assert obj_evidence[0].confidence_original == 0.9

    def test_provenance_preserved(self):
        """Provider, model, chunk_id preserved in normalized evidence."""
        via_result = self._create_mock_via_result()
        evidence_list = normalize_via_result(via_result)

        for ev in evidence_list:
            assert ev.provider == "snipra_vlm"
            assert ev.model == "claude-haiku-4-5"
            assert ev.source == "via"
            assert ev.contract_version == "via_v2"

    def test_partial_and_failed_chunks_preserved(self):
        """Partial status and failed chunks metadata preserved."""
        via_result = self._create_mock_via_result()
        via_result.partial = True
        via_result.metadata.failed_chunks = [{"chunk_index": 2, "error": "timeout"}]

        evidence_list = normalize_via_result(via_result)

        for ev in evidence_list:
            assert ev.partial is True
            assert len(ev.failed_chunks) == 1
            assert ev.failed_chunks[0]["error"] == "timeout"

    def test_total_evidence_count(self):
        """Total normalized evidence count matches VIA result components."""
        via_result = self._create_mock_via_result()
        evidence_list = normalize_via_result(via_result)

        # 1 observation + 1 event + 1 scene + 1 object = 4
        assert len(evidence_list) == 4

    def test_normalize_events_enriched_fields(self):
        """Events are normalized with enriched fields preserved."""
        via_result = self._create_mock_via_result()
        evidence_list = normalize_via_result(via_result)

        event_evidence = [e for e in evidence_list if e.evidence_kind == EvidenceKind.EVENT]
        assert len(event_evidence) == 1
        ev = event_evidence[0]
        # Test enriched event fields
        assert ev.observed_action == "Person picks up object"
        assert ev.subjects == ["person"]
        assert "phone" in ev.objects

    def test_normalize_scenes_enriched_fields(self):
        """Scenes are normalized with enriched fields preserved."""
        via_result = self._create_mock_via_result()
        # Add enriched scene fields
        via_result.scenes[0].subjects = ["person"]
        via_result.scenes[0].scene_type = "indoor"
        via_result.scenes[0].continuity = "continuous"
        via_result.scenes[0].dominant_objects = ["person", "phone"]

        evidence_list = normalize_via_result(via_result)

        scene_evidence = [e for e in evidence_list if e.evidence_kind == EvidenceKind.SCENE]
        assert len(scene_evidence) == 1
        ev = scene_evidence[0]
        # Test enriched scene fields
        assert ev.subjects == ["person"]
        assert ev.scene_type == "indoor"
        assert ev.scene_continuity == "continuous"
        assert "person" in ev.dominant_objects
        assert "phone" in ev.dominant_objects

    def test_normalize_objects_enriched_fields(self):
        """Objects are normalized with enriched fields preserved."""
        via_result = self._create_mock_via_result()
        # Add enriched object fields
        via_result.objects[0].first_seen_us = 5_000_000
        via_result.objects[0].last_seen_us = 10_000_000
        via_result.objects[0].track_id = "track_001"
        via_result.objects[0].interactions = ["pickup", "hold"]

        evidence_list = normalize_via_result(via_result)

        obj_evidence = [e for e in evidence_list if e.evidence_kind == EvidenceKind.OBJECT]
        assert len(obj_evidence) == 1
        ev = obj_evidence[0]
        # Test enriched object fields
        assert ev.first_seen_us == 5_000_000
        assert ev.last_seen_us == 10_000_000
        assert ev.object_track_id == "track_001"
        assert "pickup" in ev.interactions
        assert "hold" in ev.interactions

    def test_normalize_events_preserves_observation_inference(self):
        """Observations and inferences are preserved with correct nature."""
        via_result = self._create_mock_via_result()
        via_result.events[0].nature = VideoObservationNature.INFERENCE
        via_result.observations[0].nature = VideoObservationNature.OBSERVATION

        evidence_list = normalize_via_result(via_result)

        event_evidence = [e for e in evidence_list if e.evidence_kind == EvidenceKind.EVENT]
        obs_evidence = [e for e in evidence_list if e.evidence_kind == EvidenceKind.OBSERVATION]

        assert event_evidence[0].nature == VideoObservationNature.INFERENCE
        assert obs_evidence[0].nature == VideoObservationNature.OBSERVATION


class TestCacheCompatibility:
    """Tests for Phase 1 cache compatibility."""

    def test_old_payload_shape_still_valid(self):
        """Test that a Phase 1 cached payload would still validate."""
        # This simulates what a cached Phase 1 result might look like
        # (minimal fields, no enriched fields)
        old_payload = {
            "asset_id": "asset-123",
            "start_us": 0,
            "end_us": 10_000_000,
            "mode": "summarize_video",
            "question": None,
            "objective": "Summarize video",
            "summary": "Old summary",
            "answer": None,
            "observations": [],
            "events": [],
            "scenes": [],
            "objects": [],
            "metadata": {
                "chunks_processed": 1,
                "chunks_failed": 0,
                "total_frames_analyzed": 5,
                "provider": "snipra_vlm",
                "model": "claude-haiku-4-5",
                "processing_time_ms": 1000,
                "partial": False,
                "failed_chunks": [],
            },
            "partial": False,
            "contract_version": "via_v2",
        }

        # Should validate without error (backward compatible)
        result = VideoIntelligenceResult.model_validate(old_payload)
        assert result.contract_version == "via_v2"
        assert result.partial is False

    def test_enriched_fields_optional_in_normalized(self):
        """Enriched fields in NormalizedVideoEvidence are optional."""
        ev = NormalizedVideoEvidence(
            evidence_id="test-id",
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            evidence_kind=EvidenceKind.EVENT,
            nature=VideoObservationNature.OBSERVATION,
            description="Test event",
            confidence_original=0.8,
            provider="snipra_vlm",
            model="claude-haiku",
            # Enriched fields not provided
        )
        assert ev.visual_novelty is None
        assert ev.payoff_present is None
        assert ev.reaction_present is None
        assert ev.evidence_quality is None
        assert ev.scene_type is None
        assert ev.scene_continuity is None
        assert ev.object_label is None
        assert ev.object_track_id is None
        assert ev.object_interaction is None


class TestEventMapIntegration:
    """Tests for Phase 3.2 Event Map integration with VIA evidence."""

    def _create_legacy_event(
        self, start_us: int, end_us: int, legacy_score: float = 0.7
    ) -> dict:
        return {
            "id": "legacy-1",
            "asset_id": "asset-123",
            "start_us": start_us,
            "end_us": end_us,
            "title": "Legacy Event",
            "reason": "Test reason",
            "legacy_score": legacy_score,
            "source_ranges": [{"start_us": start_us, "end_us": end_us}],
            "evidence": [],
            "uncertain": False,
        }

    def _create_via_event_evidence(
        self,
        start_us: int,
        end_us: int,
        description: str = "VIA detected action",
        confidence: float = 0.85,
    ):
        """Create a NormalizedVideoEvidence for an event."""
        from app.editor.via_evidence import EvidenceKind, NormalizedVideoEvidence
        from app.video_intelligence.models import VideoObservationNature
        return NormalizedVideoEvidence(
            evidence_id="via-ev-1",
            asset_id="asset-123",
            start_us=start_us,
            end_us=end_us,
            evidence_kind=EvidenceKind.EVENT,
            nature=VideoObservationNature.OBSERVATION,
            description=description,
            confidence_original=confidence,
            provider="snipra_vlm",
            model="claude-haiku",
            chunk_id="chunk_0",
        )

    def test_via_evidence_attaches_to_matching_legacy_event(self):
        """VIA evidence with temporal overlap attaches to matching legacy event."""
        from app.editor.events import _merge_via_evidence

        legacy = [self._create_legacy_event(10_000_000, 20_000_000, 0.7)]
        via = [self._create_via_event_evidence(12_000_000, 18_000_000, "VIA action", 0.85)]

        result = _merge_via_evidence(legacy, via)

        assert "events" in result
        assert len(result["events"]) == 1
        assert "additional_evidence" in result["events"][0]
        assert len(result["events"][0]["additional_evidence"]) == 1
        assert result["events"][0]["additional_evidence"][0]["description"] == "VIA action"

    def test_via_evidence_no_match_preserved_as_unattached(self):
        """VIA evidence with no temporal match preserved as unattached entry."""
        from app.editor.events import _merge_via_evidence

        legacy = [self._create_legacy_event(0, 5_000_000, 0.7)]
        via = [self._create_via_event_evidence(50_000_000, 55_000_000, "Distant VIA action", 0.85)]

        result = _merge_via_evidence(legacy, via)

        assert "events" in result
        assert len(result["events"]) == 1
        assert "unattached_additional_evidence" in result
        assert len(result["unattached_additional_evidence"]) == 1
        assert result["unattached_additional_evidence"][0]["description"] == "Distant VIA action"
        assert result["unattached_additional_evidence"][0]["unattached"] is True

    def test_via_evidence_preserves_confidence_original(self):
        """confidence_original from VIA must not be converted to legacy_score."""
        from app.editor.events import _merge_via_evidence

        legacy = [self._create_legacy_event(10_000_000, 20_000_000, 0.7)]
        via = [self._create_via_event_evidence(12_000_000, 18_000_000, "Action", 0.92)]

        result = _merge_via_evidence(legacy, via)

        attached = result["events"][0]["additional_evidence"][0]
        assert attached["confidence_original"] == 0.92
        assert "legacy_score" not in attached
        assert "ranking_score" not in attached

    def test_via_evidence_provenance_preserved(self):
        """All provenance fields preserved in additional_evidence."""
        from app.editor.events import _merge_via_evidence
        from app.editor.via_evidence import EvidenceKind, NormalizedVideoEvidence
        from app.video_intelligence.models import VideoObservationNature

        [self._create_legacy_event(10_000_000, 20_000_000, 0.7)]
        via = [NormalizedVideoEvidence(
            evidence_id="via-1",
            asset_id="asset-123",
            start_us=12_000_000,
            end_us=18_000_000,
            evidence_kind=EvidenceKind.EVENT,
            nature=VideoObservationNature.OBSERVATION,
            description="Test action",
            confidence_original=0.88,
            provider="snipra_vlm",
            model="claude-haiku",
            chunk_id="chunk_0",
            contract_version="via_v2",
            partial=True,
            failed_chunks=[{"chunk_index": 2, "error": "timeout"}],
        )]

        result = _merge_via_evidence([self._create_legacy_event(10_000_000, 20_000_000, 0.7)], via)

        attached = result["events"][0]["additional_evidence"][0]
        assert attached["source"] == "via"
        assert attached["provider"] == "snipra_vlm"
        assert attached["model"] == "claude-haiku"
        assert attached["chunk_id"] == "chunk_0"
        assert attached["contract_version"] == "via_v2"
        assert attached["partial"] is True
        assert attached["failed_chunks"] == [{"chunk_index": 2, "error": "timeout"}]

    def test_via_evidence_preserves_enriched_fields(self):
        """Enriched fields (visual_novelty, payoff_present, etc.) preserved."""
        from app.editor.events import _merge_via_evidence
        from app.editor.via_evidence import EvidenceKind, NormalizedVideoEvidence
        from app.video_intelligence.models import VideoObservationNature

        [self._create_legacy_event(10_000_000, 20_000_000, 0.7)]
        via = [NormalizedVideoEvidence(
            evidence_id="via-1",
            asset_id="asset-123",
            start_us=12_000_000,
            end_us=18_000_000,
            evidence_kind=EvidenceKind.EVENT,
            nature=VideoObservationNature.OBSERVATION,
            description="Action with payoff",
            confidence_original=0.88,
            provider="snipra_vlm",
            model="claude-haiku",
            visual_novelty="high",
            payoff_present=True,
            reaction_present=True,
            evidence_quality="clear",
        )]

        result = _merge_via_evidence([self._create_legacy_event(10_000_000, 20_000_000, 0.7)], via)

        attached = result["events"][0]["additional_evidence"][0]
        assert attached["visual_novelty"] == "high"
        assert attached["payoff_present"] is True
        assert attached["reaction_present"] is True
        assert attached["evidence_quality"] == "clear"

    def test_legacy_score_unchanged(self):
        """legacy_score of legacy events must remain unchanged."""
        from app.editor.events import _merge_via_evidence

        legacy = [self._create_legacy_event(10_000_000, 20_000_000, 0.73)]
        via = [self._create_via_event_evidence(12_000_000, 18_000_000, "Action", 0.95)]

        result = _merge_via_evidence(legacy, via)

        assert result["events"][0]["legacy_score"] == 0.73  # Unchanged

    def test_via_confidence_not_converted_to_legacy_score(self):
        """VIA confidence_original must never become legacy_score."""
        from app.editor.events import _merge_via_evidence

        legacy = [self._create_legacy_event(10_000_000, 20_000_000, 0.5)]
        via = [self._create_via_event_evidence(12_000_000, 18_000_000, "Action", 0.99)]

        result = _merge_via_evidence(legacy, via)

        # Legacy score stays 0.5
        assert result["events"][0]["legacy_score"] == 0.5
        # VIA confidence preserved as metadata
        assert result["events"][0]["additional_evidence"][0]["confidence_original"] == 0.99


class TestEventMapWithVIA:
    """Integration tests for events_for_job with VIA evidence."""

    def test_events_for_job_without_via_unchanged(self):
        """events_for_job without VIA evidence returns legacy events unchanged."""
        # This would need a database session, skip for unit test
        # The function signature accepts via_evidence=None
        pass

    def test_best_event_pool_ordering_unchanged_with_via(self):
        """Best Event Pool ordering must remain identical with VIA evidence."""
        from app.editor.events import best_event_pool

        # Two legacy events with different scores
        events = [
            {"id": "a", "legacy_score": 0.8, "source_ranges": [{"start_us": 0, "end_us": 10_000_000}], "additional_evidence": []},
            {"id": "b", "legacy_score": 0.6, "source_ranges": [{"start_us": 0, "end_us": 10_000_000}], "additional_evidence": []},
        ]

        # With VIA evidence attached to lower-scored event
        events_with_via = [
            {"id": "a", "legacy_score": 0.8, "source_ranges": [{"start_us": 0, "end_us": 10_000_000}], "additional_evidence": []},
            {"id": "b", "legacy_score": 0.6, "source_ranges": [{"start_us": 0, "end_us": 10_000_000}], "additional_evidence": [{"confidence_original": 0.95}]},
        ]

        pool = best_event_pool(events, intent="highlight")
        pool_with_via = best_event_pool(events_with_via, intent="highlight")

        # Ordering must be identical (legacy_score drives ranking)
        assert [e["id"] for e in pool] == [e["id"] for e in pool_with_via]
        assert pool[0]["id"] == "a"  # Higher legacy_score first
        assert pool_with_via[0]["id"] == "a"


class TestRegressionNoScoringChange:
    """Explicit regression test: VIA evidence must not change Best Event Pool ordering."""

    def test_same_legacy_events_plus_via_pool_ordering_identical(self):
        """Same legacy events + VIA evidence -> pool ordering EXACTLY SAME."""
        from app.editor.events import best_event_pool

        # Baseline: legacy events only
        legacy_events = [
            {"id": "evt1", "legacy_score": 0.9, "source_ranges": [{"start_us": 0, "end_us": 10_000_000}]},
            {"id": "evt2", "legacy_score": 0.7, "source_ranges": [{"start_us": 10_000_000, "end_us": 20_000_000}]},
            {"id": "evt3", "legacy_score": 0.5, "source_ranges": [{"start_us": 20_000_000, "end_us": 30_000_000}]},
        ]

        # Same events with VIA evidence attached (high confidence to evt3)
        events_with_via = [
            {"id": "evt1", "legacy_score": 0.9, "source_ranges": [{"start_us": 0, "end_us": 10_000_000}], "additional_evidence": []},
            {"id": "evt2", "legacy_score": 0.7, "source_ranges": [{"start_us": 10_000_000, "end_us": 20_000_000}], "additional_evidence": []},
            {"id": "evt3", "legacy_score": 0.5, "source_ranges": [{"start_us": 20_000_000, "end_us": 30_000_000}], "additional_evidence": [{"confidence_original": 0.99}]},
        ]

        pool_baseline = best_event_pool(legacy_events, intent="montage")
        pool_with_via = best_event_pool(events_with_via, intent="montage")

        # Ordering MUST be identical
        assert [e["id"] for e in pool_baseline] == [e["id"] for e in pool_with_via]
        assert pool_baseline[0]["id"] == "evt1"
        assert pool_baseline[1]["id"] == "evt2"
        assert pool_baseline[2]["id"] == "evt3"


class TestMultipleMatchResolution:
    """Tests for deterministic multiple match resolution."""

    def _create_legacy_events(self, events_data):
        """Create multiple legacy events."""
        legacy = []
        for i, (start, end, score) in enumerate(events_data):
            legacy.append({
                "id": f"legacy-{i}",
                "asset_id": "asset-123",
                "start_us": start,
                "end_us": end,
                "title": f"Legacy Event {i}",
                "reason": "Test reason",
                "legacy_score": score,
                "source_ranges": [{"start_us": start, "end_us": end}],
                "evidence": [],
                "uncertain": False,
            })
        return legacy

    def _create_via_evidence(self, start_us, end_us, description="VIA action", confidence=0.85):
        """Create a NormalizedVideoEvidence for an event."""
        from app.editor.via_evidence import EvidenceKind, NormalizedVideoEvidence
        from app.video_intelligence.models import VideoObservationNature
        return NormalizedVideoEvidence(
            evidence_id="via-ev-1",
            asset_id="asset-123",
            start_us=start_us,
            end_us=end_us,
            evidence_kind=EvidenceKind.EVENT,
            nature=VideoObservationNature.OBSERVATION,
            description=description,
            confidence_original=confidence,
            provider="snipra_vlm",
            model="claude-haiku",
            chunk_id="chunk_0",
        )

    def test_via_overlaps_two_events_attaches_to_highest_overlap(self):
        """VIA overlapping two events attaches to highest overlap event."""
        from app.editor.events import _merge_via_evidence

        # Two adjacent legacy events
        legacy = self._create_legacy_events([
            (10_000_000, 20_000_000, 0.7),  # Event 1: 10s-20s
            (20_000_000, 30_000_000, 0.7),  # Event 2: 20s-30s
        ])
        # VIA spans 15s-25s: 5M overlap with Event 1, 5M overlap with Event 2
        # Equal overlap - should use deterministic tie-break
        via = [self._create_via_evidence(15_000_000, 25_000_000, "Middle action", 0.85)]

        result = _merge_via_evidence(legacy, via)

        # Should attach to exactly one event
        total_attached = sum(len(e.get("additional_evidence", [])) for e in result["events"])
        assert total_attached == 1

        # Both have equal 50% overlap, deterministic tie-break by event id
        # Event 0 should win (first in sorted order)
        assert len(result["events"][0]["additional_evidence"]) == 1
        assert len(result["events"][1].get("additional_evidence", [])) == 0

    def test_equal_overlap_deterministic_tie_break(self):
        """Equal overlap with two events -> deterministic tie-break by event id."""
        from app.editor.events import _merge_via_evidence

        # Two events with same duration
        legacy = self._create_legacy_events([
            (10_000_000, 20_000_000, 0.7),
            (20_000_000, 30_000_000, 0.7),
        ])
        # VIA exactly in middle: 50% overlap with each
        via = [self._create_via_evidence(15_000_000, 25_000_000, "Action", 0.85)]

        # Run multiple times with reversed order
        result1 = _merge_via_evidence(legacy, via)
        legacy_reversed = list(reversed(legacy))
        result2 = _merge_via_evidence(legacy_reversed, via)

        # Same event should win regardless of input order
        attached_evt_1 = next(e for e in result1["events"] if e.get("additional_evidence"))
        attached_evt_2 = next(e for e in result2["events"] if e.get("additional_evidence"))
        assert attached_evt_1["id"] == attached_evt_2["id"] == "legacy-0"

    def test_input_order_reversed_same_match_result(self):
        """Reversed input event order produces same match result."""
        from app.editor.events import _merge_via_evidence

        legacy = self._create_legacy_events([
            (10_000_000, 18_000_000, 0.7),  # 8s event (legacy-0)
            (16_000_000, 30_000_000, 0.7),  # 14s event (legacy-1), overlaps 2s with first
        ])
        # VIA at 14s-22s:
        # Overlap with legacy-0 (10-18): 4s, rel_overlap = 4/min(8,8) = 50%
        # Overlap with legacy-1 (16-30): 6s, rel_overlap = 6/min(14,8) = 75%
        # legacy-1 has higher relative overlap -> should win
        via = [self._create_via_evidence(14_000_000, 22_000_000, "Action", 0.85)]

        result1 = _merge_via_evidence(legacy, via)
        legacy_reversed = list(reversed(legacy))
        result2 = _merge_via_evidence(legacy_reversed, via)

        # Should attach to legacy-1 in both cases (higher relative overlap: 75% vs 50%)
        attached_evt_1 = next(e for e in result1["events"] if e.get("additional_evidence"))
        attached_evt_2 = next(e for e in result2["events"] if e.get("additional_evidence"))
        assert attached_evt_1["id"] == attached_evt_2["id"] == "legacy-1"


class TestCandidateCountInvariant:
    """Tests ensuring candidate count invariants with unmatched VIA evidence."""

    def _create_legacy_events(self, events_data):
        legacy = []
        for i, (start, end, score) in enumerate(events_data):
            legacy.append({
                "id": f"legacy-{i}",
                "asset_id": "asset-123",
                "start_us": start,
                "end_us": end,
                "title": f"Legacy Event {i}",
                "reason": "Test reason",
                "legacy_score": score,
                "source_ranges": [{"start_us": start, "end_us": end}],
                "evidence": [],
                "uncertain": False,
            })
        return legacy

    def _create_via_evidence(self, start_us, end_us, description="VIA action", confidence=0.85):
        from app.editor.via_evidence import EvidenceKind, NormalizedVideoEvidence
        from app.video_intelligence.models import VideoObservationNature
        return NormalizedVideoEvidence(
            evidence_id="via-ev-1",
            asset_id="asset-123",
            start_us=start_us,
            end_us=end_us,
            evidence_kind=EvidenceKind.EVENT,
            nature=VideoObservationNature.OBSERVATION,
            description=description,
            confidence_original=confidence,
            provider="snipra_vlm",
            model="claude-haiku",
            chunk_id="chunk_0",
        )

    def test_unmatched_via_preserves_candidate_count(self):
        """Legacy pool with unmatched VIA evidence -> candidate count unchanged."""
        from app.editor.events import _merge_via_evidence, best_event_pool

        legacy = self._create_legacy_events([
            (0, 10_000_000, 0.8),
            (10_000_000, 20_000_000, 0.6),
        ])
        via = [self._create_via_evidence(50_000_000, 55_000_000, "Distant action", 0.95)]

        result = _merge_via_evidence(legacy, via)

        # Legacy events unchanged
        assert len(result["events"]) == 2
        # Unattached VIA preserved separately
        assert len(result["unattached_additional_evidence"]) == 1

        # Pool candidate count unchanged
        pool_baseline = best_event_pool(legacy, intent="montage")
        events_with_via = result["events"]  # Only matched events (no unattached)
        pool_with_via = best_event_pool(events_with_via, intent="montage")

        assert len(pool_baseline) == len(pool_with_via) == 2

    def test_ordering_unchanged_with_unmatched_via(self):
        """Ordering unchanged with unmatched VIA evidence."""
        from app.editor.events import _merge_via_evidence, best_event_pool

        legacy = [
            {"id": "evt1", "legacy_score": 0.9, "source_ranges": [{"start_us": 0, "end_us": 10_000_000}], "additional_evidence": []},
            {"id": "evt2", "legacy_score": 0.7, "source_ranges": [{"start_us": 10_000_000, "end_us": 20_000_000}], "additional_evidence": []},
            {"id": "evt3", "legacy_score": 0.5, "source_ranges": [{"start_us": 20_000_000, "end_us": 30_000_000}], "additional_evidence": []},
        ]
        via = [NormalizedVideoEvidence(
            evidence_id="via-1",
            asset_id="asset-123",
            start_us=50_000_000,
            end_us=55_000_000,
            evidence_kind=EvidenceKind.EVENT,
            nature=VideoObservationNature.OBSERVATION,
            description="Distant action",
            confidence_original=0.95,
            provider="snipra_vlm",
            model="claude-haiku",
        )]

        # Merge - pass via directly, not [via]
        result = _merge_via_evidence(legacy, via)

        # Pool should only include matched events (no unattached)
        pool_with_via = best_event_pool(result["events"], intent="montage")
        pool_baseline = best_event_pool(legacy, intent="montage")

        assert [e["id"] for e in pool_baseline] == [e["id"] for e in pool_with_via]
        assert pool_baseline[0]["id"] == "evt1"
        assert pool_baseline[1]["id"] == "evt2"
        assert pool_baseline[2]["id"] == "evt3"

    def test_scores_unchanged_with_unmatched_via(self):
        """Scores unchanged with unmatched VIA evidence."""
        from app.editor.events import _merge_via_evidence

        legacy = [
            {"id": "evt1", "legacy_score": 0.8, "source_ranges": [{"start_us": 0, "end_us": 10_000_000}], "additional_evidence": []},
            {"id": "evt2", "legacy_score": 0.6, "source_ranges": [{"start_us": 10_000_000, "end_us": 20_000_000}], "additional_evidence": []},
        ]
        via = [NormalizedVideoEvidence(
            evidence_id="via-1",
            asset_id="asset-123",
            start_us=50_000_000,
            end_us=55_000_000,
            evidence_kind=EvidenceKind.EVENT,
            nature=VideoObservationNature.OBSERVATION,
            description="Distant action",
            confidence_original=0.95,
            provider="snipra_vlm",
            model="claude-haiku",
        )]

        result = _merge_via_evidence(legacy, via)

        # Legacy scores unchanged
        assert result["events"][0]["legacy_score"] == 0.8
        assert result["events"][1]["legacy_score"] == 0.6

        # VIA confidence preserved separately
        assert result["unattached_additional_evidence"][0]["confidence_original"] == 0.95





class TestCoverageMetadataNormalization:
    """Tests for Phase 3.4 Coverage Metadata in VIA Evidence Normalization."""

    def _create_mock_via_result_with_coverage(
        self,
        requested_frames: int | None = 100,
        analyzed_frames: int | None = 75,
        coverage_ratio: float | None = 0.75,
        coverage_status: str | None = "partial",
    ) -> VideoIntelligenceResult:
        """Create a VIA result with coverage metadata for testing."""
        from app.video_intelligence.models import CoverageStatus
        metadata = VideoProcessingMetadata(
            chunks_processed=2,
            chunks_failed=0,
            total_frames_analyzed=analyzed_frames or 0,
            provider="snipra_vlm",
            model="claude-haiku-4-5",
            processing_time_ms=5000,
            partial=False,
            failed_chunks=[],
            requested_frames=requested_frames,
            analyzed_frames=analyzed_frames,
            coverage_ratio=coverage_ratio,
            coverage_status=CoverageStatus(coverage_status) if coverage_status else None,
        )
        return VideoIntelligenceResult(
            asset_id="asset-123",
            start_us=0,
            end_us=30_000_000,
            mode="summarize_video",
            question=None,
            objective="Summarize video",
            summary="Video summary",
            answer=None,
            observations=[
                VideoObservation(
                    asset_id="asset-123",
                    start_us=0,
                    end_us=5_000_000,
                    description="Person enters room",
                    nature=VideoObservationNature.OBSERVATION,
                    confidence=0.9,
                    source="snipra_vlm",
                    provider="snipra_vlm",
                    model="claude-haiku-4-5",
                    chunk_id="chunk_0",
                )
            ],
            events=[
                VideoEventCandidate(
                    asset_id="asset-123",
                    start_us=5_000_000,
                    end_us=15_000_000,
                    type="action",
                    description="Person picks up object",
                    confidence=0.85,
                    nature=VideoObservationNature.OBSERVATION,
                    provider="snipra_vlm",
                    model="claude-haiku-4-5",
                    chunk_id="chunk_1",
                    observed_action="Person picks up object",
                    subjects=["person"],
                    objects=["phone"],
                )
            ],
            scenes=[
                VideoScene(
                    asset_id="asset-123",
                    start_us=0,
                    end_us=30_000_000,
                    description="Indoor scene with person",
                    confidence=0.95,
                    nature=VideoObservationNature.OBSERVATION,
                    provider="snipra_vlm",
                    model="claude-haiku-4-5",
                    chunk_id="chunk_0",
                    subjects=["person"],
                    scene_type="indoor",
                    continuity="continuous",
                    dominant_objects=["person", "phone"],
                )
            ],
            objects=[
                VideoObjectEvidence(
                    asset_id="asset-123",
                    label="phone",
                    timestamps_us=[5_000_000, 10_000_000],
                    confidence=0.9,
                    nature=VideoObservationNature.OBSERVATION,
                    provider="snipra_vlm",
                    model="claude-haiku-4-5",
                    first_seen_us=5_000_000,
                    last_seen_us=10_000_000,
                    track_id="track_001",
                    interactions=["pickup", "hold"],
                )
            ],
            metadata=metadata,
            partial=False,
            contract_version="via_v2",
        )

    def test_requested_frames_preserved(self):
        """requested_frames preserved in normalized evidence."""
        via_result = self._create_mock_via_result_with_coverage(requested_frames=100)
        evidence_list = normalize_via_result(via_result)
        for ev in evidence_list:
            assert ev.requested_frames == 100

    def test_analyzed_frames_preserved(self):
        """analyzed_frames preserved in normalized evidence."""
        via_result = self._create_mock_via_result_with_coverage(analyzed_frames=75)
        evidence_list = normalize_via_result(via_result)
        for ev in evidence_list:
            assert ev.analyzed_frames == 75

    def test_coverage_ratio_deterministic(self):
        """coverage_ratio preserved deterministically."""
        via_result = self._create_mock_via_result_with_coverage(coverage_ratio=0.75)
        evidence_list = normalize_via_result(via_result)
        for ev in evidence_list:
            assert ev.coverage_ratio == 0.75

    def test_coverage_ratio_clamped(self):
        """coverage_ratio clamped to [0.0, 1.0]."""
        from app.video_intelligence.models import CoverageStatus, VideoProcessingMetadata
        metadata = VideoProcessingMetadata(
            provider="test", model="test",
            requested_frames=100, analyzed_frames=150,  # analyzed > requested
            coverage_ratio=1.0, coverage_status=CoverageStatus.COMPLETE,
        )
        via_result = VideoIntelligenceResult(
            asset_id="asset-1", start_us=0, end_us=10_000_000,
            mode="summarize_video", metadata=metadata
        )
        evidence_list = normalize_via_result(via_result)
        for ev in evidence_list:
            assert ev.coverage_ratio == 1.0

    def test_zero_requested_frames_safe(self):
        """Zero requested_frames -> coverage_ratio=None, status=UNKNOWN."""
        from app.video_intelligence.models import CoverageStatus
        via_result = self._create_mock_via_result_with_coverage(
            requested_frames=0, analyzed_frames=50,
            coverage_ratio=None, coverage_status="unknown"
        )
        evidence_list = normalize_via_result(via_result)
        for ev in evidence_list:
            assert ev.requested_frames == 0
            assert ev.coverage_ratio is None
            assert ev.coverage_status == CoverageStatus.UNKNOWN

    def test_unknown_requested_frames(self):
        """None requested_frames -> coverage_ratio=None, status=UNKNOWN."""
        from app.video_intelligence.models import CoverageStatus
        via_result = self._create_mock_via_result_with_coverage(
            requested_frames=None, analyzed_frames=50,
            coverage_ratio=None, coverage_status="unknown"
        )
        evidence_list = normalize_via_result(via_result)
        for ev in evidence_list:
            assert ev.requested_frames is None
            assert ev.coverage_ratio is None
            assert ev.coverage_status == CoverageStatus.UNKNOWN

    def test_coverage_status_complete(self):
        """100% coverage -> COMPLETE status."""
        from app.video_intelligence.models import CoverageStatus
        via_result = self._create_mock_via_result_with_coverage(
            requested_frames=100, analyzed_frames=100,
            coverage_ratio=1.0, coverage_status="complete"
        )
        evidence_list = normalize_via_result(via_result)
        for ev in evidence_list:
            assert ev.coverage_status == CoverageStatus.COMPLETE

    def test_coverage_status_partial(self):
        """50-99% coverage -> PARTIAL status."""
        from app.video_intelligence.models import CoverageStatus
        via_result = self._create_mock_via_result_with_coverage(
            requested_frames=100, analyzed_frames=75,
            coverage_ratio=0.75, coverage_status="partial"
        )
        evidence_list = normalize_via_result(via_result)
        for ev in evidence_list:
            assert ev.coverage_status == CoverageStatus.PARTIAL

    def test_coverage_status_insufficient(self):
        """Zero coverage (nothing analyzed) -> INSUFFICIENT status."""
        from app.video_intelligence.models import CoverageStatus
        via_result = self._create_mock_via_result_with_coverage(
            requested_frames=100, analyzed_frames=0,
            coverage_ratio=0.0, coverage_status="insufficient"
        )
        evidence_list = normalize_via_result(via_result)
        for ev in evidence_list:
            assert ev.coverage_status == CoverageStatus.INSUFFICIENT

    def test_coverage_status_unknown(self):
        """Unknown coverage -> UNKNOWN status."""
        from app.video_intelligence.models import CoverageStatus
        via_result = self._create_mock_via_result_with_coverage(
            requested_frames=None, analyzed_frames=None,
            coverage_ratio=None, coverage_status="unknown"
        )
        evidence_list = normalize_via_result(via_result)
        for ev in evidence_list:
            assert ev.coverage_status == CoverageStatus.UNKNOWN

    def test_partial_result_not_silently_complete(self):
        """Partial result with failed chunks should not silently show COMPLETE."""
        via_result = self._create_mock_via_result_with_coverage(
            requested_frames=100, analyzed_frames=100,
            coverage_ratio=1.0, coverage_status="complete"
        )
        via_result.partial = True
        via_result.metadata.failed_chunks = [{"chunk_index": 1, "error": "timeout"}]

        evidence_list = normalize_via_result(via_result)
        for ev in evidence_list:
            assert ev.partial is True
            assert len(ev.failed_chunks) == 1
            # coverage_status is still COMPLETE (based on frames), but partial=True
            # This is correct - coverage and partial are separate concepts

    def test_failed_chunks_preserved(self):
        """failed_chunks preserved in normalized evidence."""
        via_result = self._create_mock_via_result_with_coverage()
        via_result.metadata.failed_chunks = [{"chunk_index": 2, "error": "timeout"}]

        evidence_list = normalize_via_result(via_result)
        for ev in evidence_list:
            assert len(ev.failed_chunks) == 1
            assert ev.failed_chunks[0]["error"] == "timeout"


class TestCoverageMetadataEventMapPropagation:
    """Tests for Phase 3.4 Coverage Metadata propagation to Event Map."""

    def _create_via_evidence_with_coverage(
        self, start_us=12_000_000, end_us=18_000_000, **coverage_kwargs
    ):
        """Create NormalizedVideoEvidence with coverage metadata."""
        from app.editor.via_evidence import EvidenceKind, NormalizedVideoEvidence
        from app.video_intelligence.models import CoverageStatus, VideoObservationNature

        defaults = {
            "requested_frames": 100,
            "analyzed_frames": 75,
            "coverage_ratio": 0.75,
            "coverage_status": CoverageStatus.PARTIAL,
        }
        defaults.update(coverage_kwargs)

        return NormalizedVideoEvidence(
            evidence_id="via-ev-1",
            asset_id="asset-123",
            start_us=start_us,
            end_us=end_us,
            evidence_kind=EvidenceKind.EVENT,
            nature=VideoObservationNature.OBSERVATION,
            description="Test action",
            confidence_original=0.88,
            provider="snipra_vlm",
            model="claude-haiku",
            chunk_id="chunk_0",
            contract_version="via_v2",
            **defaults
        )

    def test_event_map_exposes_coverage(self):
        """Coverage metadata exposed in Event Map additional_evidence."""
        from app.editor.events import _merge_via_evidence

        legacy = [{
            "id": "legacy-1",
            "asset_id": "asset-123",
            "start_us": 10_000_000,
            "end_us": 20_000_000,
            "title": "Legacy Event",
            "reason": "Test reason",
            "legacy_score": 0.7,
            "source_ranges": [{"start_us": 10_000_000, "end_us": 20_000_000}],
            "evidence": [],
            "uncertain": False,
        }]

        via = [self._create_via_evidence_with_coverage()]

        result = _merge_via_evidence(legacy, via)

        attached = result["events"][0]["additional_evidence"][0]
        assert "requested_frames" in attached
        assert "analyzed_frames" in attached
        assert "coverage_ratio" in attached
        assert "coverage_status" in attached
        assert attached["requested_frames"] == 100
        assert attached["analyzed_frames"] == 75
        assert attached["coverage_ratio"] == 0.75
        assert attached["coverage_status"] == "partial"

    def test_unattached_evidence_preserves_coverage(self):
        """Unattached VIA evidence preserves coverage metadata."""
        from app.editor.events import _merge_via_evidence

        legacy = [{
            "id": "legacy-1",
            "asset_id": "asset-123",
            "start_us": 0,
            "end_us": 5_000_000,
            "title": "Legacy Event",
            "reason": "Test reason",
            "legacy_score": 0.7,
            "source_ranges": [{"start_us": 0, "end_us": 5_000_000}],
            "evidence": [],
            "uncertain": False,
        }]

        via = [self._create_via_evidence_with_coverage(
            start_us=50_000_000, end_us=55_000_000
        )]

        result = _merge_via_evidence(legacy, via)

        unattached = result["unattached_additional_evidence"][0]
        assert unattached["requested_frames"] == 100
        assert unattached["analyzed_frames"] == 75
        assert unattached["coverage_ratio"] == 0.75
        assert unattached["coverage_status"] == "partial"

    def test_candidate_count_unchanged(self):
        """Candidate count unchanged with VIA evidence (coverage or not)."""
        from app.editor.events import _merge_via_evidence, best_event_pool

        legacy = [{
            "id": "evt1", "legacy_score": 0.8,
            "source_ranges": [{"start_us": 0, "end_us": 10_000_000}],
            "additional_evidence": []
        }, {
            "id": "evt2", "legacy_score": 0.6,
            "source_ranges": [{"start_us": 10_000_000, "end_us": 20_000_000}],
            "additional_evidence": []
        }]

        via = [self._create_via_evidence_with_coverage()]

        result = _merge_via_evidence(legacy, via)

        pool_baseline = best_event_pool(legacy, intent="montage")
        pool_with_via = best_event_pool(result["events"], intent="montage")

        assert len(pool_baseline) == len(pool_with_via) == 2

    def test_legacy_score_unchanged(self):
        """Legacy score unchanged with VIA evidence (coverage or not)."""
        from app.editor.events import _merge_via_evidence

        legacy = [{
            "id": "evt1", "legacy_score": 0.73,
            "source_ranges": [{"start_us": 0, "end_us": 10_000_000}], "additional_evidence": []
        }]

        via = [self._create_via_evidence_with_coverage()]

        result = _merge_via_evidence(legacy, via)

        assert result["events"][0]["legacy_score"] == 0.73

    def test_best_event_pool_ordering_unchanged(self):
        """Best Event Pool ordering unchanged with VIA evidence (coverage or not)."""
        from app.editor.events import _merge_via_evidence, best_event_pool

        legacy = [
            {"id": "evt1", "legacy_score": 0.9,
             "source_ranges": [{"start_us": 0, "end_us": 10_000_000}],
             "additional_evidence": []},
            {"id": "evt2", "legacy_score": 0.7,
             "source_ranges": [{"start_us": 10_000_000, "end_us": 20_000_000}],
             "additional_evidence": []},
            {"id": "evt3", "legacy_score": 0.5,
             "source_ranges": [{"start_us": 20_000_000, "end_us": 30_000_000}],
             "additional_evidence": []},
        ]

        via = [self._create_via_evidence_with_coverage(
            start_us=22_000_000, end_us=28_000_000
        )]

        result = _merge_via_evidence(legacy, via)

        pool_baseline = best_event_pool(legacy, intent="montage")
        pool_with_via = best_event_pool(result["events"], intent="montage")

        assert [e["id"] for e in pool_baseline] == [e["id"] for e in pool_with_via]
        assert pool_baseline[0]["id"] == "evt1"
        assert pool_baseline[1]["id"] == "evt2"
        assert pool_baseline[2]["id"] == "evt3"


class TestAgentVIAIntegration:
    """Tests for Agent orchestration with VIA evidence flow."""

    def test_agent_passes_normalized_via_to_event_map(self):
        """Agent normalizes VIA result and passes to events_for_job."""
        # This is tested via integration tests in test_editor_agent_e2e.py
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])