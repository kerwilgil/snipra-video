"""Normalized VIA Evidence Layer — provider-neutral, deterministic editorial evidence."""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.video_intelligence.models import (
    ConfidenceSource,
    CoverageStatus,
    EvidenceNature,
    NatureSource,
    TemporalSource,
    VideoEventCandidate,
    VideoIntelligenceResult,
    VideoObjectEvidence,
    VideoScene,
)
from app.video_intelligence.scene_cache import (
    read_cached_boundaries,
    resolve_scene_cache_lookup,
)
from app.video_intelligence.scene_detection import DEFAULT_SCENE_CHANGE_THRESHOLD


class EvidenceKind(StrEnum):
    OBSERVATION = "observation"
    EVENT = "event"
    SCENE = "scene"
    OBJECT = "object"


class NormalizedVideoEvidence(BaseModel):
    """Normalized evidence from VIA, ready for editorial pipeline."""

    evidence_id: str
    asset_id: str
    start_us: int
    end_us: int
    evidence_kind: EvidenceKind
    nature: EvidenceNature
    description: str
    confidence_original: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_source: ConfidenceSource = ConfidenceSource.UNKNOWN
    nature_source: NatureSource = NatureSource.UNKNOWN
    source: Literal["via"] = "via"
    provider: str
    model: str
    chunk_id: str | None = None
    contract_version: str = "via_v2"
    partial: bool = False
    failed_chunks: list[dict] = Field(default_factory=list)

    # Provenance fields from EvidenceProvenance
    provenance_frame_ids: list[str] = Field(default_factory=list)
    provenance_frame_timestamps_us: list[int] = Field(default_factory=list)
    provenance_analysis_window_start_us: int | None = None
    provenance_analysis_window_end_us: int | None = None
    provenance_temporal_source: TemporalSource = TemporalSource.UNKNOWN
    provenance_confidence_source: ConfidenceSource = ConfidenceSource.UNKNOWN
    provenance_nature_source: NatureSource = NatureSource.UNKNOWN

    # Coverage metadata (Phase 3.4)
    requested_frames: int | None = Field(default=None, ge=0)
    analyzed_frames: int | None = Field(default=None, ge=0)
    coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    coverage_status: CoverageStatus | None = Field(default=None)

    # Event enrichment
    observed_action: str | None = None
    subjects: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)

    # Scene enrichment
    scene_type_value: str | None = Field(default=None, alias="scene_type")
    continuity_value: str | None = Field(default=None, alias="scene_continuity")
    dominant_objects: list[str] = Field(default_factory=list)

    # Object enrichment
    object_label: str | None = None
    first_seen_us: int | None = None
    last_seen_us: int | None = None
    object_track_id_value: str | None = Field(default=None, alias="object_track_id")
    interactions: list[str] = Field(default_factory=list)

    # Private fields for property storage (using double underscore prefix for name mangling)
    scene_type_storage: str | None = Field(default=None, exclude=True)
    scene_continuity_storage: str | None = Field(default=None, exclude=True)
    object_track_id_storage: str | None = Field(default=None, exclude=True)
    object_interaction_storage: str | None = Field(default=None, exclude=True)

    # Legacy metadata fields (backward compat) - renamed to avoid conflicts
    visual_novelty: str | None = None
    payoff_present: bool | None = None
    reaction_present: bool | None = None
    evidence_quality: str | None = None
    scene_type_legacy: str | None = None
    scene_continuity_legacy: str | None = None
    object_track_id_legacy: str | None = None
    object_interaction_legacy: str | None = None

    # Temporal evidence continuity (Phase 4.4A). Optional, backward compatible:
    # older persisted evidence (pre-4.4A) simply validates with both None.
    # Populated only by annotate_continuity() for EVENT/OBSERVATION evidence
    # that was actually grouped under the STRICT scene-cache policy - never
    # set for OBJECT/SCENE, and never inferred from identity/tracking/scoring.
    continuity_group_id: str | None = None
    continuity_basis: str | None = None

    # Handle object_interaction alias for backward compatibility
    @model_validator(mode="before")
    @classmethod
    def _handle_object_interaction(cls, data: dict) -> dict:
        if isinstance(data, dict) and "object_interaction" in data:
            interaction = data.pop("object_interaction")
            if interaction:
                interactions = data.get("interactions", [])
                if interaction not in interactions:
                    interactions.append(interaction)
                data["interactions"] = interactions
        return data

# Backward-compatible property aliases
    @property
    def scene_type(self) -> str | None:
        """Backward compat: returns scene_type_value (new) or scene_type_legacy."""
        return self.scene_type_value or self.scene_type_legacy

    @scene_type.setter
    def scene_type(self, value: str | None) -> None:
        self.scene_type_value = value
        self.scene_type_legacy = value

    @property
    def scene_continuity(self) -> str | None:
        """Backward compat: returns continuity_value (new) or scene_continuity_legacy."""
        return self.continuity_value or self.scene_continuity_legacy

    @scene_continuity.setter
    def scene_continuity(self, value: str | None) -> None:
        self.continuity_value = value
        self.scene_continuity_legacy = value

    @property
    def object_track_id(self) -> str | None:
        return self.object_track_id_value or self.object_track_id_legacy

    @object_track_id.setter
    def object_track_id(self, value: str | None) -> None:
        self.object_track_id_value = value
        self.object_track_id_legacy = value

    @property
    def object_interaction(self) -> str | None:
        """Backward compat: returns first interaction or object_interaction_legacy."""
        interactions = self.__dict__.get('interactions', [])
        if interactions:
            return interactions[0]
        return self.__dict__.get('object_interaction_legacy')

    @object_interaction.setter
    def object_interaction(self, value: str | None) -> None:
        if value:
            interactions = self.__dict__.get('interactions', [])
            if value not in interactions:
                interactions.append(value)


def compute_evidence_id(
    asset_id: str,
    start_us: int,
    end_us: int,
    evidence_kind: EvidenceKind,
    description: str,
    provider: str,
    model: str,
    chunk_id: str | None = None,
) -> str:
    """Compute deterministic SHA256 evidence ID."""
    # Normalize description for identity: lowercase, strip, limit length
    normalized_desc = description.strip().lower()[:160]
    parts = [
        asset_id,
        str(start_us),
        str(end_us),
        evidence_kind.value,
        normalized_desc,
        provider,
        model,
        chunk_id or "",
    ]
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _observation_to_evidence(
    obs,
    asset_id: str,
    provider: str,
    model: str,
    contract_version: str,
    partial: bool,
    failed_chunks: list[dict],
    requested_frames: int | None = None,
    analyzed_frames: int | None = None,
    coverage_ratio: float | None = None,
    coverage_status: CoverageStatus | None = None,
) -> NormalizedVideoEvidence:
    evidence_id = compute_evidence_id(
        asset_id=asset_id,
        start_us=obs.start_us,
        end_us=obs.end_us,
        evidence_kind=EvidenceKind.OBSERVATION,
        description=obs.description,
        provider=provider,
        model=model,
        chunk_id=obs.chunk_id,
    )
    # Extract provenance from obs.provenance if available
    prov = obs.provenance
    return NormalizedVideoEvidence(
        evidence_id=evidence_id,
        asset_id=asset_id,
        start_us=obs.start_us,
        end_us=obs.end_us,
        evidence_kind=EvidenceKind.OBSERVATION,
        nature=obs.nature,
        description=obs.description,
        confidence_original=obs.confidence,
        confidence_source=obs.confidence_source,
        nature_source=obs.nature_source,
        provider=provider,
        model=model,
        chunk_id=obs.chunk_id,
        contract_version=contract_version,
        partial=partial,
        failed_chunks=failed_chunks,
        # Coverage metadata (Phase 3.4)
        requested_frames=requested_frames,
        analyzed_frames=analyzed_frames,
        coverage_ratio=coverage_ratio,
        coverage_status=coverage_status,
        # Provenance fields
        provenance_frame_ids=prov.frame_ids if prov else [],
        provenance_frame_timestamps_us=prov.frame_timestamps_us if prov else [],
        provenance_analysis_window_start_us=prov.analysis_window_start_us if prov else None,
        provenance_analysis_window_end_us=prov.analysis_window_end_us if prov else None,
        provenance_temporal_source=prov.temporal_source if prov else TemporalSource.UNKNOWN,
        provenance_confidence_source=prov.confidence_source if prov else ConfidenceSource.UNKNOWN,
        provenance_nature_source=prov.nature_source if prov else NatureSource.UNKNOWN,
    )


def _event_to_evidence(
    ev: VideoEventCandidate,
    asset_id: str,
    provider: str,
    model: str,
    contract_version: str,
    partial: bool,
    failed_chunks: list[dict],
    requested_frames: int | None = None,
    analyzed_frames: int | None = None,
    coverage_ratio: float | None = None,
    coverage_status: CoverageStatus | None = None,
) -> NormalizedVideoEvidence:
    evidence_id = compute_evidence_id(
        asset_id=asset_id,
        start_us=ev.start_us,
        end_us=ev.end_us,
        evidence_kind=EvidenceKind.EVENT,
        description=ev.description,
        provider=provider,
        model=model,
        chunk_id=ev.chunk_id,
    )
    # Extract provenance from ev.provenance if available
    prov = ev.provenance
    return NormalizedVideoEvidence(
        evidence_id=evidence_id,
        asset_id=asset_id,
        start_us=ev.start_us,
        end_us=ev.end_us,
        evidence_kind=EvidenceKind.EVENT,
        nature=ev.nature,
        description=ev.description,
        confidence_original=ev.confidence,
        confidence_source=ev.confidence_source,
        nature_source=ev.nature_source,
        provider=provider,
        model=model,
        chunk_id=ev.chunk_id,
        contract_version=contract_version,
        partial=partial,
        failed_chunks=failed_chunks,
        # Coverage metadata (Phase 3.4)
        requested_frames=requested_frames,
        analyzed_frames=analyzed_frames,
        coverage_ratio=coverage_ratio,
        coverage_status=coverage_status,
        # Direct fields from VideoEventCandidate
        observed_action=ev.observed_action,
        subjects=ev.subjects,
        objects=ev.objects,
        # Legacy metadata fields (backward compat)
        visual_novelty=ev.metadata.get("visual_novelty") if ev.metadata else None,
        payoff_present=ev.metadata.get("payoff_present") if ev.metadata else None,
        reaction_present=ev.metadata.get("reaction_present") if ev.metadata else None,
        evidence_quality=ev.metadata.get("evidence_quality") if ev.metadata else None,
        # Provenance fields
        provenance_frame_ids=prov.frame_ids if prov else [],
        provenance_frame_timestamps_us=prov.frame_timestamps_us if prov else [],
        provenance_analysis_window_start_us=prov.analysis_window_start_us if prov else None,
        provenance_analysis_window_end_us=prov.analysis_window_end_us if prov else None,
        provenance_temporal_source=prov.temporal_source if prov else TemporalSource.UNKNOWN,
        provenance_confidence_source=prov.confidence_source if prov else ConfidenceSource.UNKNOWN,
        provenance_nature_source=prov.nature_source if prov else NatureSource.UNKNOWN,
    )


def _scene_to_evidence(
    scene: VideoScene,
    asset_id: str,
    provider: str,
    model: str,
    contract_version: str,
    partial: bool,
    failed_chunks: list[dict],
    requested_frames: int | None = None,
    analyzed_frames: int | None = None,
    coverage_ratio: float | None = None,
    coverage_status: CoverageStatus | None = None,
) -> NormalizedVideoEvidence:
    evidence_id = compute_evidence_id(
        asset_id=asset_id,
        start_us=scene.start_us,
        end_us=scene.end_us,
        evidence_kind=EvidenceKind.SCENE,
        description=scene.description,
        provider=provider,
        model=model,
        chunk_id=scene.chunk_id,
    )
    # Extract provenance from scene.provenance if available
    prov = scene.provenance
    return NormalizedVideoEvidence(
        evidence_id=evidence_id,
        asset_id=asset_id,
        start_us=scene.start_us,
        end_us=scene.end_us,
        evidence_kind=EvidenceKind.SCENE,
        nature=scene.nature,
        description=scene.description,
        confidence_original=scene.confidence,
        confidence_source=scene.confidence_source,
        nature_source=scene.nature_source,
        provider=provider,
        model=model,
        chunk_id=scene.chunk_id,
        contract_version=contract_version,
        partial=partial,
        failed_chunks=failed_chunks,
        # Coverage metadata (Phase 3.4)
        requested_frames=requested_frames,
        analyzed_frames=analyzed_frames,
        coverage_ratio=coverage_ratio,
        coverage_status=coverage_status,
        # Direct fields from VideoScene
        subjects=scene.subjects,
        scene_type=scene.scene_type,
        scene_continuity=scene.continuity,
        dominant_objects=scene.dominant_objects,
        # Legacy metadata fields (backward compat) - use different names to avoid conflict
        scene_type_legacy=scene.metadata.get("scene_type") if scene.metadata else None,
        scene_continuity_legacy=scene.metadata.get("scene_continuity") if scene.metadata else None,
        # Provenance fields
        provenance_frame_ids=prov.frame_ids if prov else [],
        provenance_frame_timestamps_us=prov.frame_timestamps_us if prov else [],
        provenance_analysis_window_start_us=prov.analysis_window_start_us if prov else None,
        provenance_analysis_window_end_us=prov.analysis_window_end_us if prov else None,
        provenance_temporal_source=prov.temporal_source if prov else TemporalSource.UNKNOWN,
        provenance_confidence_source=prov.confidence_source if prov else ConfidenceSource.UNKNOWN,
        provenance_nature_source=prov.nature_source if prov else NatureSource.UNKNOWN,
    )


def _object_to_evidence(
    obj: VideoObjectEvidence,
    asset_id: str,
    provider: str,
    model: str,
    contract_version: str,
    partial: bool,
    failed_chunks: list[dict],
    requested_frames: int | None = None,
    analyzed_frames: int | None = None,
    coverage_ratio: float | None = None,
    coverage_status: CoverageStatus | None = None,
    analysis_window_start_us: int | None = None,
    analysis_window_end_us: int | None = None,
) -> NormalizedVideoEvidence:
    prov = obj.provenance
    has_prov_window = (
        prov is not None
        and prov.analysis_window_start_us is not None
        and prov.analysis_window_end_us is not None
    )

    # For objects, use first/last timestamp as range.
    # NO synthetic zero timestamps - use real timestamps, provenance window, or
    # the VIA result window; otherwise raise.
    timestamps = obj.timestamps_us if obj.timestamps_us else []
    if timestamps:
        start_us = min(timestamps)
        end_us = max(timestamps)
    elif has_prov_window:
        start_us = prov.analysis_window_start_us
        end_us = prov.analysis_window_end_us
    elif analysis_window_start_us is not None and analysis_window_end_us is not None:
        # Use the general analyzed range from VIA result as conservative window
        start_us = analysis_window_start_us
        end_us = analysis_window_end_us
    else:
        # No timestamps available - raise error instead of using synthetic zero
        raise ValueError(
            "Object evidence has no temporal grounding or analysis window"
        )

    # Temporal source: an explicit provenance source is authoritative and is
    # preserved verbatim - model-provided timestamps stay MODEL_GROUNDED and a
    # bare timestamps_us list is NEVER auto-promoted to FRAME_MAPPED. Only when
    # there is no usable provenance source do we classify from what we used.
    if prov is not None and prov.temporal_source != TemporalSource.UNKNOWN:
        obj_temporal_source = prov.temporal_source
    elif obj.timestamps_us:
        obj_temporal_source = TemporalSource.FRAME_MAPPED
    elif has_prov_window or (
        analysis_window_start_us is not None and analysis_window_end_us is not None
    ):
        obj_temporal_source = TemporalSource.CHUNK_WINDOW
    else:
        obj_temporal_source = TemporalSource.UNKNOWN

    evidence_id = compute_evidence_id(
        asset_id=asset_id,
        start_us=start_us,
        end_us=end_us,
        evidence_kind=EvidenceKind.OBJECT,
        description=f"Object: {obj.label}",
        provider=provider,
        model=model,
        chunk_id=None,
    )
    return NormalizedVideoEvidence(
        evidence_id=evidence_id,
        asset_id=asset_id,
        start_us=start_us,
        end_us=end_us,
        evidence_kind=EvidenceKind.OBJECT,
        nature=obj.nature,
        description=f"Object: {obj.label}",
        confidence_original=obj.confidence,
        confidence_source=obj.confidence_source,
        nature_source=obj.nature_source,
        provider=provider,
        model=model,
        chunk_id=None,
        contract_version=contract_version,
        partial=partial,
        failed_chunks=failed_chunks,
        # Coverage metadata (Phase 3.4)
        requested_frames=requested_frames,
        analyzed_frames=analyzed_frames,
        coverage_ratio=coverage_ratio,
        coverage_status=coverage_status,
        # Direct fields from VideoObjectEvidence
        object_label=obj.label,
        first_seen_us=obj.first_seen_us,
        last_seen_us=obj.last_seen_us,
        object_track_id=obj.track_id,
        interactions=obj.interactions,
        # Legacy metadata fields (backward compat)
        object_track_id_legacy=obj.metadata.get("track_id") if obj.metadata else None,
        object_interaction_legacy=obj.metadata.get("interaction") if obj.metadata else None,
        # Provenance fields
        provenance_frame_ids=prov.frame_ids if prov else [],
        provenance_frame_timestamps_us=prov.frame_timestamps_us if prov else [],
        provenance_analysis_window_start_us=prov.analysis_window_start_us if prov else None,
        provenance_analysis_window_end_us=prov.analysis_window_end_us if prov else None,
        provenance_temporal_source=obj_temporal_source,
        provenance_confidence_source=prov.confidence_source if prov else ConfidenceSource.UNKNOWN,
        provenance_nature_source=prov.nature_source if prov else NatureSource.UNKNOWN,
    )


def normalize_via_result(
    via_result: VideoIntelligenceResult,
) -> list[NormalizedVideoEvidence]:
    """Transform VideoIntelligenceResult into normalized editorial evidence list."""
    provider = via_result.metadata.provider if via_result.metadata else "unknown"
    model = via_result.metadata.model if via_result.metadata else "unknown"
    contract_version = via_result.contract_version
    partial = via_result.partial
    failed_chunks = via_result.metadata.failed_chunks if via_result.metadata else []

    # Extract coverage metadata from VIA result (Phase 3.4)
    requested_frames = via_result.metadata.requested_frames if via_result.metadata else None
    analyzed_frames = via_result.metadata.analyzed_frames if via_result.metadata else None
    coverage_ratio = via_result.metadata.coverage_ratio if via_result.metadata else None
    coverage_status = via_result.metadata.coverage_status if via_result.metadata else None

    evidence_list: list[NormalizedVideoEvidence] = []

    # Observations
    for obs in via_result.observations:
        evidence_list.append(
            _observation_to_evidence(
                obs,
                asset_id=via_result.asset_id,
                provider=provider,
                model=model,
                contract_version=contract_version,
                partial=partial,
                failed_chunks=failed_chunks,
                requested_frames=requested_frames,
                analyzed_frames=analyzed_frames,
                coverage_ratio=coverage_ratio,
                coverage_status=coverage_status,
            )
        )

    # Events
    for ev in via_result.events:
        evidence_list.append(
            _event_to_evidence(
                ev,
                asset_id=via_result.asset_id,
                provider=provider,
                model=model,
                contract_version=contract_version,
                partial=partial,
                failed_chunks=failed_chunks,
                requested_frames=requested_frames,
                analyzed_frames=analyzed_frames,
                coverage_ratio=coverage_ratio,
                coverage_status=coverage_status,
            )
        )

    # Scenes
    for scene in via_result.scenes:
        evidence_list.append(
            _scene_to_evidence(
                scene,
                asset_id=via_result.asset_id,
                provider=provider,
                model=model,
                contract_version=contract_version,
                partial=partial,
                failed_chunks=failed_chunks,
                requested_frames=requested_frames,
                analyzed_frames=analyzed_frames,
                coverage_ratio=coverage_ratio,
                coverage_status=coverage_status,
            )
        )

    # Objects
    for obj in via_result.objects:
        evidence_list.append(
            _object_to_evidence(
                obj,
                asset_id=via_result.asset_id,
                provider=provider,
                model=model,
                contract_version=contract_version,
                partial=partial,
                failed_chunks=failed_chunks,
                requested_frames=requested_frames,
                analyzed_frames=analyzed_frames,
                coverage_ratio=coverage_ratio,
                coverage_status=coverage_status,
                analysis_window_start_us=via_result.start_us,
                analysis_window_end_us=via_result.end_us,
            )
        )

    return evidence_list


def temporal_overlap_us(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Compute temporal overlap in microseconds."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _normalize_description(desc: str) -> str:
    """Normalize description for comparison."""
    return desc.strip().lower()[:160]


def _relative_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Compute relative overlap: intersection / min(duration_a, duration_b)."""
    overlap = temporal_overlap_us(a_start, a_end, b_start, b_end)
    if overlap == 0:
        return 0.0
    duration_a = a_end - a_start
    duration_b = b_end - b_start
    min_duration = min(duration_a, duration_b)
    if min_duration == 0:
        return 0.0
    return overlap / min_duration


def deduplicate_evidence(
    evidence_list: list[NormalizedVideoEvidence],
    relative_overlap_threshold: float = 0.8,
    absolute_overlap_floor_us: int = 5_000_000,  # 5 seconds for long events
) -> list[NormalizedVideoEvidence]:
    """
    Conservative deduplication of normalized evidence.

    Only merges when there is strong evidence of duplication:
    - same asset
    - same evidence_kind
    - relative overlap >= 80% of shorter event's duration, OR
      absolute overlap >= 5s for long events
    - normalized descriptions match exactly

    If in doubt, PRESERVE BOTH.
    """
    if not evidence_list:
        return []

    # Sort by start time for deterministic processing
    sorted_evidence = sorted(evidence_list, key=lambda e: (e.start_us, e.evidence_id))

    deduplicated: list[NormalizedVideoEvidence] = []

    for current in sorted_evidence:
        merged = False

        for existing in deduplicated:
            # Must be same asset and kind
            if current.asset_id != existing.asset_id:
                continue
            if current.evidence_kind != existing.evidence_kind:
                continue

            # Check temporal overlap - relative criterion with absolute floor
            rel_overlap = _relative_overlap(
                current.start_us, current.end_us,
                existing.start_us, existing.end_us
            )
            abs_overlap = temporal_overlap_us(
                current.start_us, current.end_us,
                existing.start_us, existing.end_us
            )
            if rel_overlap < relative_overlap_threshold and abs_overlap < absolute_overlap_floor_us:
                continue

            # Check description match (normalized)
            curr_desc = _normalize_description(current.description)
            exist_desc = _normalize_description(existing.description)
            if curr_desc != exist_desc:
                continue

            # Strong evidence of duplicate - merge
            # Keep the one with higher confidence
            if current.confidence_original > existing.confidence_original:
                # Replace existing with current (higher confidence)
                idx = deduplicated.index(existing)
                deduplicated[idx] = current
            # Else keep existing (higher or equal confidence)
            merged = True
            break

        if not merged:
            deduplicated.append(current)

    return deduplicated


# ---------------------------------------------------------------------------
# Phase 4.4A - Temporal Evidence Continuity (precision-first, conservative)
# ---------------------------------------------------------------------------
#
# Continuity links cross-chunk EVENT/OBSERVATION evidence that is very likely
# describing the *same ongoing occurrence* seen from adjacent chunks. It is a
# TECHNICAL evidence-coherence signal for the Director to read, never an
# editorial one: no ranking, no scoring, no identity/person/object tracking,
# no embeddings, no LLM call, no synonym/semantic matching. Absence of a
# group is always the default; a group is only ever formed when every
# positive rule + every veto agrees.
_GENERIC_TOKENS_FIXED: frozenset[str] = frozenset(
    {
        # Global generic subjects (any corpus)
        "person", "people", "man", "woman", "someone", "scene", "video", "frame",
        "persona", "personas", "gente", "alguien", "escena",
        # Gameplay-corpus generic terms (Precision Gate observation)
        "streamer", "screen", "pantalla", "game", "juego", "character", "personaje",
        "combat", "fighting", "combate", "health", "salud", "vida",
        "background", "fondo",
    }
)

# Pure grammatical function words (articles, prepositions, conjunctions,
# copulas, common possessive/demonstrative pronouns) - ES/EN. These carry
# ZERO subject-identity content by construction (unlike _GENERIC_TOKENS_FIXED
# above, which lists specific over-used SUBJECT/domain nouns), so excluding
# them is not "convirtiendo terminos de gameplay en stopwords" (section 8's
# caution) - it is what "remove punctuation, tokenize, remove generic
# tokens" (section 7) already implies: a grammatical connective can never be
# a valid positive lexical anchor (section 10). Confirmed necessary by the
# Phase 4.4A real-smoke run: without this, "with"/"in"/"the" alone produced
# spurious cross-chunk links between otherwise unrelated observations whose
# corpus frequency happened to stay under CORPUS_GENERIC_FREQUENCY_THRESHOLD.
_STOPWORDS_FIXED: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "in", "on", "at", "with", "from", "to",
        "is", "are", "was", "were", "be", "been", "being", "his", "her", "their",
        "its", "this", "that", "these", "those", "as", "by", "for", "into", "onto",
        "over", "under", "up", "down", "out", "off", "not", "no", "all",
        "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "de",
        "en", "con", "desde", "hasta", "al", "del", "es", "son", "era", "eran",
        "ser", "siendo", "sido", "su", "sus", "este", "esta", "estos", "estas",
        "ese", "esa", "esos", "esas", "por", "para", "entre", "sobre", "bajo",
    }
)

# Conservative default within the Precision Gate's suggested ~50-60% band.
# A token appearing in MORE than this fraction of the request's evidence
# items (of the same evidence_kind, after fixed-generic filtering) is treated
# as corpus-generic for THIS request only - never persisted, never a
# Director-facing setting. Strict ">" so an exact 50/50 split (n=2, 1 shared
# token) does not get filtered - see test_continuity.py for the boundary
# case.
CORPUS_GENERIC_FREQUENCY_THRESHOLD = 0.55

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúüñ]+")

# Fixed, small, symmetric ES/EN color vocabulary. Colors are never a positive
# anchor signal (two same-colored items sharing only "red"/"rojo" is not
# evidence of continuity) - they exist solely to VETO a relation when two
# candidate items report clearly different colors for what would otherwise
# look like the same subject.
_COLOR_CANON: dict[str, str] = {
    "red": "red", "rojo": "red",
    "blue": "blue", "azul": "blue",
    "green": "green", "verde": "green",
    "yellow": "yellow", "amarillo": "yellow",
    "black": "black", "negro": "black",
    "white": "white", "blanco": "white",
    "orange": "orange", "naranja": "orange",
    "purple": "purple", "morado": "purple",
    "pink": "pink", "rosa": "pink",
    "gray": "gray", "grey": "gray", "gris": "gray",
}

_CHUNK_ID_RE = re.compile(r"^chunk_(\d+)$")


def compute_continuity_group_id(evidence_ids: list[str]) -> str:
    """Deterministic SHA256 identity for a continuity group.

    A function only of the SET of member evidence_ids, canonically sorted -
    never confidence, legacy_score, or accidental list position. Same
    membership -> same id, regardless of input ordering, chunk-completion
    ordering, or concurrency (Phase 4.3).
    """
    canonical = "|".join(sorted(evidence_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _extract_chunk_index(chunk_id: str | None) -> int | None:
    """Extract the integer index from a `chunk_{N}` chunk_id.

    Returns None (never raises, never guesses) for any chunk_id that does not
    match the actual format providers.py emits
    (`f"chunk_{chunk.index}"` - see ExistingSnipraVLMProvider.ask_video /
    summarize_video). Items with an unparseable/missing chunk_id are simply
    excluded from continuity candidate generation - a safe abstention, never
    an invented association.
    """
    if not chunk_id:
        return None
    match = _CHUNK_ID_RE.match(chunk_id)
    if not match:
        return None
    return int(match.group(1))


def _tokenize(description: str) -> list[str]:
    """lowercase/casefold -> strip punctuation -> tokenize. No stemming, no
    embeddings, no synonym mapping - conservative textual identity only."""
    return _TOKEN_RE.findall(description.strip().casefold())


def _fixed_filtered_tokens(description: str) -> set[str]:
    """Anchor-eligible tokens: fixed generic terms removed, AND colors always
    excluded (section 11: "Color NO es senal positiva" - colors are a veto-
    only signal and must never themselves satisfy the positive-anchor
    requirement, e.g. two items sharing only "green"/"purple" must not
    count as related). Single-character tokens are also dropped - they are
    near-always punctuation/apostrophe artifacts (e.g. "person's" -> "person"
    + a stray "s") rather than meaningful textual identity, and the
    tokenizer intentionally does no stemming/contraction handling beyond
    this conservative floor."""
    return {
        t
        for t in _tokenize(description)
        if t not in _GENERIC_TOKENS_FIXED
        and t not in _STOPWORDS_FIXED
        and t not in _COLOR_CANON
        and len(t) > 1
    }


def _corpus_generic_tokens(
    fixed_filtered_token_sets: list[set[str]],
    threshold: float = CORPUS_GENERIC_FREQUENCY_THRESHOLD,
) -> set[str]:
    """Tokens appearing in MORE than `threshold` of this request's items (of
    one evidence_kind) - determined per-request, never a global stopword."""
    n = len(fixed_filtered_token_sets)
    if n == 0:
        return set()
    counts: dict[str, int] = {}
    for tokens in fixed_filtered_token_sets:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    return {token for token, count in counts.items() if (count / n) > threshold}


def _canonical_colors(tokens: set[str]) -> set[str]:
    return {_COLOR_CANON[t] for t in tokens if t in _COLOR_CANON}


def _color_conflict(colors_a: set[str], colors_b: set[str]) -> bool:
    """Colors are a veto-only signal: conflict requires BOTH sides to name a
    known color AND those colors to be fully disjoint. One-sided color
    mentions are a no-op; shared colors never block."""
    if not colors_a or not colors_b:
        return False
    return colors_a.isdisjoint(colors_b)


def _scene_boundary_between(
    a: NormalizedVideoEvidence, b: NormalizedVideoEvidence, boundaries_us: list[int]
) -> bool:
    """True if any known scene boundary timestamp falls between a and b's
    temporal footprints (order-independent). A boundary here means the two
    chunks straddle a real detected cut, which vetoes the relation
    regardless of any positive anchor."""
    if a.start_us <= b.start_us:
        earlier, later = a, b
    else:
        earlier, later = b, a
    lo = min(earlier.end_us, later.start_us)
    hi = max(earlier.end_us, later.start_us)
    return any(lo <= ts <= hi for ts in boundaries_us)


class _ContinuityItem:
    """Precomputed, per-evidence continuity inputs. Internal to this module -
    never exposed to the Director or persisted."""

    __slots__ = ("evidence", "chunk_index", "anchors", "colors")

    def __init__(
        self,
        evidence: NormalizedVideoEvidence,
        chunk_index: int,
        anchors: set[str],
        colors: set[str],
    ) -> None:
        self.evidence = evidence
        self.chunk_index = chunk_index
        self.anchors = anchors
        self.colors = colors


def _build_groups_for_kind(
    items: list[_ContinuityItem], boundaries_us: list[int]
) -> list[list[_ContinuityItem]]:
    """Precision-first, transitivity-safe group construction for a single
    evidence_kind's items (already sorted canonically by (start_us,
    evidence_id) and already restricted to same asset_id).

    Bounded complexity (section 22): candidates are only ever generated
    between an item and the OPEN chains whose last member sits in the
    immediately preceding chunk index - looked up via `chains_by_last_chunk`
    - never a global all-pairs scan. Cost is O(items * avg_items_per_chunk),
    not O(N^2).

    Transitivity guard (section 16): extending a chain re-checks the color
    veto against EVERY existing member, not just the immediate neighbor - a
    naive union-find over shared anchors (e.g. "car") would otherwise chain
    unrelated items (red car / car turns left / blue car) into one group.

    Scene boundary veto (section 15) is checked per candidate LINK (against
    the immediate neighbor only, at join time) rather than post-hoc on the
    finished chain - a boundary between two specific members must block only
    that link, never discard an otherwise-valid earlier/later sub-chain.
    """
    chains: list[list[_ContinuityItem]] = []
    chains_by_last_chunk: dict[int, list[list[_ContinuityItem]]] = {}

    for item in items:
        joined = False
        for chain in chains_by_last_chunk.get(item.chunk_index - 1, []):
            last = chain[-1]
            # Bucket entries are never pruned from their OLD key when a chain
            # is extended (append-only, for O(1) amortized bookkeeping), so a
            # bucket can hold stale references whose current last member has
            # since moved to a later chunk. Re-verify the actual adjacency
            # here rather than trusting bucket placement alone - this is what
            # keeps "chunks consecutive i -> i+1" a hard invariant regardless
            # of that staleness.
            if last.chunk_index != item.chunk_index - 1:
                continue
            if not (last.anchors & item.anchors):
                continue
            if _scene_boundary_between(last.evidence, item.evidence, boundaries_us):
                continue
            if _color_conflict(last.colors, item.colors):
                continue
            # Group-wide veto: candidate must not conflict with ANY existing
            # member, not only the immediate neighbor.
            if any(_color_conflict(member.colors, item.colors) for member in chain):
                continue
            chain.append(item)
            joined = True
            break
        if not joined:
            chain = [item]
            chains.append(chain)

        # Re-index chains_by_last_chunk for this chain's (possibly new) last
        # chunk index. A chain that just gained `item` is removed from its
        # old bucket implicitly (buckets are rebuilt lazily below) and added
        # to the new one.
        bucket = chains_by_last_chunk.setdefault(item.chunk_index, [])
        bucket.append(chain if joined else chains[-1])

    return [chain for chain in chains if len(chain) >= 2]


def annotate_continuity(
    evidence_list: list[NormalizedVideoEvidence],
    via_result: VideoIntelligenceResult | None,
) -> list[NormalizedVideoEvidence]:
    """Populate continuity_group_id/continuity_basis on EVENT/OBSERVATION
    evidence that forms a valid cross-chunk continuity group. Must run AFTER
    normalize_via_result() and deduplicate_evidence() (section 3) - never
    before, so continuity never links duplicate evidence.

    SCOPE: EVENT and OBSERVATION only. OBJECT and SCENE evidence pass through
    completely untouched (never read, never grouped).

    STRICT scene policy (section 12): a group can only be formed when scene
    cache data is a validated HIT for `via_result`'s EXACT asset_id/start_us/
    end_us/threshold/detector-version (section 13). No detector is ever
    invoked here - only the existing persisted scene cache is read. If
    `via_result` is None (evidence came in pre-normalized, e.g. legacy
    callers that never had a VideoIntelligenceResult to consult) or the cache
    lookup/read is anything but a validated hit, continuity abstains
    entirely and every item's continuity fields stay None - the same
    behavior FULL_SCAN/TARGETED_SCAN naturally get today since Phase 4.2
    only ever populates the scene cache for COARSE_SCAN.
    """
    if via_result is None:
        return evidence_list

    lookup = resolve_scene_cache_lookup(via_result.asset_id)
    if lookup is None:
        return evidence_list

    boundaries = read_cached_boundaries(
        lookup.asset_metadata,
        fingerprint=lookup.fingerprint,
        threshold=DEFAULT_SCENE_CHANGE_THRESHOLD,
        start_us=via_result.start_us,
        end_us=via_result.end_us,
    )
    if boundaries is None:
        # Cache miss / mismatched range / corrupt entry - never fabricate an
        # association without validated scene data.
        return evidence_list

    boundaries_us = sorted(b.timestamp_us for b in boundaries)

    eligible_kinds = (EvidenceKind.EVENT, EvidenceKind.OBSERVATION)
    groups_by_evidence_id: dict[str, list[NormalizedVideoEvidence]] = {}

    for kind in eligible_kinds:
        kind_items = [
            e
            for e in evidence_list
            if e.evidence_kind == kind and e.asset_id == via_result.asset_id
        ]
        if len(kind_items) < 2:
            continue

        # Canonical order (section 17): deterministic regardless of input
        # ordering, chunk-completion ordering, or concurrency.
        kind_items = sorted(kind_items, key=lambda e: (e.start_us, e.evidence_id))

        fixed_filtered = {e.evidence_id: _fixed_filtered_tokens(e.description) for e in kind_items}
        corpus_generic = _corpus_generic_tokens(list(fixed_filtered.values()))

        continuity_items: list[_ContinuityItem] = []
        for e in kind_items:
            chunk_index = _extract_chunk_index(e.chunk_id)
            if chunk_index is None:
                # No reliable chunk index - safe abstention, never invent one.
                continue
            anchors = fixed_filtered[e.evidence_id] - corpus_generic
            colors = _canonical_colors(_tokenize(e.description))
            continuity_items.append(_ContinuityItem(e, chunk_index, anchors, colors))

        for chain in _build_groups_for_kind(continuity_items, boundaries_us):
            member_evidence = [c.evidence for c in chain]
            group_id = compute_continuity_group_id([e.evidence_id for e in member_evidence])
            for e in member_evidence:
                groups_by_evidence_id[e.evidence_id] = (group_id,)

    if not groups_by_evidence_id:
        return evidence_list

    result: list[NormalizedVideoEvidence] = []
    for e in evidence_list:
        grouping = groups_by_evidence_id.get(e.evidence_id)
        if grouping is None:
            result.append(e)
            continue
        (group_id,) = grouping
        result.append(
            e.model_copy(
                update={
                    "continuity_group_id": group_id,
                    "continuity_basis": "temporal_anchor_continuity",
                }
            )
        )
    return result