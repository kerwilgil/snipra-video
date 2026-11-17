"""Deterministic highlight scoring and relative ranking (SNIPRA v1.5C).

Consumes existing evidence (ClipCandidate, VIA NormalizedVideoEvidence, transcript)
and produces an overall quality score with traceable breakdown + deterministic relative rank.
No LLM calls, no mutations, pure intelligence for downstream consumers.

Key design invariants:
- Input order independence: identical candidate sets produce identical scores/ranks
- No double counting: signal bonus already baked into base score is not re-added
- No saturation: normalized positive weights sum to 1.0; penalty applied after
- Symmetric redundancy: penalty depends on mutual overlap, not processing order
- Single-asset context: all candidates in a call belong to one video/job
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from app.editor.via_evidence import NormalizedVideoEvidence
from app.providers.base import (
    DURATION_PRESETS,
    ClipCandidate,
    ScoreContribution,
    overlap_ratio,
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (explicit normalized weights, no magic numbers)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScoringWeights:
    """
    Normalized component weights. Positive weights sum to 1.0.
    Redundancy penalty is applied AFTER the weighted sum.
    signal_bonus and continuity are kept at 0.0 (neutral) to avoid double-counting
    and unsafe interpretation respectively.
    """
    base: float = 0.50              # existing candidate.score (LLM/fallback/signal)
    event_strength: float = 0.20    # VIA EVENT confidence + payoff + reaction
    transcript_relevance: float = 0.15  # semantic match, payoff terms
    visual_novelty: float = 0.10    # VIA evidence_quality, visual_novelty
    duration_quality: float = 0.05  # within preset ranges
    continuity: float = 0.00        # neutral - semantics not safely interpretable at 1.5C
    signal_bonus: float = 0.00      # neutral - already represented in base score
    redundancy_penalty: float = -0.10  # applied after weighted sum (negative)

    def __post_init__(self) -> None:
        positive_sum = (
            self.base + self.event_strength + self.transcript_relevance +
            self.visual_novelty + self.duration_quality + self.continuity +
            self.signal_bonus
        )
        # Allow small floating point tolerance
        if abs(positive_sum - 1.0) > 1e-6:
            raise ValueError(f"Positive weights must sum to 1.0, got {positive_sum}")

DEFAULT_WEIGHTS = ScoringWeights()

# Payoff terms from gaming_edits.py (reused for transcript relevance)
TRANSCRIPT_PAYOFF_TERMS = frozenset({
    "estoy a uno", "ganamos", "gané", "gane", "victoria", "último", "ultimo",
    "ellos, papá", "ellos papa", "lo maté", "lo mate", "clutch",
})

REWARD_TERMS = frozenset({
    "victoria", "victory", "ganaste", "winner", "baja final", "final kill",
    "killcam", "jugada de la partida", "play of the game", "elimination",
})


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ComponentScore:
    """Single scoring component with traceability."""
    name: str
    raw_value: float        # 0.0–1.0 raw signal
    weight: float           # configured weight
    contribution: float     # raw_value * weight
    evidence_refs: list[str]  # evidence_ids or descriptive refs
    note: str = ""


@dataclass(frozen=True)
class ScoredCandidate:
    """Enriched candidate with overall score, breakdown, and rank."""
    candidate: ClipCandidate
    overall_score: float
    rank: int | None
    component_scores: tuple[ComponentScore, ...]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _stable_candidate_id(candidate: ClipCandidate) -> str:
    """Deterministic identity for tie-breaking: hash of immutable fields."""
    parts = [
        str(candidate.start),
        str(candidate.end),
        candidate.title or "",
        candidate.hook or "",
        candidate.reason or "",
    ]
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _candidate_key(candidate: ClipCandidate) -> str:
    """Stable dict key for a candidate (uses stable_id)."""
    return _stable_candidate_id(candidate)


def _normalize_text(text: str) -> str:
    """Lowercase, remove accents, remove punctuation, collapse whitespace."""
    import re
    import unicodedata
    folded = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(c for c in folded if not unicodedata.combining(c))
    ascii_text = re.sub(r"[^\w\s]", "", ascii_text)
    return " ".join(ascii_text.lower().split())


def _compute_symmetric_redundancy(
    candidates: list[ClipCandidate],
) -> dict[str, float]:
    """
    Compute symmetric redundancy penalty for each candidate.

    For each candidate, find the maximum overlap with ANY other candidate.
    If max_overlap > 0.5, penalty = (max_overlap - 0.5) * 2.0 (scales 0..1).
    This is order-independent and symmetric - both overlapping candidates
    receive the same penalty.
    """
    n = len(candidates)
    if n <= 1:
        return {_candidate_key(c): 0.0 for c in candidates}

    # Compute overlap matrix
    max_overlap_per_candidate = {}
    for i, c1 in enumerate(candidates):
        max_ov = 0.0
        for j, c2 in enumerate(candidates):
            if i == j:
                continue
            ov = overlap_ratio(c1, c2)
            if ov > max_ov:
                max_ov = ov
        max_overlap_per_candidate[_candidate_key(c1)] = max_ov

    # Convert to penalty: 0 for overlap <= 0.5, linear 0..1 for overlap > 0.5
    penalties = {}
    for key, max_ov in max_overlap_per_candidate.items():
        if max_ov > 0.5:
            # Scale: 0.5 -> 0.0, 1.0 -> 1.0
            penalties[key] = (max_ov - 0.5) * 2.0
        else:
            penalties[key] = 0.0

    return penalties


# ──────────────────────────────────────────────────────────────────────────────
# Evidence attachment
# ──────────────────────────────────────────────────────────────────────────────

def _attach_via_evidence(
    candidates: list[ClipCandidate],
    via_evidence: list[NormalizedVideoEvidence],
) -> dict[str, list[NormalizedVideoEvidence]]:
    """Attach VIA evidence to candidates by temporal overlap.

    All candidates in a single score_candidates call belong to the same
    video/asset (the job context), so no asset_id filtering is needed.
    """
    attached: dict[str, list[NormalizedVideoEvidence]] = defaultdict(list)

    for candidate in candidates:
        cand_key = _candidate_key(candidate)
        cand_start_us = int(round(candidate.start * 1_000_000))
        cand_end_us = int(round(candidate.end * 1_000_000))

        for via in via_evidence:
            overlap_us = max(0, min(cand_end_us, via.end_us) - max(cand_start_us, via.start_us))
            if overlap_us == 0:
                continue

            cand_dur_us = cand_end_us - cand_start_us
            via_dur_us = via.end_us - via.start_us
            min_dur_us = min(cand_dur_us, via_dur_us)
            if min_dur_us == 0 or (overlap_us / min_dur_us) < 0.3:
                continue

            attached[cand_key].append(via)

    return attached


# ──────────────────────────────────────────────────────────────────────────────
# Component scoring functions
# ──────────────────────────────────────────────────────────────────────────────

def _score_base(candidate: ClipCandidate, weights: ScoringWeights) -> ComponentScore:
    """Base score from candidate (LLM, fallback, or signal-derived)."""
    return ComponentScore(
        name="base_score",
        raw_value=candidate.score,
        weight=weights.base,
        contribution=candidate.score * weights.base,
        evidence_refs=[],
        note="Base score from detector (LLM/fallback/signals)",
    )


def _score_event_strength(
    candidate: ClipCandidate,
    attached_evidence: list[NormalizedVideoEvidence],
    weights: ScoringWeights,
) -> ComponentScore:
    """Event strength from VIA EVENT evidence: confidence, payoff, reaction."""
    if not attached_evidence:
        return ComponentScore(
            name="event_strength",
            raw_value=0.0,
            weight=weights.event_strength,
            contribution=0.0,
            evidence_refs=[],
            note="No VIA evidence attached",
        )

    event_evidence = [e for e in attached_evidence if e.evidence_kind.value == "event"]
    if not event_evidence:
        return ComponentScore(
            name="event_strength",
            raw_value=0.0,
            weight=weights.event_strength,
            contribution=0.0,
            evidence_refs=[e.evidence_id for e in attached_evidence],
            note="No EVENT-kind evidence attached",
        )

    # Average confidence of EVENT evidence
    avg_confidence = sum(e.confidence_original for e in event_evidence) / len(event_evidence)

    # Payoff bonus
    payoff_bonus = 0.0
    payoff_count = sum(1 for e in event_evidence if e.payoff_present)
    if payoff_count:
        payoff_bonus = min(0.3, payoff_count * 0.15)

    # Reaction bonus
    reaction_bonus = 0.0
    reaction_count = sum(1 for e in event_evidence if e.reaction_present)
    if reaction_count:
        reaction_bonus = min(0.2, reaction_count * 0.1)

    # Evidence quality bonus
    quality_bonus = 0.0
    quality_map = {"high": 0.15, "medium": 0.08, "low": 0.0}
    for e in event_evidence:
        if e.evidence_quality:
            quality_bonus += quality_map.get(e.evidence_quality.lower(), 0.0)
    quality_bonus = min(0.2, quality_bonus)

    raw = _clamp01(avg_confidence + payoff_bonus + reaction_bonus + quality_bonus)

    return ComponentScore(
        name="event_strength",
        raw_value=raw,
        weight=weights.event_strength,
        contribution=raw * weights.event_strength,
        evidence_refs=[e.evidence_id for e in event_evidence],
        note=f"avg_conf={avg_confidence:.2f}, payoff={payoff_count}, reaction={reaction_count}, quality_bonus={quality_bonus:.2f}",
    )


def _score_transcript_relevance(
    candidate: ClipCandidate,
    transcript_segments: list[dict],
    weights: ScoringWeights,
) -> ComponentScore:
    """Transcript relevance: payoff terms, semantic density in candidate window."""
    if not transcript_segments:
        return ComponentScore(
            name="transcript_relevance",
            raw_value=0.0,
            weight=weights.transcript_relevance,
            contribution=0.0,
            evidence_refs=[],
            note="No transcript segments available",
        )

    # Collect transcript text within candidate window
    relevant_text = []
    for seg in transcript_segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        if seg_end <= candidate.start or seg_start >= candidate.end:
            continue
        text = str(seg.get("text", "")).strip()
        if text:
            relevant_text.append(text)

    if not relevant_text:
        return ComponentScore(
            name="transcript_relevance",
            raw_value=0.0,
            weight=weights.transcript_relevance,
            contribution=0.0,
            evidence_refs=[],
            note="No transcript text in candidate window",
        )

    joined = " ".join(relevant_text)
    normalized = _normalize_text(joined)

    # Payoff term hits
    payoff_hits = sum(1 for term in TRANSCRIPT_PAYOFF_TERMS if term in normalized)
    payoff_score = min(1.0, payoff_hits * 0.4)

    # Reward term hits (visual payoff terms)
    reward_hits = sum(1 for term in REWARD_TERMS if term in normalized)
    reward_score = min(1.0, reward_hits * 0.3)

    # Density: chars per second in window
    window_dur = max(0.1, candidate.end - candidate.start)
    char_density = len(joined) / window_dur
    density_score = min(1.0, char_density / 200.0)  # ~200 chars/sec = high density

    raw = _clamp01(0.5 * payoff_score + 0.3 * reward_score + 0.2 * density_score)

    return ComponentScore(
        name="transcript_relevance",
        raw_value=raw,
        weight=weights.transcript_relevance,
        contribution=raw * weights.transcript_relevance,
        evidence_refs=[f"transcript:{len(relevant_text)}_segments"],
        note=f"payoff_hits={payoff_hits}, reward_hits={reward_hits}, density={char_density:.1f} chars/s",
    )


def _score_signal_bonus(candidate: ClipCandidate, weights: ScoringWeights) -> ComponentScore:
    """
    Signal bonus from score_breakdown (already applied by signals.py).

    This component is INFORMATIONAL ONLY (weight 0.0) to avoid double-counting.
    The signals.py bonus is already baked into candidate.score.
    """
    signal_contributions = [
        c for c in candidate.score_breakdown
        if c.signal == "señales_audio_escena"
    ]
    if not signal_contributions:
        return ComponentScore(
            name="signal_bonus",
            raw_value=0.0,
            weight=weights.signal_bonus,  # 0.0
            contribution=0.0,
            evidence_refs=[],
            note="No signal contribution in breakdown",
        )

    total_bonus = sum(c.contribution for c in signal_contributions)
    # Normalize: max observed bonus is ~0.18 (from signals.py)
    raw = _clamp01(total_bonus / 0.18)

    return ComponentScore(
        name="signal_bonus",
        raw_value=raw,
        weight=weights.signal_bonus,  # 0.0 - informational only
        contribution=0.0,  # always 0 since weight is 0
        evidence_refs=[c.note for c in signal_contributions],
        note=f"signal_bonus_applied={total_bonus:.3f} (already in base score, not re-added)",
    )


def _score_visual_novelty(
    candidate: ClipCandidate,
    attached_evidence: list[NormalizedVideoEvidence],
    weights: ScoringWeights,
) -> ComponentScore:
    """Visual novelty from VIA evidence: evidence_quality, visual_novelty."""
    if not attached_evidence:
        return ComponentScore(
            name="visual_novelty",
            raw_value=0.0,
            weight=weights.visual_novelty,
            contribution=0.0,
            evidence_refs=[],
            note="No VIA evidence attached",
        )

    quality_map = {"high": 1.0, "medium": 0.6, "low": 0.2}
    total_quality = 0.0
    count = 0
    novelty_bonus = 0.0

    for e in attached_evidence:
        if e.evidence_quality:
            total_quality += quality_map.get(e.evidence_quality.lower(), 0.5)
            count += 1
        if e.visual_novelty and e.visual_novelty.lower() == "high":
            novelty_bonus += 0.15

    avg_quality = (total_quality / count) if count else 0.0
    raw = _clamp01(avg_quality + novelty_bonus)

    return ComponentScore(
        name="visual_novelty",
        raw_value=raw,
        weight=weights.visual_novelty,
        contribution=raw * weights.visual_novelty,
        evidence_refs=[e.evidence_id for e in attached_evidence],
        note=f"avg_quality={avg_quality:.2f}, novelty_bonus={novelty_bonus:.2f}",
    )


def _score_continuity(
    candidate: ClipCandidate,
    attached_evidence: list[NormalizedVideoEvidence],
    weights: ScoringWeights,
) -> ComponentScore:
    """
    Continuity component - NEUTRAL (weight 0.0) at 1.5C.

    continuity_group_id in VIA indicates cross-chunk grouping of evidence
    describing the same ongoing occurrence. However, the semantics of
    "number of distinct groups" != "coherent continuity". A single coherent
    event may have one group; multiple unrelated groups don't mean better
    continuity. Until semantics are clarified, keep neutral.
    """
    if not attached_evidence:
        return ComponentScore(
            name="continuity",
            raw_value=0.0,
            weight=weights.continuity,  # 0.0
            contribution=0.0,
            evidence_refs=[],
            note="No VIA evidence attached",
        )

    continuity_groups = set()
    for e in attached_evidence:
        if e.continuity_group_id:
            continuity_groups.add(e.continuity_group_id)

    # Informational raw_value (not used since weight=0)
    raw = _clamp01(len(continuity_groups) / 3.0) if continuity_groups else 0.0

    return ComponentScore(
        name="continuity",
        raw_value=raw,
        weight=weights.continuity,  # 0.0
        contribution=0.0,
        evidence_refs=list(continuity_groups),
        note=f"continuity_groups={len(continuity_groups)} (neutral weight)",
    )


def _score_duration_quality(candidate: ClipCandidate, weights: ScoringWeights) -> ComponentScore:
    """Duration quality: how well duration fits social/standard/long presets."""
    dur = candidate.duration
    if dur <= 0:
        return ComponentScore(
            name="duration_quality",
            raw_value=0.0,
            weight=weights.duration_quality,
            contribution=0.0,
            evidence_refs=[],
            note="Invalid duration",
        )

    best_fit = 0.0
    for preset in DURATION_PRESETS.values():
        pmin, pmax = preset["min"], preset["max"]
        if pmin <= dur <= pmax:
            fit = 1.0
        elif dur < pmin:
            fit = max(0.0, dur / pmin)
        else:
            fit = max(0.0, 1.0 - (dur - pmax) / pmax)
        best_fit = max(best_fit, fit)

    return ComponentScore(
        name="duration_quality",
        raw_value=best_fit,
        weight=weights.duration_quality,
        contribution=best_fit * weights.duration_quality,
        evidence_refs=[f"duration={dur:.1f}s"],
        note=f"best_preset_fit={best_fit:.2f}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main scoring pipeline
# ──────────────────────────────────────────────────────────────────────────────

def score_candidates(
    candidates: list[ClipCandidate],
    via_evidence: list[NormalizedVideoEvidence] | None = None,
    transcript_segments: list[dict] | None = None,
    weights: ScoringWeights | None = None,
) -> list[ScoredCandidate]:
    """
    Score and rank candidates deterministically.

    Args:
        candidates: List of ClipCandidate to score (all from same video/job)
        via_evidence: Optional VIA normalized evidence for event/visual/continuity signals
        transcript_segments: Optional Whisper segments for transcript relevance
        weights: Optional custom ScoringWeights (uses DEFAULT_WEIGHTS if None)

    Returns:
        List of ScoredCandidate sorted by overall_score DESC, with rank assigned.
        Rank is 1-indexed.
    """
    if not candidates:
        return []

    w = weights or DEFAULT_WEIGHTS
    via_evidence = via_evidence or []
    transcript_segments = transcript_segments or []

    # 1. Attach VIA evidence to candidates (all same asset context)
    attached_evidence = _attach_via_evidence(candidates, via_evidence)

    # 2. Compute symmetric redundancy penalties (order-independent)
    redundancy_penalties = _compute_symmetric_redundancy(candidates)

    # 3. Compute component scores for each candidate
    scored_list: list[ScoredCandidate] = []

    for candidate in candidates:
        cand_key = _candidate_key(candidate)
        attached = attached_evidence.get(cand_key, [])

        components = [
            _score_base(candidate, w),
            _score_event_strength(candidate, attached, w),
            _score_transcript_relevance(candidate, transcript_segments, w),
            _score_signal_bonus(candidate, w),  # weight 0 - informational
            _score_visual_novelty(candidate, attached, w),
            _score_continuity(candidate, attached, w),  # weight 0 - neutral
            _score_duration_quality(candidate, w),
        ]

        # Weighted sum (positive weights sum to 1.0)
        weighted_sum = sum(c.contribution for c in components)

        # Apply symmetric redundancy penalty (negative weight applied after)
        redundancy_raw = redundancy_penalties.get(cand_key, 0.0)
        redundancy_contribution = redundancy_raw * w.redundancy_penalty
        overall = _clamp01(weighted_sum + redundancy_contribution)

        # Add redundancy penalty as informational component for observability
        if redundancy_raw > 0.0:
            components = list(components) + [ComponentScore(
                name="redundancy_penalty",
                raw_value=redundancy_raw,
                weight=w.redundancy_penalty,
                contribution=redundancy_contribution,
                evidence_refs=[],
                note=f"max_overlap_with_peer={redundancy_raw:.2f} (symmetric)",
            )]

        scored_list.append(ScoredCandidate(
            candidate=candidate,
            overall_score=overall,
            rank=None,  # assigned after sorting
            component_scores=tuple(components),
        ))

    # 4. Assign relative ranks (deterministic sort)
    # Sort by: overall_score DESC, start ASC, stable_id ASC
    scored_list.sort(key=lambda sc: (
        -sc.overall_score,
        sc.candidate.start,
        _stable_candidate_id(sc.candidate),
    ))

    final_list: list[ScoredCandidate] = []
    for rank, sc in enumerate(scored_list, start=1):
        final_list.append(ScoredCandidate(
            candidate=sc.candidate,
            overall_score=sc.overall_score,
            rank=rank,
            component_scores=sc.component_scores,
        ))

    return final_list


# ──────────────────────────────────────────────────────────────────────────────
# Enrichment: apply scores back to ClipCandidate (for persistence/API)
# ──────────────────────────────────────────────────────────────────────────────

def enrich_candidates_with_scores(
    scored: list[ScoredCandidate],
) -> list[ClipCandidate]:
    """Create new ClipCandidate instances with updated score and score_breakdown.

    Preserves all original fields; only updates:
    - score → overall_score
    - score_breakdown → appended with component ScoreContributions
    """
    enriched: list[ClipCandidate] = []

    for sc in scored:
        new_breakdown = list(sc.candidate.score_breakdown)

        for comp in sc.component_scores:
            if comp.weight == 0 and comp.name not in ("signal_bonus", "continuity"):
                continue
            new_breakdown.append(ScoreContribution(
                signal=f"highlight_{comp.name}",
                stage="highlight_scoring.score_candidates",
                score_before=sc.candidate.score,
                contribution=round(comp.contribution, 4),
                score_after=sc.overall_score,
                at=sc.candidate.start,
                note=comp.note,
                metadata={
                    "component": comp.name,
                    "raw_value": round(comp.raw_value, 4),
                    "weight": comp.weight,
                    "evidence_refs": comp.evidence_refs,
                },
            ))

        enriched.append(sc.candidate.model_copy(update={
            "score": round(sc.overall_score, 4),
            "score_breakdown": new_breakdown,
        }))

    return enriched


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: score and enrich in one call
# ──────────────────────────────────────────────────────────────────────────────

def score_and_enrich(
    candidates: list[ClipCandidate],
    via_evidence: list[NormalizedVideoEvidence] | None = None,
    transcript_segments: list[dict] | None = None,
    weights: ScoringWeights | None = None,
) -> tuple[list[ScoredCandidate], list[ClipCandidate]]:
    """Score candidates and return both ScoredCandidate (for inspection) and enriched ClipCandidate (for persistence)."""
    scored = score_candidates(candidates, via_evidence, transcript_segments, weights)
    enriched = enrich_candidates_with_scores(scored)
    return scored, enriched