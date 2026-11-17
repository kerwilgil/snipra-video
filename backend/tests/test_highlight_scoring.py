"""Tests for highlight_scoring.py (SNIPRA v1.5C).

Covers deterministic scoring, relative ranking, tie-breaking, missing evidence,
edge cases, breakdown traceability, and input permutation invariance.
"""

import itertools

import pytest
from app.editor.via_evidence import NormalizedVideoEvidence, EvidenceKind
from app.pipeline.highlight_scoring import (
    ScoringWeights,
    DEFAULT_WEIGHTS,
    ComponentScore,
    ScoredCandidate,
    score_candidates,
    enrich_candidates_with_scores,
    score_and_enrich,
    _attach_via_evidence,
    _score_base,
    _score_event_strength,
    _score_transcript_relevance,
    _score_signal_bonus,
    _score_visual_novelty,
    _score_continuity,
    _score_duration_quality,
    _stable_candidate_id,
    _normalize_text,
    _compute_symmetric_redundancy,
)
from app.providers.base import ClipCandidate, ScoreContribution, overlap_ratio


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def make_candidate(
    start: float,
    end: float,
    score: float = 0.5,
    title: str = "",
    reason: str = "",
    hook: str = "",
    breakdown: list[ScoreContribution] | None = None,
) -> ClipCandidate:
    c = ClipCandidate(start=start, end=end, score=score, title=title, reason=reason, hook=hook)
    if breakdown:
        c = c.model_copy(update={"score_breakdown": breakdown})
    return c


def make_via_evidence(
    asset_id: str,
    start_s: float,
    end_s: float,
    kind: str = "event",
    confidence: float = 0.8,
    payoff: bool = False,
    reaction: bool = False,
    quality: str = "medium",
    novelty: str | None = None,
    continuity_id: str | None = None,
) -> NormalizedVideoEvidence:
    return NormalizedVideoEvidence(
        evidence_id=f"ev_{asset_id}_{int(start_s)}",
        asset_id=asset_id,
        start_us=int(start_s * 1_000_000),
        end_us=int(end_s * 1_000_000),
        evidence_kind=EvidenceKind(kind),
        nature="inference" if kind == "event" else "observation",
        description=f"Test {kind}",
        confidence_original=confidence,
        provider="test",
        model="test",
        payoff_present=payoff,
        reaction_present=reaction,
        evidence_quality=quality,
        visual_novelty=novelty,
        continuity_group_id=continuity_id,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper function tests
# ──────────────────────────────────────────────────────────────────────────────

def test_stable_candidate_id_deterministic():
    c1 = make_candidate(10.0, 20.0, 0.7, "title", "reason")
    c2 = make_candidate(10.0, 20.0, 0.7, "title", "reason")
    c3 = make_candidate(10.0, 20.0, 0.8, "title", "reason")  # different score (not in stable ID)
    c4 = make_candidate(15.0, 25.0, 0.7, "title", "reason")  # different time

    assert _stable_candidate_id(c1) == _stable_candidate_id(c2)
    assert _stable_candidate_id(c1) == _stable_candidate_id(c3)  # score not in stable ID
    assert _stable_candidate_id(c1) != _stable_candidate_id(c4)


def test_normalize_text():
    assert _normalize_text("Hola Mundo!") == "hola mundo"
    assert _normalize_text("  Múltiples   espacios  ") == "multiples espacios"
    assert _normalize_text("VICTORIA") == "victoria"
    assert _normalize_text("") == ""


def test_overlap_ratio_from_base():
    """Test that overlap_ratio is correctly imported from providers.base."""
    a = make_candidate(10.0, 20.0)  # duration 10
    b = make_candidate(15.0, 25.0)  # duration 10, overlap 5
    assert overlap_ratio(a, b) == pytest.approx(0.5)

    c = make_candidate(10.0, 30.0)  # duration 20
    d = make_candidate(15.0, 20.0)  # duration 5, overlap 5
    assert overlap_ratio(c, d) == pytest.approx(1.0)  # 5/5 = 1.0

    e = make_candidate(0.0, 5.0)
    f = make_candidate(10.0, 15.0)
    assert overlap_ratio(e, f) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Symmetric redundancy tests (P0-1)
# ──────────────────────────────────────────────────────────────────────────────

def test_symmetric_redundancy_single_candidate():
    c = make_candidate(0, 10, 0.5)
    penalties = _compute_symmetric_redundancy([c])
    assert penalties[_stable_candidate_id(c)] == 0.0


def test_symmetric_redundancy_no_overlap():
    c1 = make_candidate(0, 10, 0.9)
    c2 = make_candidate(20, 30, 0.8)
    penalties = _compute_symmetric_redundancy([c1, c2])
    assert penalties[_stable_candidate_id(c1)] == 0.0
    assert penalties[_stable_candidate_id(c2)] == 0.0


def test_symmetric_redundancy_overlap_50_percent():
    """Exactly 50% overlap → no penalty (threshold is > 0.5)."""
    c1 = make_candidate(0, 10, 0.9)
    c2 = make_candidate(5, 15, 0.8)  # 50% overlap
    penalties = _compute_symmetric_redundancy([c1, c2])
    assert penalties[_stable_candidate_id(c1)] == 0.0
    assert penalties[_stable_candidate_id(c2)] == 0.0


def test_symmetric_redundancy_overlap_above_50_percent():
    """Overlap > 50% → both candidates penalized equally."""
    c1 = make_candidate(0, 10, 0.9)
    c2 = make_candidate(2, 12, 0.8)  # 80% overlap
    penalties = _compute_symmetric_redundancy([c1, c2])
    # (0.8 - 0.5) * 2 = 0.6
    assert penalties[_stable_candidate_id(c1)] == pytest.approx(0.6)
    assert penalties[_stable_candidate_id(c2)] == pytest.approx(0.6)


def test_symmetric_redundancy_three_candidates_chain():
    """Chain overlap: A-B overlap, B-C overlap, A-C no overlap."""
    c1 = make_candidate(0, 10, 0.9)      # A
    c2 = make_candidate(4, 14, 0.8)      # B - overlaps A (60%) and C (60%)
    c3 = make_candidate(8, 18, 0.7)      # C
    penalties = _compute_symmetric_redundancy([c1, c2, c3])
    # c1: max overlap with c2 = 0.6 → penalty = (0.6-0.5)*2 = 0.2
    # c2: max overlap with c1 or c3 = 0.6 → penalty = 0.2
    # c3: max overlap with c2 = 0.6 → penalty = 0.2
    assert penalties[_stable_candidate_id(c1)] == pytest.approx(0.2)
    assert penalties[_stable_candidate_id(c2)] == pytest.approx(0.2)
    assert penalties[_stable_candidate_id(c3)] == pytest.approx(0.2)


def test_symmetric_redundancy_input_permutation_invariance():
    """Same candidate set, different input orders → identical penalties."""
    c1 = make_candidate(0, 10, 0.9)
    c2 = make_candidate(2, 12, 0.8)
    c3 = make_candidate(20, 30, 0.5)

    for perm in itertools.permutations([c1, c2, c3]):
        penalties = _compute_symmetric_redundancy(list(perm))
        # All permutations should give same penalties per candidate key
        assert penalties[_stable_candidate_id(c1)] == pytest.approx(0.6)
        assert penalties[_stable_candidate_id(c2)] == pytest.approx(0.6)
        assert penalties[_stable_candidate_id(c3)] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Component scoring tests
# ──────────────────────────────────────────────────────────────────────────────

def test_score_base():
    c = make_candidate(0, 10, 0.75)
    comp = _score_base(c, DEFAULT_WEIGHTS)
    assert comp.name == "base_score"
    assert comp.raw_value == 0.75
    assert comp.contribution == 0.75 * DEFAULT_WEIGHTS.base


def test_score_event_strength_no_evidence():
    c = make_candidate(0, 10, 0.5)
    comp = _score_event_strength(c, [], DEFAULT_WEIGHTS)
    assert comp.raw_value == 0.0
    assert comp.contribution == 0.0


def test_score_event_strength_with_payoff_reaction():
    c = make_candidate(0, 10, 0.5)
    ev = [
        make_via_evidence("asset1", 0, 10, confidence=0.9, payoff=True, reaction=True, quality="high"),
        make_via_evidence("asset1", 0, 10, confidence=0.8, payoff=False, reaction=False, quality="medium"),
    ]
    comp = _score_event_strength(c, ev, DEFAULT_WEIGHTS)
    assert comp.raw_value > 0.0
    assert comp.contribution == comp.raw_value * DEFAULT_WEIGHTS.event_strength
    assert len(comp.evidence_refs) == 2


def test_score_transcript_relevance_payoff_terms():
    c = make_candidate(10, 20, 0.5)
    segments = [
        {"start": 10.0, "end": 15.0, "text": "Estoy a uno, vamos a ganar"},
        {"start": 15.0, "end": 20.0, "text": "Victoria final, clutch"},
    ]
    comp = _score_transcript_relevance(c, segments, DEFAULT_WEIGHTS)
    assert comp.raw_value > 0.0
    assert "payoff_hits=2" in comp.note or "payoff_hits=3" in comp.note


def test_score_transcript_relevance_no_segments():
    c = make_candidate(0, 10, 0.5)
    comp = _score_transcript_relevance(c, [], DEFAULT_WEIGHTS)
    assert comp.raw_value == 0.0


def test_score_signal_bonus_informational_only():
    """signal_bonus has weight 0.0 - contribution is always 0 (already in base)."""
    c = make_candidate(0, 10, 0.5, breakdown=[
        ScoreContribution(signal="señales_audio_escena", contribution=0.12, score_before=0.5, score_after=0.62),
        ScoreContribution(signal="other", contribution=0.05, score_before=0.62, score_after=0.67),
    ])
    comp = _score_signal_bonus(c, DEFAULT_WEIGHTS)
    assert comp.weight == 0.0
    assert comp.contribution == 0.0
    assert "already in base score" in comp.note


def test_score_signal_bonus_none():
    c = make_candidate(0, 10, 0.5)
    comp = _score_signal_bonus(c, DEFAULT_WEIGHTS)
    assert comp.raw_value == 0.0
    assert comp.weight == 0.0


def test_score_visual_novelty():
    c = make_candidate(0, 10, 0.5)
    ev = [
        make_via_evidence("asset1", 0, 10, quality="high", novelty="high"),
        make_via_evidence("asset1", 0, 10, quality="medium"),
    ]
    comp = _score_visual_novelty(c, ev, DEFAULT_WEIGHTS)
    assert comp.raw_value > 0.0
    assert comp.contribution == comp.raw_value * DEFAULT_WEIGHTS.visual_novelty


def test_score_continuity_neutral_weight():
    """Continuity has weight 0.0 - contribution always 0."""
    c = make_candidate(0, 10, 0.5)
    ev = [
        make_via_evidence("asset1", 0, 5, continuity_id="group1"),
        make_via_evidence("asset1", 5, 10, continuity_id="group1"),
        make_via_evidence("asset1", 10, 15, continuity_id="group2"),
    ]
    comp = _score_continuity(c, ev, DEFAULT_WEIGHTS)
    assert comp.weight == 0.0
    assert comp.contribution == 0.0
    assert "neutral weight" in comp.note


def test_score_continuity_none():
    c = make_candidate(0, 10, 0.5)
    ev = [make_via_evidence("asset1", 0, 10)]  # no continuity_id
    comp = _score_continuity(c, ev, DEFAULT_WEIGHTS)
    assert comp.weight == 0.0
    assert comp.contribution == 0.0


def test_score_duration_quality_within_preset():
    c = make_candidate(0, 60, 0.5)  # standard preset 45-90
    comp = _score_duration_quality(c, DEFAULT_WEIGHTS)
    assert comp.raw_value == 1.0
    assert comp.contribution == DEFAULT_WEIGHTS.duration_quality


def test_score_duration_quality_outside_preset():
    c = make_candidate(0, 10, 0.5)  # too short
    comp = _score_duration_quality(c, DEFAULT_WEIGHTS)
    assert comp.raw_value < 1.0
    assert comp.raw_value > 0.0

    c2 = make_candidate(0, 200, 0.5)  # too long
    comp2 = _score_duration_quality(c2, DEFAULT_WEIGHTS)
    assert comp2.raw_value < 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline tests
# ──────────────────────────────────────────────────────────────────────────────

def test_single_candidate():
    """One candidate only → rank 1."""
    c = make_candidate(10, 20, 0.7)
    scored = score_candidates([c])
    assert len(scored) == 1
    assert scored[0].rank == 1
    assert scored[0].overall_score > 0.0


def test_multiple_candidates_clear_differences():
    """Multiple candidates with clear score differences → correct ranking."""
    c1 = make_candidate(0, 10, 0.9)
    c2 = make_candidate(20, 30, 0.5)
    c3 = make_candidate(40, 50, 0.2)

    scored = score_candidates([c1, c2, c3])
    assert scored[0].rank == 1
    assert scored[1].rank == 2
    assert scored[2].rank == 3
    assert scored[0].overall_score > scored[1].overall_score > scored[2].overall_score


def test_equal_overall_scores_stable_tiebreak():
    """Equal overall scores → deterministic tie-break by start ASC, then stable_id."""
    c1 = make_candidate(10, 20, 0.5, title="A")
    c2 = make_candidate(5, 15, 0.5, title="B")   # earlier start
    c3 = make_candidate(15, 25, 0.5, title="C")  # later start

    scored = score_candidates([c1, c2, c3])
    assert scored[0].candidate.start == 5.0
    assert scored[1].candidate.start == 10.0
    assert scored[2].candidate.start == 15.0


def test_missing_via_evidence():
    """Missing VIA evidence → no crash, components score 0."""
    c = make_candidate(0, 10, 0.6)
    scored = score_candidates([c], via_evidence=None)
    assert len(scored) == 1
    assert scored[0].overall_score > 0.0

    comps = {c.name: c for c in scored[0].component_scores}
    assert comps["event_strength"].raw_value == 0.0
    assert comps["visual_novelty"].raw_value == 0.0
    assert comps["continuity"].weight == 0.0


def test_missing_transcript():
    """Missing transcript → transcript_relevance scores 0, no crash."""
    c = make_candidate(0, 10, 0.6)
    scored = score_candidates([c], transcript_segments=None)
    comps = {c.name: c for c in scored[0].component_scores}
    assert comps["transcript_relevance"].raw_value == 0.0


def test_empty_candidate_collection():
    """Empty input → empty output."""
    scored = score_candidates([])
    assert scored == []


def test_deterministic_rerun():
    """Same input → identical output (scores, ranks, breakdown)."""
    c1 = make_candidate(0, 10, 0.8)
    c2 = make_candidate(20, 30, 0.6)
    via = [make_via_evidence("a1", 0, 10, payoff=True), make_via_evidence("a1", 20, 30)]
    segments = [{"start": 0, "end": 10, "text": "victoria"}, {"start": 20, "end": 30, "text": "gameplay"}]

    scored1 = score_candidates([c1, c2], via, segments)
    scored2 = score_candidates([c1, c2], via, segments)

    assert len(scored1) == len(scored2) == 2
    for s1, s2 in zip(scored1, scored2):
        assert s1.overall_score == s2.overall_score
        assert s1.rank == s2.rank
        assert len(s1.component_scores) == len(s2.component_scores)
        for comp1, comp2 in zip(s1.component_scores, s2.component_scores):
            assert comp1.raw_value == comp2.raw_value
            assert comp1.contribution == comp2.contribution


def test_input_order_independence():
    """Input ordering does not affect ranking."""
    c1 = make_candidate(0, 10, 0.9)
    c2 = make_candidate(20, 30, 0.5)

    scored1 = score_candidates([c1, c2])
    scored2 = score_candidates([c2, c1])

    ranks1 = [(s.candidate.start, s.rank) for s in scored1]
    ranks2 = [(s.candidate.start, s.rank) for s in scored2]
    assert ranks1 == ranks2


def test_permutation_invariance_overlapping_candidates():
    """
    P0-1: Identical candidate sets with overlapping windows produce identical
    scores and ranks regardless of input permutation.
    """
    c1 = make_candidate(0, 10, 0.8)      # high base
    c2 = make_candidate(2, 12, 0.75)     # overlaps c1 at 80%
    c3 = make_candidate(20, 30, 0.6)     # no overlap

    # Test all permutations
    reference = score_candidates([c1, c2, c3])
    ref_scores = {s.candidate.start: s.overall_score for s in reference}
    ref_ranks = {s.candidate.start: s.rank for s in reference}

    for perm in itertools.permutations([c1, c2, c3]):
        scored = score_candidates(list(perm))
        scores = {s.candidate.start: s.overall_score for s in scored}
        ranks = {s.candidate.start: s.rank for s in scored}
        assert scores == ref_scores
        assert ranks == ref_ranks


def test_permutation_invariance_equal_base_overlapping():
    """
    P0-1: Equal base scores with overlapping windows - permutation invariance.
    Both overlapping candidates should get symmetric redundancy penalty.
    """
    c1 = make_candidate(0, 10, 0.7)      # base 0.7
    c2 = make_candidate(3, 13, 0.7)      # base 0.7, 70% overlap with c1
    c3 = make_candidate(20, 30, 0.7)     # base 0.7, no overlap

    reference = score_candidates([c1, c2, c3])
    ref_scores = {s.candidate.start: s.overall_score for s in reference}
    ref_ranks = {s.candidate.start: s.rank for s in reference}

    for perm in itertools.permutations([c1, c2, c3]):
        scored = score_candidates(list(perm))
        scores = {s.candidate.start: s.overall_score for s in scored}
        ranks = {s.candidate.start: s.rank for s in scored}
        assert scores == ref_scores, f"Perm {perm}: scores differ"
        assert ranks == ref_ranks, f"Perm {perm}: ranks differ"


def test_permutation_invariance_differing_evidence():
    """
    P0-1: Equal base scores, DIFFERENT evidence strength (non-overlapping windows).
    The one with stronger evidence should win regardless of input order.
    """
    c1 = make_candidate(0, 10, 0.7)
    c2 = make_candidate(20, 30, 0.7)     # no overlap with c1
    c3 = make_candidate(40, 50, 0.7)     # no overlap

    # c1 has strong VIA evidence, c2 has weak
    via1 = [make_via_evidence("a1", 0, 10, confidence=0.95, payoff=True, quality="high")]
    via2 = [make_via_evidence("a1", 20, 30, confidence=0.3, quality="low")]

    # Test both input orders
    scored1 = score_candidates([c1, c2, c3], via1 + via2)
    scored2 = score_candidates([c2, c1, c3], via1 + via2)

    # c1 should win (stronger evidence) regardless of input order
    for scored in [scored1, scored2]:
        winner = next(s for s in scored if s.candidate.start == 0)
        loser = next(s for s in scored if s.candidate.start == 20)
        assert winner.overall_score > loser.overall_score
        assert winner.rank < loser.rank


def test_breakdown_sums_correctly():
    """Component contributions sum to overall score (within clamping)."""
    from app.pipeline.highlight_scoring import _clamp01
    c = make_candidate(0, 10, 0.7)
    via = [make_via_evidence("a1", 0, 10, confidence=0.9, payoff=True, quality="high")]
    segments = [{"start": 0, "end": 10, "text": "victoria clutch"}]

    scored = score_candidates([c], via, segments)
    sc = scored[0]

    total_contrib = sum(comp.contribution for comp in sc.component_scores)
    # Add redundancy contribution
    redundancy_contrib = 0.0  # single candidate
    assert sc.overall_score == pytest.approx(_clamp01(total_contrib + redundancy_contrib), abs=1e-4)


def test_low_confidence_candidate():
    """Low-confidence candidate behaves as designed (lower overall)."""
    c_high = make_candidate(0, 10, 0.9)
    c_low = make_candidate(20, 30, 0.3)

    via_high = [make_via_evidence("a1", 0, 10, confidence=0.95, payoff=True, quality="high")]
    via_low = [make_via_evidence("a1", 20, 30, confidence=0.2, quality="low")]

    scored = score_candidates([c_high, c_low], via_high + via_low)

    assert scored[0].candidate.start == 0.0
    assert scored[0].overall_score > scored[1].overall_score


def test_strong_evidence_beats_weak():
    """Strong evidence beats materially weaker evidence."""
    c_strong = make_candidate(0, 10, 0.7)
    c_weak = make_candidate(20, 30, 0.7)  # same base

    via_strong = [make_via_evidence("a1", 0, 10, confidence=0.95, payoff=True, reaction=True, quality="high", novelty="high", continuity_id="grp1")]
    via_weak = [make_via_evidence("a1", 20, 30, confidence=0.3)]

    scored = score_candidates([c_strong, c_weak], via_strong + via_weak)

    assert scored[0].candidate.start == 0.0
    assert scored[0].overall_score > scored[1].overall_score


def test_symmetric_redundancy_decreases_score():
    """Symmetric redundancy penalty decreases score for BOTH overlapping candidates."""
    c1 = make_candidate(0, 10, 0.8)
    c2 = make_candidate(1, 11, 0.75)  # high overlap with c1

    scored = score_candidates([c1, c2])

    # Both should have redundancy_penalty component in breakdown
    for sc in scored:
        redundancy_comp = next((c for c in sc.component_scores if c.name == "redundancy_penalty"), None)
        assert redundancy_comp is not None, f"Missing redundancy_penalty for candidate at {sc.candidate.start}"
        assert redundancy_comp.contribution < 0

    # The one with lower base should be more affected relatively
    sc1 = next(s for s in scored if s.candidate.start == 0)
    sc2 = next(s for s in scored if s.candidate.start == 1)
    assert sc1.overall_score > sc2.overall_score


def test_enrich_preserves_original_fields():
    """enrich_candidates_with_scores preserves all original fields."""
    c = make_candidate(0, 10, 0.7, title="Test", hook="Hook", reason="Reason")
    scored = score_candidates([c])
    enriched = enrich_candidates_with_scores(scored)

    assert len(enriched) == 1
    e = enriched[0]
    assert e.title == "Test"
    assert e.hook == "Hook"
    assert e.reason == "Reason"
    assert e.start == 0.0
    assert e.end == 10.0
    assert e.score == pytest.approx(scored[0].overall_score, abs=1e-4)
    assert len(e.score_breakdown) > len(c.score_breakdown)


def test_enrich_adds_highlight_breakdown_entries():
    """Enrichment adds ScoreContribution entries for each component (including 0-weight informational)."""
    c = make_candidate(0, 10, 0.5)
    scored = score_candidates([c])
    enriched = enrich_candidates_with_scores(scored)

    e = enriched[0]
    highlight_entries = [b for b in e.score_breakdown if b.signal.startswith("highlight_")]
    # 7 components (signal_bonus and continuity have weight 0 but are included)
    assert len(highlight_entries) == 7


def test_custom_weights():
    """Custom ScoringWeights are respected (positive weights must sum to 1.0)."""
    custom = ScoringWeights(
        base=0.6, event_strength=0.2, transcript_relevance=0.1,
        visual_novelty=0.05, duration_quality=0.05,
        continuity=0.0, signal_bonus=0.0, redundancy_penalty=-0.1
    )
    c = make_candidate(0, 60, 0.4)  # 60s = within standard preset
    scored = score_candidates([c], weights=custom)
    # 0.4 * 0.6 + 0 * 0.2 + 0 * 0.1 + 0 * 0.05 + 1.0 * 0.05 = 0.24 + 0.05 = 0.29
    assert scored[0].overall_score == pytest.approx(0.29, abs=1e-4)


def test_weights_validation():
    """Invalid weights (positive sum != 1.0) raise ValueError."""
    with pytest.raises(ValueError):
        ScoringWeights(base=1.0, event_strength=0.5)  # sum = 1.5


def test_no_asset_id_in_scored_candidate():
    """ScoredCandidate does not have asset_id field (single-asset context)."""
    c = make_candidate(0, 10, 0.7)
    scored = score_candidates([c])
    assert not hasattr(scored[0], "asset_id")


def test_signal_bonus_not_double_counted():
    """
    P1-1: Signal bonus from signals.py is already in candidate.score.
    It appears in breakdown for observability but does not contribute to overall.
    """
    c = make_candidate(0, 10, 0.62, breakdown=[
        ScoreContribution(signal="señales_audio_escena", contribution=0.12, score_before=0.5, score_after=0.62),
    ])
    scored = score_candidates([c])
    sc = scored[0]

    # Find signal_bonus component
    signal_comp = next(comp for comp in sc.component_scores if comp.name == "signal_bonus")
    assert signal_comp.weight == 0.0
    assert signal_comp.contribution == 0.0
    assert "already in base score" in signal_comp.note

    # Overall score should be based on base (0.62) + other components, NOT 0.62 + 0.12 again
    assert sc.overall_score <= 1.0


def test_no_saturation_at_1_0():
    """
    P1-2: Strong candidates with overlap do not saturate at 1.0.

    With normalized weights (sum=1.0) and max raw_values=1.0:
    max weighted_sum = 1.0
    max penalty = 0.1 (redundancy_penalty=-0.1, raw=1.0)
    max overall = 0.9 (before clamp)
    """
    # Two overlapping perfect candidates -> both get redundancy penalty
    c1 = make_candidate(0, 60, 1.0)
    c2 = make_candidate(10, 70, 1.0)  # overlaps c1
    via = [
        make_via_evidence("a1", 0, 60, confidence=1.0, payoff=True, reaction=True, quality="high", novelty="high"),
        make_via_evidence("a1", 10, 70, confidence=1.0, payoff=True, reaction=True, quality="high", novelty="high"),
    ]
    segments = [{"start": 0, "end": 70, "text": "victoria clutch victoria " * 10}]

    scored = score_candidates([c1, c2], via, segments)

    # Both should have redundancy penalty and score < 1.0
    for sc in scored:
        assert sc.overall_score < 1.0, f"Candidate at {sc.candidate.start} saturated: {sc.overall_score}"
        assert sc.overall_score > 0.8  # but still high

    # Rank 1 and 2 should be distinguishable
    assert abs(scored[0].overall_score - scored[1].overall_score) < 1e-9  # symmetric penalty


def test_single_candidate_no_saturation():
    """
    P1-2: Single candidate without overlap doesn't artificially saturate.

    The score is determined by actual component values, not forced to 1.0.
    No redundancy penalty is applied (correct).
    """
    c = make_candidate(0, 60, 1.0)
    via = [make_via_evidence("a1", 0, 60, confidence=1.0, payoff=True, reaction=True, quality="high", novelty="high")]
    segments = [{"start": 0, "end": 60, "text": "victoria clutch victoria " * 10}]

    scored = score_candidates([c], via, segments)
    # Score reflects actual component values (no redundancy penalty)
    assert 0.0 <= scored[0].overall_score <= 1.0
    # Verify no redundancy penalty component added
    redundancy_comp = next((comp for comp in scored[0].component_scores if comp.name == "redundancy_penalty"), None)
    assert redundancy_comp is None or redundancy_comp.raw_value == 0.0


def test_score_discrimination_typical_range():
    """
    P1-2: Candidates with typical base scores (0.6–0.9) produce
    distinguishable, monotonic overall scores.
    """
    candidates = [
        make_candidate(i * 10, i * 10 + 10, base)
        for i, base in enumerate([0.9, 0.8, 0.7, 0.6])
    ]
    scored = score_candidates(candidates)

    scores = [s.overall_score for s in scored]
    # Should be strictly decreasing
    for i in range(len(scores) - 1):
        assert scores[i] > scores[i + 1], f"Score {i} ({scores[i]}) not > {i+1} ({scores[i+1]})"

    # Differences should be meaningful (> 0.01)
    for i in range(len(scores) - 1):
        assert scores[i] - scores[i + 1] > 0.01


def test_zero_duration_candidate():
    """Zero duration handled safely."""
    c = make_candidate(10, 10, 0.5)
    scored = score_candidates([c])
    assert scored[0].overall_score >= 0.0


def test_score_clamped_to_bounds():
    """Overall score never exceeds 1.0 or goes below 0.0."""
    c = make_candidate(0, 10, 1.0)
    via = [make_via_evidence("a1", 0, 10, confidence=1.0, payoff=True, reaction=True, quality="high", novelty="high", continuity_id="grp1")]
    segments = [{"start": 0, "end": 10, "text": "victoria clutch victoria"}]

    scored = score_candidates([c], via, segments)
    assert 0.0 <= scored[0].overall_score <= 1.0


def test_nan_infinity_guard():
    """No NaN/Infinity in output."""
    c = make_candidate(0, 10, 0.5)
    scored = score_candidates([c])
    assert scored[0].overall_score == scored[0].overall_score  # not NaN
    assert scored[0].overall_score != float('inf')
    assert scored[0].overall_score != float('-inf')


# ──────────────────────────────────────────────────────────────────────────────
# Integration: score_and_enrich
# ──────────────────────────────────────────────────────────────────────────────

def test_score_and_enrich_returns_both():
    c = make_candidate(0, 10, 0.6)
    scored, enriched = score_and_enrich([c])

    assert len(scored) == 1
    assert len(enriched) == 1
    assert scored[0].overall_score == pytest.approx(enriched[0].score, abs=1e-4)
    assert scored[0].rank == 1


# ──────────────────────────────────────────────────────────────────────────────
# Negative tests: weights validation
# ──────────────────────────────────────────────────────────────────────────────

def test_negative_base_weight_allowed():
    """Negative base weight is allowed (though not recommended)."""
    custom = ScoringWeights(
        base=-0.5, event_strength=1.0, transcript_relevance=0.5,
        visual_novelty=0.0, duration_quality=0.0,
        continuity=0.0, signal_bonus=0.0, redundancy_penalty=0.0
    )
    c = make_candidate(0, 10, 0.5)
    scored = score_candidates([c], weights=custom)
    # Should work without error
    assert isinstance(scored[0].overall_score, float)