"""P1.1-C offline scorer tests; no provider is involved."""

from __future__ import annotations

from app.providers.structured_visual import StructuredEventFeatures
from scripts import p11_offline_scoring_analysis as analysis


def _features(**updates: object) -> StructuredEventFeatures:
    values = {
        "event_present": True,
        "event_type": "combat",
        "event_strength": "high",
        "payoff_present": False,
        "outcome": "uncertain",
        "reaction_present": False,
        "visual_novelty": "high",
        "menu_or_idle": False,
        "peak_frame": 3,
        "evidence_frames": [3],
        "evidence_quality": "clear",
        "observed_evidence": "impacto visible",
    }
    values.update(updates)
    return StructuredEventFeatures.model_validate(values)


def test_candidate_profiles_are_deterministic_and_menu_is_a_guardrail():
    sample = _features()
    assert analysis.score_v2_candidate(sample) == analysis.score_v2_candidate(sample)
    assert analysis.score_v3_candidate(sample) == analysis.score_v3_candidate(sample)
    menu = _features(menu_or_idle=True, outcome="menu")
    assert analysis.score_v2_candidate(menu) == 0
    assert analysis.score_v3_candidate(menu) == 0


def test_weight_simulation_waits_for_pre_visual_inputs():
    result = analysis.legacy_weight_simulation(
        {"candidates": {"one": {}}, "statistics": {"legacy": {"one": {"score": {"mean": 0.5}}}}},
        {"candidates": {"one": {"pre_visual_score": None}}},
    )
    assert result["status"] == "blocked_missing_pre_visual_score"
