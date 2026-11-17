import pytest

from app.providers.structured_visual import (
    StructuredEventFeatures,
    StructuredVisualCache,
    cache_key,
    deterministic_visual_score,
    parse_structured_response,
)


def features(**overrides):
    data = dict(
        event_present=True,
        event_type="combat",
        event_strength="high",
        payoff_present=False,
        outcome="uncertain",
        reaction_present=False,
        visual_novelty="medium",
        menu_or_idle=False,
        peak_frame=3,
        evidence_frames=[3],
        evidence_quality="partial",
        observed_evidence="golpe visible",
    )
    data.update(overrides)
    return StructuredEventFeatures(**data)


def test_same_json_has_same_deterministic_score():
    assert deterministic_visual_score(features()) == deterministic_visual_score(features())


def test_uncertain_has_no_kill_bonus():
    assert deterministic_visual_score(features(outcome="uncertain")) < deterministic_visual_score(
        features(outcome="kill")
    )


def test_menu_and_insufficient_are_conservative():
    assert deterministic_visual_score(features(menu_or_idle=True)) == 0
    assert deterministic_visual_score(
        features(evidence_quality="insufficient")
    ) < deterministic_visual_score(features())


def test_parser_accepts_uncertain_and_rejects_invalid_schema():
    assert parse_structured_response(features().model_dump_json()).outcome == "uncertain"
    with pytest.raises(ValueError):
        parse_structured_response('{"outcome":"maybe"}')


def test_cache_key_versions_and_provider_invalidate(tmp_path):
    key = cache_key("f", 1, 2, [1.2], "codex", "gpt-5.6-terra")
    assert key != cache_key("f", 1, 2, [1.2], "claude", "gpt-5.6-terra")
    cache = StructuredVisualCache(tmp_path / "cache.db")
    assert cache.get(key) is None
    cache.put(key, {"legacy_vlm_score": 0.7})
    assert cache.get(key)["legacy_vlm_score"] == 0.7


def test_scoring_profile_change_reuses_cached_features(tmp_path):
    key = cache_key("video", 1, 2, [1.2], "codex", "gpt-5.6-terra")
    cache = StructuredVisualCache(tmp_path / "cache.db")
    cached = features().model_dump()
    cache.put(key, {"features": cached})
    assert cache.get(key)["features"] == cached
    first = deterministic_visual_score(StructuredEventFeatures(**cached))
    second = deterministic_visual_score(StructuredEventFeatures(**cache.get(key)["features"]))
    assert first == second
