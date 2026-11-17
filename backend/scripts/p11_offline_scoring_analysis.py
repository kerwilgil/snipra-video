"""P1.1-C offline analysis from persisted Terra evidence; never calls a VLM."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.structured_visual import StructuredEventFeatures, deterministic_visual_score

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "artifacts/qa/editorial-experiments/terra_experiment.json"
GROUND_TRUTH = ROOT / "artifacts/qa/editorial-experiments/human_ground_truth_template.json"
OUTPUT = ROOT / "artifacts/qa/editorial-experiments/offline_scoring_analysis.json"
FEATURES = (
    "event_present",
    "event_type",
    "event_strength",
    "payoff_present",
    "outcome",
    "reaction_present",
    "visual_novelty",
    "menu_or_idle",
    "peak_frame",
    "evidence_quality",
)


def score_v2_candidate(features: StructuredEventFeatures) -> float:
    """Conservative candidate: collapse unstable strength/outcome/novelty effects."""
    if features.menu_or_idle:
        return 0.0
    if features.evidence_quality == "insufficient":
        return 0.05
    score = 0.15 if features.event_present else 0.0
    score += 0.15 if features.event_strength in {"medium", "high"} else 0.05
    if features.payoff_present and features.evidence_quality == "clear":
        score += 0.08
    return round(min(score, 1.0), 3)


def score_v3_candidate(features: StructuredEventFeatures) -> float:
    """Guardrail-oriented candidate: score only verified presence, not outcome semantics."""
    if features.menu_or_idle:
        return 0.0
    if not features.event_present or features.evidence_quality == "insufficient":
        return 0.05
    score = 0.20
    if features.event_strength in {"medium", "high"}:
        score += 0.10
    if features.payoff_present and features.evidence_quality == "clear":
        score += 0.05
    return round(score, 3)


SCORERS = {
    "gaming_v1": deterministic_visual_score,
    "gaming_v2_candidate": score_v2_candidate,
    "gaming_v3_candidate": score_v3_candidate,
}


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 3),
        "std": round(statistics.stdev(values), 3),
        "range": round(max(values) - min(values), 3),
    }


def _mode(values: list[object]) -> object:
    return Counter(map(str, values)).most_common(1)[0][0]


def _features(rows: list[dict]) -> list[StructuredEventFeatures]:
    return [StructuredEventFeatures.model_validate(row["features"]) for row in rows]


def profile_results(evidence: dict) -> dict:
    result: dict[str, dict] = {}
    candidates = evidence["candidates"]
    for profile, scorer in SCORERS.items():
        by_candidate = {}
        for key in candidates:
            values = [scorer(features) for features in _features(evidence["structured"][key])]
            by_candidate[key] = _stats(values)
        ranking = sorted(by_candidate, key=lambda key: (-by_candidate[key]["mean"], key))
        result[profile] = {
            "by_candidate": by_candidate,
            "mean_candidate_std": round(
                statistics.mean(row["std"] for row in by_candidate.values()), 3
            ),
            "mean_candidate_range": round(
                statistics.mean(row["range"] for row in by_candidate.values()), 3
            ),
            "ranking_by_mean": ranking,
        }
    return result


def feature_stability_and_sensitivity(evidence: dict) -> list[dict]:
    rows_by_candidate = {
        key: _features(evidence["structured"][key]) for key in evidence["candidates"]
    }
    output = []
    for field in FEATURES:
        agreements = []
        deltas = []
        for features in rows_by_candidate.values():
            values = [getattr(item, field) for item in features]
            mode = _mode(values)
            mode_value = next(item for item in values if str(item) == mode)
            agreements.append(values.count(mode_value) / len(values))
            for item in features:
                replacement = item.model_copy(update={field: mode_value})
                delta = abs(
                    deterministic_visual_score(item) - deterministic_visual_score(replacement)
                )
                deltas.append(delta)
        impact = round(statistics.mean(deltas), 3)
        agreement = statistics.mean(agreements)
        risk = "high" if impact >= 0.05 and agreement < 0.9 else "medium" if impact else "low"
        output.append(
            {
                "feature": field,
                "agreement_pct": round(statistics.mean(agreements) * 100, 1),
                "mean_observed_v1_score_delta": impact,
                "risk": risk,
            }
        )
    return sorted(
        output,
        key=lambda row: (-row["agreement_pct"], -row["mean_observed_v1_score_delta"]),
    )


def guardrail_design(evidence: dict) -> dict:
    candidates = {}
    for key in evidence["candidates"]:
        features = _features(evidence["structured"][key])
        modes = {field: _mode([getattr(item, field) for item in features]) for field in FEATURES}
        candidates[key] = {
            "menu_or_idle": modes["menu_or_idle"] == "True",
            "evidence_quality": modes["evidence_quality"],
            "outcome": modes["outcome"],
            "payoff_present": modes["payoff_present"] == "True",
            "recommended_guardrail": (
                "hard_reject_menu_or_idle"
                if modes["menu_or_idle"] == "True"
                else "withhold_outcome_bonus" if modes["outcome"] == "uncertain" else "observe_only"
            ),
        }
    return {
        "policy": {
            "menu_or_idle_true": "hard reject or require legacy confirmation before selection",
            "evidence_quality_insufficient": "block strong visual claim; do not promote",
            "outcome_uncertain": "withhold kill/victory/death bonuses; preserve uncertain",
            "payoff_present_and_clear": "allow only as confirmation, not standalone ranking",
        },
        "candidate_modes": candidates,
    }


def _ranking_metrics(ranking: list[str], ground_truth: dict) -> dict:
    items = ground_truth["candidates"]
    ranks = {key: row["preferred_rank"] for key, row in items.items()}
    relevant = {key for key, row in items.items() if row["relevant"] is True}
    result = {"status": "pending_human_validation"}
    if all(value is not None for value in ranks.values()):
        expected = sorted(ranks, key=ranks.get)
        result["top_1_correct"] = ranking[0] == expected[0]
        result["spearman"] = round(
            1 - 6 * sum((ranking.index(key) + 1 - ranks[key]) ** 2 for key in ranking) / 120,
            3,
        )
    if relevant:
        result["top_3_recall"] = round(len(set(ranking[:3]) & relevant) / len(relevant), 3)
    return result


def human_quality_plan(evidence: dict, ground_truth: dict, profiles: dict) -> dict:
    result = {
        "status": "pending_human_validation",
        "required_fields": [
            "relevant", "hook_quality", "highlight_quality", "payoff_quality", "correct_outcome",
            "preferred_rank", "desired_start", "desired_peak", "desired_end",
        ],
        "rankings": {},
        "missing": {},
    }
    for name, profile in profiles.items():
        result["rankings"][name] = _ranking_metrics(profile["ranking_by_mean"], ground_truth)
    result["rankings"]["relative_terra"] = _ranking_metrics(
        evidence["final_analysis"]["relative_ranking"]["mode_ordering"], ground_truth
    )
    result["rankings"]["legacy_absolute"] = _ranking_metrics(
        evidence["final_analysis"]["relative_ranking"]["absolute_score_mode_ordering"], ground_truth
    )
    for key, row in ground_truth["candidates"].items():
        result["missing"][key] = [
            field for field in result["required_fields"] if row.get(field) in (None, "unknown")
        ]
    return result


def legacy_weight_simulation(evidence: dict, ground_truth: dict) -> dict:
    pre_visual = {
        key: row.get("pre_visual_score") for key, row in ground_truth["candidates"].items()
    }
    if any(value is None for value in pre_visual.values()):
        return {
            "status": "blocked_missing_pre_visual_score",
            "required_input": "pre_visual_score for all five candidates",
            "weights": [0.45, 0.30, 0.20, 0.10, 0.0],
        }
    legacy = {
        key: evidence["statistics"]["legacy"][key]["score"]["mean"]
        for key in evidence["candidates"]
    }
    return {
        "status": "ready",
        "rankings": {
            str(weight): sorted(
                legacy,
                key=lambda key: (-(1 - weight) * pre_visual[key] - weight * legacy[key], key),
            )
            for weight in (0.45, 0.30, 0.20, 0.10, 0.0)
        },
    }


def analyze(evidence: dict, ground_truth: dict) -> dict:
    profiles = profile_results(evidence)
    return {
        "source": str(EVIDENCE),
        "no_vlm_calls": True,
        "profiles": profiles,
        "feature_stability": feature_stability_and_sensitivity(evidence),
        "structured_guardrail": guardrail_design(evidence),
        "human_quality": human_quality_plan(evidence, ground_truth, profiles),
        "legacy_weight_simulation": legacy_weight_simulation(evidence, ground_truth),
    }


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    payload = json.dumps(analyze(evidence, ground_truth), ensure_ascii=False, indent=2)
    OUTPUT.write_text(payload, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
