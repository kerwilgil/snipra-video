"""Recovery guarantees for the P1.1-B experimental Terra harness."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import p11_terra_experiment as experiment


def _legacy_record(run_id: int) -> dict:
    return {
        "run_id": run_id,
        "raw_response": '{"score": 0.5}',
        "legacy_vlm_score": 0.5,
        "outcome": "neutral",
        "peak_frame": "2",
        "request_duration_s": 1.0,
    }


def test_resume_ignores_incomplete_rows_and_never_repeats_valid_runs(tmp_path: Path):
    output = tmp_path / "evidence.json"
    result = experiment.prepare_result(
        {"legacy": {"fuerte": [_legacy_record(1), {"run_id": 2, "raw_response": "partial"}]}}
    )
    experiment.atomic_write(output, result)
    calls: list[int] = []

    def fake_runner(kind: str, title: str, frames: list[bytes], run_id: int) -> dict:
        calls.append(run_id)
        return _legacy_record(run_id)

    completed = experiment.resume_candidate(
        result, "legacy", "fuerte", "title", [], output, fake_runner
    )

    loaded = experiment.load_results(output)
    assert completed == 4
    assert calls == [2, 3, 4, 5]
    assert [row["run_id"] for row in loaded["legacy"]["fuerte"]] == [1, 2, 3, 4, 5]
    assert loaded["interrupted"]["legacy"]["fuerte"][0]["status"] == "invalid"


def test_atomic_checkpoint_survives_interrupted_temporary_write(tmp_path: Path):
    output = tmp_path / "evidence.json"
    expected = experiment.prepare_result({"legacy": {"fuerte": [_legacy_record(1)]}})
    experiment.atomic_write(output, expected)
    output.with_suffix(".json.tmp").write_text("{interrupted", encoding="utf-8")

    restored = experiment.load_results(output)

    record = restored["legacy"]["fuerte"][0]
    assert record["run_id"] == 1
    assert record["candidate"] == "fuerte"
    assert record["provider"] == experiment.PROVIDER
    assert json.loads(output.read_text(encoding="utf-8"))["legacy"]["fuerte"][0]["run_id"] == 1


def test_cache_probe_reuses_inference_when_only_profile_changes(tmp_path: Path):
    result = experiment.prepare_result(
        {
            "structured": {
                "medio": [
                    {
                        "run_id": 1,
                        "raw_response": "{}",
                        "features": {
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
                        },
                        "deterministic_visual_score": 0.49,
                        "request_duration_s": 1.0,
                    }
                ]
            }
        }
    )
    probe = experiment.verify_cache_functionally(result, tmp_path / "cache.db")

    assert probe["case_a"]["provider_calls"] == 1
    assert probe["case_b"]["additional_provider_calls"] == 0
    assert probe["case_c"]["additional_provider_calls"] == 0
    assert probe["case_b"]["features_identical"]
    assert probe["case_c"]["features_identical"]
    assert probe["case_c"]["score_recalculated"]
