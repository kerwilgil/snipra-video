"""P1.1-B Terra-only experiment; resumable and intentionally cache-bypassed."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from shutil import which

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.cli_subscription import resolve_provider_cli
from app.providers.structured_visual import (
    GAMING_V1,
    StructuredEventFeatures,
    StructuredVisualCache,
    deterministic_visual_score,
    inference_cache_key,
    parse_structured_response,
    structured_prompt,
)
from app.providers.vision import _extract_frames, _parse_visual_response, _prompt_for

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "storage/editorial-experiment/source.mp4"
OUT = ROOT / "artifacts/qa/editorial-experiments/terra_experiment.json"
MODEL = "gpt-5.6-terra"
PROVIDER = "codex"
RUNS_PER_CANDIDATE = 5
CANDIDATES = {
    "fuerte": ("Jugada decisiva (evento fuerte)", 862.0, 886.0),
    "payoff": ("Jugada decisiva (payoff highlights)", 1042.0, 1066.0),
    "medio": ("Que tal (kill/death/victory histórico Claude)", 763.1, 789.5),
    "malo": ("Vamos a jugar (menu)", 608.2, 617.6),
    "ambiguo": ("Just a cop (ambiguo)", 294.9, 363.2),
}


def atomic_write(path: Path, payload: dict) -> None:
    """Replace the evidence file only after a complete JSON write succeeds."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_results(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _has_required(record: dict, keys: tuple[str, ...]) -> bool:
    return all(record.get(key) is not None and record.get(key) != "" for key in keys)


def valid_record(kind: str, record: object) -> bool:
    if not isinstance(record, dict):
        return False
    common = ("run_id", "raw_response", "request_duration_s")
    if not _has_required(record, common):
        return False
    try:
        if kind == "legacy":
            return _has_required(record, ("legacy_vlm_score", "outcome", "peak_frame"))
        if kind == "structured":
            StructuredEventFeatures.model_validate(record["features"])
            return record.get("deterministic_visual_score") is not None
    except (KeyError, TypeError, ValueError):
        return False
    return False


def normalize_results(result: dict) -> bool:
    """Move interrupted/malformed rows aside without losing their raw evidence."""
    changed = False
    interruptions = result.setdefault("interrupted", {})
    for kind in ("legacy", "structured"):
        bucket = result.setdefault(kind, {})
        interrupted_bucket = interruptions.setdefault(kind, {})
        for key in CANDIDATES:
            rows = bucket.get(key, [])
            valid_rows = []
            invalid_rows = interrupted_bucket.setdefault(key, [])
            seen_ids: set[int] = set()
            for record in rows:
                run_id = record.get("run_id") if isinstance(record, dict) else None
                is_valid = valid_record(kind, record) and run_id not in seen_ids
                if is_valid:
                    seen_ids.add(run_id)
                    valid_rows.append(record)
                    continue
                if isinstance(record, dict):
                    record = {
                        **record,
                        "status": "invalid",
                        "invalid_reason": "interrupted_or_incomplete",
                    }
                else:
                    record = {
                        "status": "invalid",
                        "invalid_reason": "non_object_record",
                        "record": record,
                    }
                invalid_rows.append(record)
                changed = True
            if rows != valid_rows:
                bucket[key] = valid_rows
    return changed


def valid_runs(result: dict, kind: str, key: str) -> list[dict]:
    rows = result.setdefault(kind, {}).setdefault(key, [])
    return [record for record in rows if valid_record(kind, record)]


def next_run_id(rows: list[dict]) -> int:
    used = {row["run_id"] for row in rows}
    return next(run_id for run_id in range(1, RUNS_PER_CANDIDATE + 1) if run_id not in used)


def call(prompt: str, frames: list[bytes]) -> tuple[str, float]:
    exe = which("codex.exe") or resolve_provider_cli("codex")
    if not exe:
        raise RuntimeError("Codex CLI unavailable")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        paths = []
        for index, frame in enumerate(frames):
            path = tmp / f"frame-{index}.jpg"
            path.write_bytes(frame)
            paths.append(str(path))
        out = tmp / "answer.txt"
        started = time.monotonic()
        result = subprocess.run(
            [
                exe, "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-rules",
                "--model", MODEL, "-o", str(out), prompt, "-i", *paths,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            cwd=tmp,
            timeout=300,
        )
        elapsed = round(time.monotonic() - started, 3)
        if result.returncode or not out.exists():
            raise RuntimeError((result.stderr or result.stdout)[:500])
        return out.read_text(encoding="utf-8"), elapsed


def stats(values: list[float]) -> dict:
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "std": round(statistics.stdev(values), 3),
        "min": min(values),
        "max": max(values),
        "range": round(max(values) - min(values), 3),
    }


def agreement(values: list[object]) -> dict:
    counts = Counter(map(str, values))
    mode, count = counts.most_common(1)[0]
    return {"mode": mode, "counts": dict(counts), "agreement": round(count / len(values) * 100, 1)}


def calculate_statistics(result: dict) -> dict:
    """Summaries use only records passing the same validation used for resume."""
    summaries: dict[str, dict] = {"legacy": {}, "structured": {}}
    for key in CANDIDATES:
        legacy = valid_runs(result, "legacy", key)
        structured = valid_runs(result, "structured", key)
        if len(legacy) == RUNS_PER_CANDIDATE:
            summaries["legacy"][key] = {
                "score": stats([row["legacy_vlm_score"] for row in legacy]),
                "outcome": agreement([row["outcome"] for row in legacy]),
                "peak_frame": agreement([row["peak_frame"] for row in legacy]),
            }
        if len(structured) == RUNS_PER_CANDIDATE:
            feature_names = StructuredEventFeatures.model_fields.keys()
            summaries["structured"][key] = {
                "score": stats([row["deterministic_visual_score"] for row in structured]),
                "features": {
                    name: agreement([row["features"][name] for row in structured])
                    for name in feature_names
                    if name != "observed_evidence"
                },
            }
    return summaries


def verify_cache_functionally(result: dict, cache_path: Path) -> dict:
    """Exercise the actual SQLite cache without consuming another Terra inference."""
    candidate = result["candidates"]["medio"]
    source = valid_runs(result, "structured", "medio")[0]
    features = StructuredEventFeatures.model_validate(source["features"])
    key = inference_cache_key(
        result["video"],
        candidate["start"],
        candidate["end"],
        candidate["sampled_timestamps"],
        PROVIDER,
        MODEL,
    )
    cache = StructuredVisualCache(cache_path)
    calls = 0

    def provider() -> dict:
        nonlocal calls
        calls += 1
        return {"features": features.model_dump(), "raw_response": source["raw_response"]}

    first = cache.get(key)
    if first is None:
        first = provider()
        cache.put(key, first)
    first_features = StructuredEventFeatures.model_validate(first["features"])
    score_v1 = deterministic_visual_score(first_features)
    calls_after_first = calls

    second = cache.get(key)
    if second is None:
        second = provider()
    second_features = StructuredEventFeatures.model_validate(second["features"])
    score_v1_repeat = deterministic_visual_score(second_features)

    profile_v2 = {name: values.copy() for name, values in GAMING_V1.items()}
    profile_v2["visual_novelty"]["high"] -= 1
    score_v2 = deterministic_visual_score(second_features, profile_v2)
    return {
        "provider": "controlled_cache_probe_no_terra",
        "cache_path": str(cache_path),
        "inference_cache_key": key,
        "case_a": {
            "cache_hit": False,
            "provider_calls": calls_after_first,
            "deterministic_visual_score": score_v1,
            "scoring_profile_version": "gaming_v1",
        },
        "case_b": {
            "cache_hit": True,
            "additional_provider_calls": calls - calls_after_first,
            "features_identical": first_features == second_features,
            "deterministic_score_identical": score_v1 == score_v1_repeat,
            "scoring_profile_version": "gaming_v1",
        },
        "case_c": {
            "cache_hit": True,
            "additional_provider_calls": calls - calls_after_first,
            "features_identical": first_features == second_features,
            "scoring_profile_version": "cache_probe_v2",
            "deterministic_visual_score": score_v2,
            "score_recalculated": score_v1 != score_v2,
        },
    }


def run(kind: str, title: str, frames: list[bytes], run_id: int) -> dict:
    prompt = _prompt_for(title) if kind == "legacy" else structured_prompt(title)
    raw, duration = call(prompt, frames)
    if kind == "legacy":
        score, reason, _ = _parse_visual_response(raw)
        outcome = ""
        peak = ""
        if "resultado visual: " in reason:
            outcome = reason.split("resultado visual: ")[-1].split(";")[0]
        if "pico visual frame " in reason:
            peak = reason.split("pico visual frame ")[-1].split("/")[0]
        return {
            "run_id": run_id,
            "raw_response": raw,
            "legacy_vlm_score": score,
            "outcome": outcome,
            "peak_frame": peak,
            "request_duration_s": duration,
        }
    features = parse_structured_response(raw)
    return {
        "run_id": run_id,
        "raw_response": raw,
        "features": features.model_dump(),
        "deterministic_visual_score": deterministic_visual_score(features),
        "request_duration_s": duration,
    }


Runner = Callable[[str, str, list[bytes], int], dict]


def resume_candidate(
    result: dict,
    kind: str,
    key: str,
    title: str,
    frames: list[bytes],
    output: Path,
    runner: Runner = run,
) -> int:
    """Persist each valid response before attempting the next independent inference."""
    completed = 0
    rows = valid_runs(result, kind, key)
    result[kind][key] = rows
    while len(rows) < RUNS_PER_CANDIDATE:
        run_id = next_run_id(rows)
        try:
            record = runner(kind, title, frames, run_id)
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            result.setdefault("interrupted", {}).setdefault(kind, {}).setdefault(key, []).append(
                {"run_id": run_id, "status": "invalid", "invalid_reason": str(exc)[:500]}
            )
            atomic_write(output, result)
            raise
        if not valid_record(kind, record):
            result.setdefault("interrupted", {}).setdefault(kind, {}).setdefault(key, []).append(
                {**record, "status": "invalid", "invalid_reason": "incomplete_or_invalid_response"}
            )
            atomic_write(output, result)
            raise ValueError(f"{kind} {key} run {run_id} did not produce a valid record")
        rows.append(record)
        result[kind][key] = rows
        atomic_write(output, result)
        completed += 1
        print(
            f"{kind} {key} {len(rows)}/{RUNS_PER_CANDIDATE} "
            f"{record['request_duration_s']}s",
            flush=True,
        )
    return completed


def prepare_result(result: dict) -> dict:
    result.update(
        {
            "label": "TERRA EXPERIMENT — P1.1",
            "provider": PROVIDER,
            "model": MODEL,
            "cache": "BYPASSED",
            "video": str(VIDEO),
        }
    )
    result.setdefault("candidates", {})
    result.setdefault("legacy", {})
    result.setdefault("structured", {})
    normalize_results(result)
    for key, (title, start, end) in CANDIDATES.items():
        candidate = {
            "title": title,
            "start": start,
            "end": end,
            "sampled_timestamps": [round(start + (end - start) * i / 5, 3) for i in range(1, 5)],
        }
        result["candidates"][key] = candidate
        for kind in ("legacy", "structured"):
            for record in valid_runs(result, kind, key):
                record.setdefault("candidate", key)
                record.setdefault("provider", PROVIDER)
                record.setdefault("model", MODEL)
                record.setdefault("sampled_timestamps", candidate["sampled_timestamps"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--kind", choices=("legacy", "structured"))
    args = parser.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = prepare_result(load_results(OUT))
    atomic_write(OUT, result)
    frames = {
        key: _extract_frames(str(VIDEO), start, end)
        for key, (_, start, end) in CANDIDATES.items()
    }
    if args.smoke:
        record = run("legacy", CANDIDATES["fuerte"][0], frames["fuerte"], 1)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return
    kinds = (args.kind,) if args.kind else ("legacy", "structured")
    for kind in kinds:
        for key, (title, _, _) in CANDIDATES.items():
            resume_candidate(result, kind, key, title, frames[key], OUT)


if __name__ == "__main__":
    main()
