"""Resumable non-production follow-up experiments for P1.1-B."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import p11_terra_experiment as base


def _valid_batch_row(row: object) -> bool:
    return isinstance(row, dict) and base.valid_record("structured", row)


def _existing_batches(result: dict, workers: int) -> set[tuple[int, str]]:
    rows = result.get("followups", {}).get("concurrency", {}).get(str(workers), [])
    return {(row["batch"], row["candidate"]) for row in rows if _valid_batch_row(row)}


def _run_one(key: str, title: str, frames: list[bytes], batch: int, workers: int) -> dict:
    record = base.run("structured", title, frames, batch)
    record.update(
        {
            "candidate": key,
            "provider": base.PROVIDER,
            "model": base.MODEL,
            "concurrency": workers,
            "batch": batch,
        }
    )
    return record


def concurrency(result: dict, workers: int, rounds: int, output: Path) -> None:
    """Run only missing independent samples and atomically save each success."""
    result.setdefault("followups", {}).setdefault("concurrency", {}).setdefault(str(workers), [])
    frames = {
        key: base._extract_frames(str(base.VIDEO), start, end)
        for key, (_, start, end) in base.CANDIDATES.items()
    }
    for batch in range(1, rounds + 1):
        existing = _existing_batches(result, workers)
        missing = [
            (key, title, frames[key])
            for key, (title, _, _) in base.CANDIDATES.items()
            if (batch, key) not in existing
        ]
        if not missing:
            continue
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_one, key, title, candidate_frames, batch, workers): key
                for key, title, candidate_frames in missing
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # persisted; follow-up failures remain non-fatal evidence
                    result.setdefault("interrupted", {}).setdefault("concurrency", []).append(
                        {
                            "candidate": key,
                            "batch": batch,
                            "concurrency": workers,
                            "status": "invalid",
                            "invalid_reason": str(exc)[:500],
                        }
                    )
                else:
                    result["followups"]["concurrency"][str(workers)].append(record)
                    print(
                        f"concurrency={workers} batch={batch} {key} "
                        f"{record['request_duration_s']}s",
                        flush=True,
                    )
                base.atomic_write(output, result)


def refined_peak(result: dict, output: Path, repeats: int = 3) -> None:
    target_keys = ("fuerte", "ambiguo")
    rows = result.setdefault("followups", {}).setdefault("refined_peak", [])
    for key in target_keys:
        candidate = result["candidates"][key]
        coarse = int(result["statistics"]["structured"][key]["features"]["peak_frame"]["mode"])
        timestamps = candidate["sampled_timestamps"]
        coarse_time = timestamps[coarse - 1]
        choices = [
            max(candidate["start"], min(candidate["end"], coarse_time + delta))
            for delta in (-2, -1, 0, 1, 2)
        ]
        frames = [
            base._extract_frames(str(base.VIDEO), point - 0.05, point + 0.05)[0]
            for point in choices
        ]
        prompt = (
            "Solo JSON {\"peak_frame\":1..5}. Elige el frame que representa mejor el pico "
            "temporal del evento visual; no clasifiques outcome."
        )
        for run_id in range(1, repeats + 1):
            if any(row["candidate"] == key and row["run_id"] == run_id for row in rows):
                continue
            raw, duration = base.call(prompt, frames)
            choice = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])["peak_frame"]
            refined_time = choices[choice - 1]
            row = {
                "candidate": key,
                "run_id": run_id,
                "coarse_peak_time": coarse_time,
                "timestamps": choices,
                "refined_peak_frame": choice,
                "refined_peak_time": refined_time,
                "delta_s": round(refined_time - coarse_time, 3),
                "raw_response": raw,
                "request_duration_s": duration,
            }
            rows.append(row)
            base.atomic_write(output, result)
            print(f"refined {key} {run_id}/{repeats} {duration}s", flush=True)


def relative_ranking(result: dict, output: Path, repeats: int = 5) -> None:
    rows = result.setdefault("followups", {}).setdefault("relative_ranking", [])
    frames = []
    for _key, (_, start, end) in base.CANDIDATES.items():
        frames.extend(base._extract_frames(str(base.VIDEO), start, end))
    keys = list(base.CANDIDATES)
    prompt = (
        f"Ordena estos cinco candidatos {keys} por relevancia visual. Cada candidato tiene cuatro "
        "frames consecutivos en ese orden. Responde solo JSON {\"ordering\":[cinco claves]}."
    )
    for run_id in range(1, repeats + 1):
        if any(row["run_id"] == run_id for row in rows):
            continue
        raw, duration = base.call(prompt, frames)
        ordering = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])["ordering"]
        if sorted(ordering) != sorted(keys):
            raise ValueError(f"invalid relative ordering: {ordering}")
        rows.append(
            {
                "run_id": run_id,
                "ordering": ordering,
                "raw_response": raw,
                "request_duration_s": duration,
            }
        )
        base.atomic_write(output, result)
        print(f"relative ranking {run_id}/{repeats} {duration}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, choices=(1, 3))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--refined-peak", action="store_true")
    parser.add_argument("--relative-ranking", action="store_true")
    args = parser.parse_args()
    result = base.prepare_result(base.load_results(base.OUT))
    base.atomic_write(base.OUT, result)
    if args.refined_peak:
        refined_peak(result, base.OUT)
    elif args.relative_ranking:
        relative_ranking(result, base.OUT)
    else:
        if args.concurrency is None:
            parser.error("--concurrency is required unless --refined-peak is used")
        concurrency(result, args.concurrency, args.rounds, base.OUT)


if __name__ == "__main__":
    main()
