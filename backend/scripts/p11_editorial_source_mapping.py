"""Offline perceptual-frame mapping from four renders back to the P1.1 source."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "storage/editorial-experiment/source.mp4"
OUTPUT = ROOT / "artifacts/qa/editorial-experiments/editorial_source_mapping.json"
RENDERS = {
    "chatcut_highlight": ROOT / "storage/editorial-experiment/reference-highlight.mp4",
    "chatcut_hook": ROOT / "storage/editorial-experiment/reference-hook.mp4",
    "snipra_highlight": ROOT / "storage/editorial-experiment/snipra-highlight.mp4",
    "snipra_hook": ROOT / "storage/editorial-experiment/snipra-hook.mp4",
}
SAMPLE_HZ = 2.0


def fingerprint(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))[:8, :8]
    return dct > np.median(dct[1:, 1:])


def sampled_fingerprints(path: Path) -> list[tuple[float, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, round(fps / SAMPLE_HZ))
    frame_index = 0
    samples = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % every == 0:
            samples.append((round(frame_index / fps, 3), fingerprint(frame)))
        frame_index += 1
    capture.release()
    return samples


def ranges(matches: list[dict]) -> list[dict]:
    accepted = [item for item in matches if item["distance"] <= 16]
    if not accepted:
        return []
    groups = [[accepted[0]]]
    for item in accepted[1:]:
        previous = groups[-1][-1]
        output_step = item["output_t"] - previous["output_t"]
        source_step = item["source_t"] - previous["source_t"]
        if output_step <= 0.75 and -0.25 <= source_step <= 1.25:
            groups[-1].append(item)
        else:
            groups.append([item])
    return [
        {
            "output_start": group[0]["output_t"],
            "output_end": group[-1]["output_t"],
            "source_start": group[0]["source_t"],
            "source_end": group[-1]["source_t"],
            "matched_samples": len(group),
            "mean_distance": round(float(np.mean([item["distance"] for item in group])), 2),
        }
        for group in groups
        if len(group) >= 2
    ]


def map_render(source: list[tuple[float, np.ndarray]], render: Path) -> dict:
    matches = []
    for output_t, output_hash in sampled_fingerprints(render):
        distances = [np.count_nonzero(output_hash != source_hash) for _, source_hash in source]
        index = int(np.argmin(distances))
        matches.append(
            {
                "output_t": output_t,
                "source_t": source[index][0],
                "distance": int(distances[index]),
            }
        )
    source_ranges = ranges(matches)
    jumps = [
        round(source_ranges[index + 1]["source_start"] - source_ranges[index]["source_end"], 3)
        for index in range(len(source_ranges) - 1)
    ]
    return {
        "duration_s": round(cv2.VideoCapture(str(render)).get(cv2.CAP_PROP_FRAME_COUNT)
                            / (cv2.VideoCapture(str(render)).get(cv2.CAP_PROP_FPS) or 30), 3),
        "sample_hz": SAMPLE_HZ,
        "source_ranges": source_ranges,
        "source_range_count": len(source_ranges),
        "timeline_gap_total_s": round(sum(max(0, item) for item in jumps), 3),
        "largest_gap_s": round(max([0.0, *jumps]), 3),
        "chronological_jumps": sum(item > 1.25 for item in jumps),
        "mapping_note": (
            "Approximate pHash mapping; inspect ranges manually before editorial claims."
        ),
        "sample_matches": matches,
    }


def main() -> None:
    source = sampled_fingerprints(SOURCE)
    result = {
        "source": str(SOURCE),
        "method": "pHash-like DCT fingerprints at 2 fps; threshold distance <=16",
        "renders": {key: map_render(source, path) for key, path in RENDERS.items()},
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
