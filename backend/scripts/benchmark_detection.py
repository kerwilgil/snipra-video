"""CLI de benchmark del motor de deteccion (Paso 7 del plan P0).

Uso:
    uv run python scripts/benchmark_detection.py <video.mp4> <ground_truth.json>

Corre la generacion de candidatos por SEÑAL (build_event_candidates, sin LLM
ni vision -- no requiere API keys ni CLI de suscripcion) sobre el video real,
y compara contra el ground truth. Para benchmarks que necesiten el pool
completo del LLM/vision, construir los ClipCandidate manualmente y llamar a
app.pipeline.benchmark.evaluate() directamente en vez de este script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402
import subprocess  # noqa: E402

from app.pipeline import gaming_edits, signals  # noqa: E402
from app.pipeline.benchmark import evaluate, load_ground_truth  # noqa: E402


def _probe_duration(video_path: str) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    video_path, ground_truth_path = sys.argv[1], sys.argv[2]

    ground_truth = load_ground_truth(ground_truth_path)
    if ground_truth.annotation_status != "human_confirmed":
        print(
            f"ADVERTENCIA: ground truth con annotation_status="
            f"'{ground_truth.annotation_status}' -- NO confirmado por un humano. "
            "Tratar estos resultados como orientativos, no como verdad de referencia."
        )

    duration = _probe_duration(video_path)
    signals_doc = signals.analyze_media("benchmark-cli", video_path, duration)
    candidates = gaming_edits.build_event_candidates(signals_doc, [], duration, top_n=12)

    report = evaluate(candidates, ground_truth)
    print(json.dumps(report.summary(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
