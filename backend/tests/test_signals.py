"""Tests unitarios de señales de medios sin depender de FFmpeg."""

from app.pipeline.signals import score_candidate
from app.providers.base import ClipCandidate


def test_signal_score_boosts_visual_and_audio_candidate():
    signals = {
        "scene_times": [10.0, 14.0],
        "active_intervals": [(8.0, 20.0)],
        "silence_intervals": [(6.5, 8.0), (20.0, 22.0)],
    }
    cand = ClipCandidate(start=8.0, end=20.0, score=0.5)
    bonus, labels = score_candidate(cand, signals, processing_mode="gaming")
    assert bonus > 0
    assert "cambios visuales" in labels
    assert "audio activo" in labels
