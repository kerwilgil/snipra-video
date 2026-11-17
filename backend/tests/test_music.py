"""Tests del modo música (corte sin transcripción/LLM)."""

from app.core.processing_modes import is_music_mode, normalize_processing_mode
from app.pipeline.detect import detect_music_clips


def test_music_mode_aliases():
    assert normalize_processing_mode("music") == "music"
    assert normalize_processing_mode("musica") == "music"
    assert normalize_processing_mode("instrumental") == "music"
    assert is_music_mode("music")
    assert not is_music_mode("auto_smart")


def test_music_clips_count_and_bounds():
    clips = detect_music_clips(
        300.0, {"scene_times": []}, num_clips=5, min_seconds=45, max_seconds=90
    )
    assert len(clips) == 5
    for c in clips:
        assert 45 <= c.duration <= 90
        assert c.start >= 0
        assert c.end <= 300.0
        assert not c.title.startswith("Clip musical 0")  # numerados desde 1


def test_music_clips_snap_to_scene():
    # un inicio cae cerca de un scene cut → se alinea a él
    clips = detect_music_clips(
        240.0,
        {"scene_times": [0.5, 119.0, 121.0]},
        num_clips=2,
        min_seconds=45,
        max_seconds=90,
    )
    starts = [c.start for c in clips]
    assert starts[0] == 0.5  # primer inicio (≈0) se acerca al scene cut 0.5


def test_music_clips_empty_for_zero_duration():
    assert detect_music_clips(0.0, {}, num_clips=5, min_seconds=45, max_seconds=90) == []


def test_music_clips_no_subtitles_source():
    # los clips no llevan segments/words → render no quemará subtítulos
    clips = detect_music_clips(120.0, {}, num_clips=2, min_seconds=30, max_seconds=60)
    assert clips and all(c.reason.startswith("modo música") for c in clips)
