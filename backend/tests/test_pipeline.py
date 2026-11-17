"""Tests del pipeline que dependen de FFmpeg (probe, extract, thumbnail)."""

from app.core.media import make_thumbnail, probe
from app.pipeline.extract import extract_clip
from tests.conftest import requires_ffmpeg


@requires_ffmpeg
def test_probe_sample(sample_video):
    info = probe(sample_video)
    assert info.width == 1280
    assert info.height == 720
    assert info.has_audio is True
    assert 19 <= info.duration <= 21


@requires_ffmpeg
def test_extract_and_thumbnail(sample_video, tmp_path):
    out = tmp_path / "clip.mp4"
    extract_clip(sample_video, start=2.0, end=8.0, out=out)
    assert out.exists()
    info = probe(out)
    assert 5.5 <= info.duration <= 6.5  # ~6s

    thumb = tmp_path / "clip.jpg"
    make_thumbnail(out, thumb, at_seconds=1.0)
    assert thumb.exists() and thumb.stat().st_size > 0
