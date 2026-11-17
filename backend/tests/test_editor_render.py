from pathlib import Path
import subprocess

import pytest

from app.core.media import probe
from app.editor import timeline as tl
from app.editor.render import _atempo, render_timeline


@pytest.mark.usefixtures("sample_video")
def test_render_timeline_trims_and_concatenates(sample_video: Path, tmp_path: Path):
    timeline = tl.empty_timeline()
    track = tl.new_track("video")
    timeline["tracks"] = [track]
    timeline = tl.add_clip(timeline, track["id"], tl.new_clip("source", 0, 1_000_000, 0))
    timeline = tl.add_clip(
        timeline, track["id"], tl.new_clip("source", 2_000_000, 3_000_000, 1_000_000)
    )
    output = render_timeline(timeline, {"source": str(sample_video)}, tmp_path / "output.mp4")
    assert output.is_file()
    assert 1.7 <= probe(output).duration <= 2.3


def test_render_timeline_fits_portrait_source_on_landscape_canvas(tmp_path: Path):
    source = tmp_path / "portrait.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=720x1280:rate=30:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    timeline = tl.empty_timeline()
    track = tl.new_track("video")
    timeline["tracks"] = [track]
    timeline = tl.add_clip(timeline, track["id"], tl.new_clip("portrait", 0, 1_000_000, 0))

    output = render_timeline(timeline, {"portrait": str(source)}, tmp_path / "portrait-output.mp4")

    info = probe(output)
    assert output.is_file()
    assert (info.width, info.height) == (1280, 720)
    assert info.duration > 0


def test_speed_filter_is_valid_for_the_entire_validated_range():
    assert _atempo(4) == ["atempo=2", "atempo=2"]
    assert _atempo(0.25) == ["atempo=0.5", "atempo=0.5"]
