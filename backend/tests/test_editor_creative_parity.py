"""Small deterministic parity contract: normalized Timeline -> preview plan -> FFmpeg."""

from __future__ import annotations

import array
import subprocess
from pathlib import Path

import pytest

from app.core.media import probe
from app.editor import timeline as tl
from app.editor.render import render_timeline


def _fixture(path: Path) -> Path:
    # Four saturated quadrants make crop/scale/position measurable without OCR/VLM.
    vf = (
        "color=black:s=320x180:d=4,drawbox=0:0:160:90:red:t=fill,"
        "drawbox=160:0:160:90:green:t=fill,drawbox=0:90:160:90:blue:t=fill,"
        "drawbox=160:90:160:90:yellow:t=fill"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            vf,
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture
def parity_video(tmp_path: Path) -> Path:
    return _fixture(tmp_path / "quadrants.mp4")


def _frame(path: Path, at: float) -> bytes:
    return subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(at),
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout


def _pixel(frame: bytes, x: int, y: int, width: int = 1280) -> tuple[int, int, int]:
    i = (y * width + x) * 3
    return tuple(frame[i : i + 3])  # type: ignore[return-value]


def _rms(path: Path, start: float = 0, length: float | None = None) -> float:
    command = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(path)]
    if length is not None:
        command += ["-t", str(length)]
    raw = subprocess.run(
        command + ["-vn", "-ac", "1", "-f", "f32le", "-"], check=True, capture_output=True
    ).stdout
    samples = array.array("f")
    samples.frombytes(raw)
    return (sum(value * value for value in samples) / max(1, len(samples))) ** 0.5


def _timeline(**properties: object) -> tuple[dict, dict]:
    timeline = tl.empty_timeline()
    video, text = tl.new_track("video"), tl.new_track("text", 1)
    timeline["tracks"] = [video, text]
    clip = tl.new_clip("fixture", 0, 4_000_000, 0)
    timeline = tl.add_clip(timeline, video["id"], clip)
    if properties:
        timeline = tl.set_clip_properties(timeline, clip["id"], **properties)
    return timeline, {"video": video, "text": text, "clip": clip}


def test_visual_transform_and_content_parity(parity_video: Path, tmp_path: Path):
    timeline, ids = _timeline(
        transform={
            "crop": {"x": 0, "y": 0, "width": 0.5, "height": 0.5},
            "scale": 2,
            "x": 0.25,
            "y": 0,
        }
    )
    timeline = tl.add_timed_content(
        timeline,
        ids["text"]["id"],
        tl.new_caption(
            "C",
            1_000_000,
            2_000_000,
            position={"x": 0, "y": 0},
            style={"font_size": 80, "color": "#ffffff"},
        ),
    )
    segment = tl.preview_segments(timeline)[0]
    assert segment["transform"] == {
        "crop": {"x": 0.0, "y": 0.0, "width": 0.5, "height": 0.5},
        "scale": 2.0,
        "x": 0.25,
        "y": 0.0,
    }
    output = render_timeline(timeline, {"fixture": str(parity_video)}, tmp_path / "visual.mp4")
    assert (probe(output).width, probe(output).height) == (1280, 720)
    before, during, after = _frame(output, 0.5), _frame(output, 1.5), _frame(output, 2.5)
    # crop top-left selects red; scale=2 and x=.25 puts it at x=640..960, y=270..450.
    red, black = _pixel(before, 700, 320), _pixel(before, 500, 320)
    assert red[0] > 180 and red[1] < 80 and red[2] < 80
    assert max(black) < 20
    # Caption exists only in its range near the declared center (white pixels change it).
    assert sum(abs(a - b) for a, b in zip(during, before, strict=True)) > 500
    assert sum(abs(a - b) for a, b in zip(after, before, strict=True)) < 200_000


def test_audio_volume_mute_fade_and_speed_parity(parity_video: Path, tmp_path: Path):
    base, ids = _timeline()
    seventy = tl.set_clip_properties(base, ids["clip"]["id"], audio={"volume": 0.7})
    muted = tl.set_clip_properties(base, ids["clip"]["id"], audio={"muted": True})
    faded = tl.set_clip_properties(
        base, ids["clip"]["id"], audio={"fade_in_us": 1_000_000, "fade_out_us": 1_000_000}
    )
    outputs = [
        render_timeline(item, {"fixture": str(parity_video)}, tmp_path / f"audio-{i}.mp4")
        for i, item in enumerate([base, seventy, muted, faded])
    ]
    ratio = _rms(outputs[1], 0.5, 2) / _rms(outputs[0], 0.5, 2)
    assert 0.62 <= ratio <= 0.78
    assert _rms(outputs[2]) < 0.003
    assert _rms(outputs[3], 0.05, 0.25) < _rms(outputs[3], 1.5, 0.5) * 0.35
    assert _rms(outputs[3], 3.7, 0.2) < _rms(outputs[3], 1.5, 0.5) * 0.35
    for rate in (0.5, 1.5, 2, 0.25):
        timed = tl.set_clip_properties(base, ids["clip"]["id"], playback_rate=rate)
        output = render_timeline(
            timed, {"fixture": str(parity_video)}, tmp_path / f"rate-{rate}.mp4"
        )
        assert (
            abs(probe(output).duration - tl.clip_duration_us(timed["tracks"][0]["clips"][0]) / 1e6)
            < 0.18
        )


def test_empty_audio_track_mute_silences_render(parity_video: Path, tmp_path: Path):
    timeline, _ = _timeline()
    audio_track = tl.new_track("audio", len(timeline["tracks"]))
    timeline["tracks"].append(audio_track)
    audible = render_timeline(timeline, {"fixture": str(parity_video)}, tmp_path / "audible.mp4")
    muted_timeline = tl.set_track_properties(timeline, audio_track["id"], muted=True)
    muted = render_timeline(muted_timeline, {"fixture": str(parity_video)}, tmp_path / "muted.mp4")
    assert tl.preview_segments(muted_timeline)[0]["audio"]["muted"] is True
    assert _rms(audible) > 0.01
    assert _rms(muted) < 0.003


def test_combination_fixture_renders_complete_semantics(parity_video: Path, tmp_path: Path):
    timeline, ids = _timeline(
        playback_rate=1.5,
        transform={
            "crop": {"x": 0.5, "y": 0.5, "width": 0.5, "height": 0.5},
            "scale": 1.4,
            "x": -0.2,
            "y": 0.1,
        },
        audio={"volume": 0.7, "fade_in_us": 300_000},
    )
    timeline = tl.add_timed_content(
        timeline,
        ids["text"]["id"],
        tl.new_text("T", 300_000, 1_000_000, position={"x": -0.2, "y": 0.1}),
    )
    output = render_timeline(timeline, {"fixture": str(parity_video)}, tmp_path / "combined.mp4")
    info = probe(output)
    assert (info.width, info.height, info.has_audio) == (1280, 720, True)
    assert abs(info.duration - 4 / 1.5) < 0.18
    pixel = _pixel(_frame(output, 0.5), 400, 400)
    assert pixel[0] > 150 and pixel[1] > 150 and pixel[2] < 100  # cropped yellow quadrant
    assert _rms(output, 0.7, 0.5) > 0.01
