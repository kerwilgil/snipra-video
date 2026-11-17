"""Trusted renderer: semantic timeline values become allowlisted FFmpeg arguments."""

from __future__ import annotations

import os
from pathlib import Path

from app.core.config import settings
from app.core.media import probe, run
from app.editor.timeline import clip_duration_us, preview_content


class RenderError(RuntimeError):
    pass


def _atempo(rate: float) -> list[str]:
    """FFmpeg accepts atempo only in [0.5, 2]; compose the validated rate."""
    filters: list[str] = []
    while rate > 2:
        filters.append("atempo=2")
        rate /= 2
    while rate < 0.5:
        filters.append("atempo=0.5")
        rate /= 0.5
    filters.append(f"atempo={rate:g}")
    return filters


def _drawtext_fontfile() -> str:
    # Windows: fontfile path escaped for FFmpeg filtergraph.
    # C:\Windows\Fonts\arial.ttf -> C\:\\Windows\\Fonts\\arial.ttf
    if os.name == "nt":
        return "fontfile='C\\:\\\\Windows\\\\Fonts\\\\arial.ttf':"
    return ""


def _drawtext(item: dict, width: int, height: int) -> str:
    # Text never enters a shell. FFmpeg filter escaping is still required because it
    # is a filter grammar; the supported style is deliberately tiny and validated.
    text = (
        str(item["text"])
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace("\n", " ")
    )
    style, position = item["style"], item["position"]
    x = round((float(position["x"]) + 1) * width / 2)
    y = round((float(position["y"]) + 1) * height / 2)
    start = item["timeline_start_us"] / 1e6
    end = item["timeline_end_us"] / 1e6
    fontfile = _drawtext_fontfile()
    return (
        f"drawtext=text='{text}':x={x}-text_w/2:y={y}-text_h/2:"
        f"fontsize={round(float(style['font_size']))}:fontcolor={style['color']}:"
        f"{fontfile}enable='between(t,{start:.6f},{end:.6f})'"
    )


def render_timeline(timeline: dict, assets: dict[str, str], output: Path) -> Path:
    tracks = [track for track in timeline.get("tracks", []) if track.get("type") == "video"]
    proxy_audio_muted = any(
        track.get("type") == "audio" and not track.get("clips") and track.get("muted")
        for track in timeline.get("tracks", [])
    )
    clips = sorted(
        (clip for track in tracks for clip in track.get("clips", [])),
        key=lambda c: c["timeline_start_us"],
    )
    if not clips:
        raise RenderError("La timeline no contiene clips de video")
    canvas = timeline.get("output_canvas", {"width": 1280, "height": 720})
    width, height = int(canvas["width"]), int(canvas["height"])
    work = output.parent / ".parts"
    work.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    for index, clip in enumerate(clips):
        source = Path(assets.get(clip["asset_id"], "")).resolve()
        if not source.is_file():
            raise RenderError("MEDIA_MISSING")
        duration = clip_duration_us(clip) / 1_000_000
        transform, audio = clip.get("transform", {}), clip.get("audio", {})
        crop = transform.get("crop", {"x": 0, "y": 0, "width": 1, "height": 1})
        info = probe(source)
        cw, ch = (
            max(2, round(info.width * float(crop["width"])) // 2 * 2),
            max(2, round(info.height * float(crop["height"])) // 2 * 2),
        )
        cx, cy = (
            round(info.width * float(crop["x"])) // 2 * 2,
            round(info.height * float(crop["y"])) // 2 * 2,
        )
        scale = float(transform.get("scale", 1))
        sw, sh = max(2, round(cw * scale) // 2 * 2), max(2, round(ch * scale) // 2 * 2)
        # `pad` only accepts an output canvas at least as large as its input.
        # A portrait source on a landscape project (or the inverse) therefore
        # needs a deterministic fit before it can be centered with letterboxing.
        # This is rendering geometry only: the timeline ranges stay untouched.
        fit = min(1.0, width / sw, height / sh)
        if fit < 1.0:
            sw = max(2, round(sw * fit) // 2 * 2)
            sh = max(2, round(sh * fit) // 2 * 2)
        x = round((width - sw) / 2 + float(transform.get("x", 0)) * width / 2)
        y = round((height - sh) / 2 + float(transform.get("y", 0)) * height / 2)
        vf = f"crop={cw}:{ch}:{cx}:{cy},scale={sw}:{sh},pad={width}:{height}:{x}:{y}:black,setsar=1"
        af = []
        rate = float(clip.get("playback_rate", 1))
        vf += f",setpts=PTS/{rate:g}"
        if proxy_audio_muted or audio.get("muted"):
            af.append("volume=0")
        else:
            af.append(f"volume={float(audio.get('volume', 1)):g}")
        if audio.get("fade_in_us", 0):
            af.append(f"afade=t=in:st=0:d={int(audio['fade_in_us']) / 1e6:.6f}")
        if audio.get("fade_out_us", 0):
            fade_out = int(audio["fade_out_us"]) / 1e6
            fade_start = max(0, duration - fade_out)
            af.append(f"afade=t=out:st={fade_start:.6f}:d={fade_out:.6f}")
        if rate != 1:
            af.extend(_atempo(rate))
        part = work / f"{index:03d}.mp4"
        run(
            [
                settings.ffmpeg_path,
                "-y",
                "-ss",
                f"{int(clip['source_in_us']) / 1e6:.6f}",
                "-i",
                str(source),
                "-t",
                f"{(int(clip['source_out_us']) - int(clip['source_in_us'])) / 1e6:.6f}",
                "-vf",
                vf,
                "-af",
                ",".join(af),
                # The input trim is source time; after setpts/atempo the part
                # must be explicitly bounded in timeline time for concat.
                "-t",
                f"{duration:.6f}",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-pix_fmt",
                "yuv420p",
                str(part),
            ],
            timeout=1800,
        )
        parts.append(part)
    concat = output.parent / ".concat.txt"
    concat.write_text("".join(f"file '{part.as_posix()}'\n" for part in parts), encoding="utf-8")
    temporary = output.with_suffix(".partial.mp4")
    filters = ",".join(_drawtext(item, width, height) for item in preview_content(timeline))
    command = [settings.ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", str(concat)]
    if filters:
        command += ["-vf", filters]
    command += ["-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", str(temporary)]
    run(command, timeout=1800)
    if probe(temporary).duration <= 0:
        raise RenderError("RENDER_FAILED: salida sin duración")
    os.replace(temporary, output)
    return output
