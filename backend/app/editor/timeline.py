"""Exact, deterministic timeline document operations (integer microseconds)."""

from __future__ import annotations

import copy
import uuid


class TimelineError(ValueError):
    code = "INVALID_TIMELINE"


class InvalidRange(TimelineError):
    code = "INVALID_RANGE"


class LockedTrack(TimelineError):
    code = "TRACK_LOCKED"


def ident() -> str:
    return uuid.uuid4().hex


def empty_timeline() -> dict:
    # All creative values are semantic and normalized.  The one transform order
    # shared by browser preview and FFmpeg is: source -> crop -> scale -> position.
    return {
        "time_unit": "us",
        "duration_us": 0,
        "output_canvas": {"width": 1280, "height": 720},
        "tracks": [],
    }


def new_track(kind: str = "video", order: int = 0) -> dict:
    if kind not in {"video", "audio", "text", "overlay"}:
        raise TimelineError("Unsupported track type")
    return {
        "id": ident(),
        "type": kind,
        "order": order,
        "locked": False,
        "muted": False,
        "solo": False,
        "clips": [],
    }


def new_clip(
    asset_id: str,
    source_in_us: int,
    source_out_us: int,
    timeline_start_us: int,
    *,
    linked_group: str | None = None,
    provenance: dict | str = "USER",
) -> dict:
    if source_in_us < 0 or source_out_us <= source_in_us or timeline_start_us < 0:
        raise InvalidRange("Invalid source or timeline range")
    return {
        "id": ident(),
        "asset_id": asset_id,
        "source_in_us": source_in_us,
        "source_out_us": source_out_us,
        "timeline_start_us": timeline_start_us,
        "playback_rate": 1.0,
        "linked_group": linked_group,
        "transform": {
            "x": 0.0,
            "y": 0.0,
            "scale": 1.0,
            "crop": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
        },
        "audio": {"volume": 1.0, "muted": False, "fade_in_us": 0, "fade_out_us": 0},
        "provenance": copy.deepcopy(provenance),
    }


def clip_duration_us(clip: dict) -> int:
    rate = float(clip.get("playback_rate", 1.0))
    if rate <= 0:
        raise InvalidRange("Playback rate must be positive")
    return round((int(clip["source_out_us"]) - int(clip["source_in_us"])) / rate)


def item_duration_us(item: dict) -> int:
    """Duration for media clips and timeline-native text/caption elements."""
    if item.get("kind") in {"caption", "text"}:
        return int(item["timeline_end_us"]) - int(item["timeline_start_us"])
    return clip_duration_us(item)


def clip_end_us(clip: dict) -> int:
    return int(clip["timeline_start_us"]) + item_duration_us(clip)


def _track(timeline: dict, track_id: str) -> dict:
    for track in timeline.get("tracks", []):
        if track["id"] == track_id:
            return track
    raise TimelineError("Track not found")


def _clip(timeline: dict, clip_id: str) -> tuple[dict, dict]:
    for track in timeline.get("tracks", []):
        for clip in track.get("clips", []):
            if clip["id"] == clip_id:
                return track, clip
    raise TimelineError("Clip not found")


def _check_track(track: dict) -> None:
    if track.get("locked"):
        raise LockedTrack("Track is locked")


def recalculate_duration(timeline: dict) -> dict:
    timeline["duration_us"] = max(
        (clip_end_us(c) for t in timeline.get("tracks", []) for c in t.get("clips", [])), default=0
    )
    return timeline


def _validate_timed_content(text: str, start_us: int, end_us: int) -> None:
    if not text.strip():
        raise InvalidRange("Text content is required")
    if start_us < 0 or end_us <= start_us:
        raise InvalidRange("Invalid content timing")


def new_caption(
    text: str,
    start_us: int,
    end_us: int,
    *,
    position: dict | None = None,
    style: dict | None = None,
    provenance: dict | str = "USER",
) -> dict:
    _validate_timed_content(text, start_us, end_us)
    return {
        "id": ident(),
        "kind": "caption",
        "text": text.strip(),
        "timeline_start_us": int(start_us),
        "timeline_end_us": int(end_us),
        "position": _position(position),
        "style": _style(style),
        "provenance": copy.deepcopy(provenance),
    }


def new_text(
    text: str,
    start_us: int,
    end_us: int,
    *,
    position: dict | None = None,
    style: dict | None = None,
    provenance: dict | str = "USER",
) -> dict:
    _validate_timed_content(text, start_us, end_us)
    return {
        "id": ident(),
        "kind": "text",
        "text": text.strip(),
        "timeline_start_us": int(start_us),
        "timeline_end_us": int(end_us),
        "position": _position(position),
        "style": _style(style),
        "provenance": copy.deepcopy(provenance),
    }


def _position(value: dict | None) -> dict:
    value = value or {}
    x, y = float(value.get("x", 0.0)), float(value.get("y", 0.75))
    if not -1 <= x <= 1 or not -1 <= y <= 1:
        raise InvalidRange("Position must be normalized between -1 and 1")
    return {"x": x, "y": y}


def _style(value: dict | None) -> dict:
    value = value or {}
    size = float(value.get("font_size", 42))
    if not 8 <= size <= 160:
        raise InvalidRange("Font size must be between 8 and 160")
    color = str(value.get("color", "#ffffff"))
    if (
        not color.startswith("#")
        or len(color) not in {4, 7}
        or any(c not in "0123456789abcdefABCDEF" for c in color[1:])
    ):
        raise InvalidRange("Color must be a hex value")
    return {"font_size": size, "color": color}


def add_timed_content(timeline: dict, track_id: str, item: dict) -> dict:
    result = copy.deepcopy(timeline)
    track = _track(result, track_id)
    _check_track(track)
    if track.get("type") not in {"text", "overlay"}:
        raise TimelineError("Timed content requires a text or overlay track")
    track["clips"].append(copy.deepcopy(item))
    track["clips"].sort(key=lambda c: (c["timeline_start_us"], c["id"]))
    return recalculate_duration(result)


def edit_timed_content(
    timeline: dict,
    item_id: str,
    *,
    text: str | None = None,
    start_us: int | None = None,
    end_us: int | None = None,
    position: dict | None = None,
    style: dict | None = None,
) -> dict:
    result = copy.deepcopy(timeline)
    track, item = _clip(result, item_id)
    _check_track(track)
    if item.get("kind") not in {"caption", "text"}:
        raise TimelineError("Timeline item is not text content")
    next_text = str(text if text is not None else item["text"])
    next_start = int(start_us if start_us is not None else item["timeline_start_us"])
    next_end = int(end_us if end_us is not None else item["timeline_end_us"])
    _validate_timed_content(next_text, next_start, next_end)
    item.update(
        {"text": next_text.strip(), "timeline_start_us": next_start, "timeline_end_us": next_end}
    )
    if position is not None:
        item["position"] = _position({**item.get("position", {}), **position})
    if style is not None:
        item["style"] = _style({**item.get("style", {}), **style})
    return recalculate_duration(result)


def add_clip(timeline: dict, track_id: str, clip: dict) -> dict:
    result = copy.deepcopy(timeline)
    track = _track(result, track_id)
    _check_track(track)
    track["clips"].append(copy.deepcopy(clip))
    track["clips"].sort(key=lambda c: (c["timeline_start_us"], c["id"]))
    return recalculate_duration(result)


def split_clip(timeline: dict, clip_id: str, at_us: int) -> dict:
    result = copy.deepcopy(timeline)
    track, clip = _clip(result, clip_id)
    _check_track(track)
    start, end = int(clip["timeline_start_us"]), clip_end_us(clip)
    if not start < at_us < end:
        raise InvalidRange("Split point must be inside clip")
    source_split = int(clip["source_in_us"]) + round(
        (at_us - start) * float(clip.get("playback_rate", 1.0))
    )
    right = copy.deepcopy(clip)
    right["id"] = ident()
    right["source_in_us"] = source_split
    right["timeline_start_us"] = at_us
    clip["source_out_us"] = source_split
    track["clips"].append(right)
    track["clips"].sort(key=lambda c: (c["timeline_start_us"], c["id"]))
    return recalculate_duration(result)


def trim_clip(
    timeline: dict,
    clip_id: str,
    *,
    source_in_us: int | None = None,
    source_out_us: int | None = None,
) -> dict:
    result = copy.deepcopy(timeline)
    track, clip = _clip(result, clip_id)
    _check_track(track)
    old_in = int(clip["source_in_us"])
    new_in = old_in if source_in_us is None else int(source_in_us)
    new_out = int(clip["source_out_us"]) if source_out_us is None else int(source_out_us)
    if new_in < 0 or new_out <= new_in:
        raise InvalidRange("Trim creates an empty range")
    if source_in_us is not None:
        clip["timeline_start_us"] += round(
            (new_in - old_in) / float(clip.get("playback_rate", 1.0))
        )
    clip["source_in_us"], clip["source_out_us"] = new_in, new_out
    return recalculate_duration(result)


def move_clip(timeline: dict, clip_id: str, track_id: str, timeline_start_us: int) -> dict:
    if timeline_start_us < 0:
        raise InvalidRange("Timeline start must be non-negative")
    result = copy.deepcopy(timeline)
    origin, clip = _clip(result, clip_id)
    destination = _track(result, track_id)
    _check_track(origin)
    _check_track(destination)
    origin["clips"] = [c for c in origin["clips"] if c["id"] != clip_id]
    clip["timeline_start_us"] = int(timeline_start_us)
    destination["clips"].append(clip)
    destination["clips"].sort(key=lambda c: (c["timeline_start_us"], c["id"]))
    return recalculate_duration(result)


def delete_clip(timeline: dict, clip_id: str, *, ripple: bool = False) -> dict:
    result = copy.deepcopy(timeline)
    track, clip = _clip(result, clip_id)
    _check_track(track)
    start, length = int(clip["timeline_start_us"]), item_duration_us(clip)
    track["clips"] = [c for c in track["clips"] if c["id"] != clip_id]
    if ripple:
        for item in track["clips"]:
            if int(item["timeline_start_us"]) >= start:
                item["timeline_start_us"] -= length
    return recalculate_duration(result)


def duplicate_clip(timeline: dict, clip_id: str, track_id: str, timeline_start_us: int) -> dict:
    result = copy.deepcopy(timeline)
    _, clip = _clip(result, clip_id)
    clone = copy.deepcopy(clip)
    clone["id"] = ident()
    clone["timeline_start_us"] = int(timeline_start_us)
    return add_clip(result, track_id, clone)


def set_clip_properties(
    timeline: dict,
    clip_id: str,
    *,
    playback_rate: float | None = None,
    transform: dict | None = None,
    audio: dict | None = None,
) -> dict:
    result = copy.deepcopy(timeline)
    track, clip = _clip(result, clip_id)
    _check_track(track)
    if playback_rate is not None:
        rate = float(playback_rate)
        if not 0.25 <= rate <= 4.0:
            raise InvalidRange("Playback rate must be between 0.25 and 4")
        clip["playback_rate"] = rate
    if transform is not None:
        scale = float(transform.get("scale", clip["transform"].get("scale", 1.0)))
        if not 0.1 <= scale <= 8.0:
            raise InvalidRange("Scale must be between 0.1 and 8")
        crop = transform.get("crop", clip["transform"].get("crop", {}))
        crop = {
            "x": float(crop.get("x", 0)),
            "y": float(crop.get("y", 0)),
            "width": float(crop.get("width", 1)),
            "height": float(crop.get("height", 1)),
        }
        if (
            crop["x"] < 0
            or crop["y"] < 0
            or crop["width"] <= 0
            or crop["height"] <= 0
            or crop["x"] + crop["width"] > 1
            or crop["y"] + crop["height"] > 1
        ):
            raise InvalidRange("Crop must stay inside the normalized source frame")
        x, y = (
            float(transform.get("x", clip["transform"].get("x", 0))),
            float(transform.get("y", clip["transform"].get("y", 0))),
        )
        if not -1 <= x <= 1 or not -1 <= y <= 1:
            raise InvalidRange("Position must be normalized between -1 and 1")
        clip["transform"].update(
            {
                "x": x,
                "y": y,
                "scale": scale,
                "crop": crop,
            }
        )
    if audio is not None:
        volume = float(audio.get("volume", clip["audio"].get("volume", 1.0)))
        if not 0 <= volume <= 4:
            raise InvalidRange("Volume must be between 0 and 4")
        fade_in = int(audio.get("fade_in_us", clip["audio"].get("fade_in_us", 0)))
        fade_out = int(audio.get("fade_out_us", clip["audio"].get("fade_out_us", 0)))
        if (
            fade_in < 0
            or fade_out < 0
            or fade_in > clip_duration_us(clip)
            or fade_out > clip_duration_us(clip)
        ):
            raise InvalidRange("Fade duration must not exceed clip duration")
        clip["audio"].update(
            {
                "volume": volume,
                "muted": bool(audio.get("muted", clip["audio"].get("muted", False))),
                "fade_in_us": fade_in,
                "fade_out_us": fade_out,
            }
        )
    return recalculate_duration(result)


def set_track_properties(
    timeline: dict,
    track_id: str,
    *,
    locked: bool | None = None,
    muted: bool | None = None,
    solo: bool | None = None,
) -> dict:
    result = copy.deepcopy(timeline)
    track = _track(result, track_id)
    if locked is not None:
        track["locked"] = bool(locked)
    if muted is not None:
        track["muted"] = bool(muted)
    if solo is not None:
        track["solo"] = bool(solo)
    return result


def preview_segments(timeline: dict) -> list[dict]:
    """Canonical frontend preview plan; never duplicate timing rules in the client."""
    segments = []
    proxy_audio_muted = any(
        track.get("type") == "audio" and not track.get("clips") and track.get("muted")
        for track in timeline.get("tracks", [])
    )
    for track in timeline.get("tracks", []):
        if track.get("type") != "video" or track.get("muted"):
            continue
        for clip in track.get("clips", []):
            transform = copy.deepcopy(clip.get("transform", {}))
            if transform.get("crop") == {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}:
                transform.pop("crop")
            audio = copy.deepcopy(clip.get("audio", {}))
            if proxy_audio_muted:
                audio["muted"] = True
            if audio.get("fade_in_us") == 0:
                audio.pop("fade_in_us", None)
            if audio.get("fade_out_us") == 0:
                audio.pop("fade_out_us", None)
            segments.append(
                {
                    "track_id": track["id"],
                    "clip_id": clip["id"],
                    "asset_id": clip["asset_id"],
                    "timeline_start_us": int(clip["timeline_start_us"]),
                    "timeline_end_us": clip_end_us(clip),
                    "source_in_us": int(clip["source_in_us"]),
                    "source_out_us": int(clip["source_out_us"]),
                    "playback_rate": float(clip.get("playback_rate", 1.0)),
                    "transform": transform,
                    "audio": audio,
                }
            )
    return sorted(segments, key=lambda item: (item["timeline_start_us"], item["track_id"]))


def preview_content(timeline: dict) -> list[dict]:
    return sorted(
        (
            {"track_id": track["id"], **copy.deepcopy(item)}
            for track in timeline.get("tracks", [])
            for item in track.get("clips", [])
            if item.get("kind") in {"caption", "text"}
        ),
        key=lambda item: (item["timeline_start_us"], item["id"]),
    )


def resolve_at(timeline: dict, at_us: int) -> list[dict]:
    active = []
    for track in timeline.get("tracks", []):
        for clip in track.get("clips", []):
            if int(clip["timeline_start_us"]) <= at_us < clip_end_us(clip):
                source_us = int(clip["source_in_us"]) + round(
                    (at_us - int(clip["timeline_start_us"])) * float(clip.get("playback_rate", 1.0))
                )
                active.append(
                    {
                        "track_id": track["id"],
                        "clip_id": clip["id"],
                        "asset_id": clip["asset_id"],
                        "source_us": source_us,
                    }
                )
    return active


def replace_clip_asset(timeline: dict, clip_id: str, new_asset_id: str) -> dict:
    """Return a new timeline with the clip's asset_id replaced.
    Preserves all other clip properties (source_in/out, timeline position, transform, audio, etc.)."""
    result = copy.deepcopy(timeline)
    track, clip = _clip(result, clip_id)
    _check_track(track)
    if clip.get("kind") in {"caption", "text"}:
        raise TimelineError("Cannot replace asset on text/caption item")
    clip["asset_id"] = new_asset_id
    return recalculate_duration(result)
