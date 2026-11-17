from __future__ import annotations

CAPABILITIES = {
    "inspect_project": {"available": True, "access": "read", "reversible": True},
    "inspect_timeline": {"available": True, "access": "read", "reversible": True},
    "get_history": {"available": True, "access": "read", "reversible": True},
    "list_events": {"available": True, "access": "read", "reversible": True},
    "split": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {"clip_id": "string", "at_us": "integer"},
    },
    "trim": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {
            "clip_id": "string",
            "source_in_us": "integer",
            "source_out_us": "integer",
        },
    },
    "move": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {"clip_id": "string", "track_id": "string", "timeline_start_us": "integer"},
    },
    "delete": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {"clip_id": "string", "ripple": "boolean"},
    },
    "duplicate": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {"clip_id": "string", "track_id": "string", "timeline_start_us": "integer"},
    },
    "insert": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {
            "asset_id": "string",
            "track_id": "string",
            "timeline_start_us": "integer",
            "source_in_us": "integer",
            "source_out_us": "integer",
        },
    },
    "clip_properties": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {
            "clip_id": "string",
            "transform": "object",
            "audio": "object",
            "playback_rate": "number",
        },
    },
    "restore": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {"version_number": "integer"},
    },
    "undo": {"available": True, "access": "write", "reversible": True, "params": {}},
    "add_caption": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {
            "track_id": "string",
            "content_id": "string",
            "text": "string",
            "start_us": "integer",
            "end_us": "integer",
            "position": "object",
            "style": "object",
        },
    },
    "edit_caption": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {
            "content_id": "string",
            "text": "string",
            "position": "object",
            "style": "object",
        },
    },
    "add_text": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {
            "track_id": "string",
            "text": "string",
            "start_us": "integer",
            "end_us": "integer",
            "position": "object",
            "style": "object",
        },
    },
    "edit_text": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {
            "content_id": "string",
            "text": "string",
            "position": "object",
            "style": "object",
        },
    },
    "set_volume": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {"clip_id": "string", "volume": "number"},
    },
    "set_mute": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {"clip_id": "string", "muted": "boolean"},
    },
    "set_fade": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {"clip_id": "string", "fade_in_us": "integer", "fade_out_us": "integer"},
    },
    "set_crop": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {"clip_id": "string", "crop": "object"},
    },
    "set_position": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {"clip_id": "string", "x": "number", "y": "number"},
    },
    "set_scale": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {"clip_id": "string", "scale": "number"},
    },
    "change_speed": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {"clip_id": "string", "playback_rate": "number"},
    },
    "track_properties": {
        "available": True,
        "access": "write",
        "reversible": True,
        "params": {"track_id": "string", "properties": "object"},
    },
    "render": {"available": False, "access": "write", "reversible": True},
    "publish": {"available": False, "access": "write", "reversible": False},
}


def public_capabilities() -> list[dict]:
    return [{"name": name, **value} for name, value in CAPABILITIES.items()]