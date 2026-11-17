"""Tests de Platform Delivery Profiles (single source of truth)."""

from __future__ import annotations

import pytest

from app.core.platform_profiles import (
    LEGACY_FORMAT_MAP,
    PROFILES,
    PlatformProfile,
    catalog,
    profile_for_format,
    resolve_profile,
)
from app.pipeline.editorial import protected_zones


def test_required_profiles_exist():
    for profile_id in (
        "generic_vertical",
        "generic_square",
        "generic_landscape",
        "instagram_reels",
        "instagram_post",
        "tiktok",
        "youtube_shorts",
        "youtube",
        "facebook_reels",
        "x_video",
        "linkedin_video",
    ):
        assert profile_id in PROFILES
        profile = PROFILES[profile_id]
        assert isinstance(profile, PlatformProfile)
        assert profile.id == profile_id
        assert profile.target_lufs == pytest.approx(-14.0)
        assert profile.preferred_codec == "h264"
        assert profile.pixel_format == "yuv420p"


def test_legacy_format_mapping():
    """Los formatos legacy se mapean a los genéricos sin migración destructiva."""
    assert LEGACY_FORMAT_MAP["9x16"] == "generic_vertical"
    assert LEGACY_FORMAT_MAP["1x1"] == "generic_square"
    assert LEGACY_FORMAT_MAP["16x9"] == "generic_landscape"
    for fmt, profile_id in LEGACY_FORMAT_MAP.items():
        profile = profile_for_format(fmt)
        assert profile.id == profile_id


def test_resolve_profile_explicit():
    assert resolve_profile("tiktok", "9x16").id == "tiktok"
    assert resolve_profile("instagram_reels", "9x16").id == "instagram_reels"


def test_resolve_profile_auto_falls_back_to_format():
    for value in (None, "", "auto", "unknown-profile"):
        assert resolve_profile(value, "9x16").id == "generic_vertical"
        assert resolve_profile(value, "16x9").id == "generic_landscape"
        assert resolve_profile(value, "1x1").id == "generic_square"


def test_profile_orthogonal_to_format():
    """Formato = geometría; perfil = reglas. 9x16 sirve a varios destinos."""
    tiktok = resolve_profile("tiktok", "9x16")
    reels = resolve_profile("instagram_reels", "9x16")
    assert tiktok.id != reels.id
    assert tiktok.max_duration_seconds == 600.0
    assert reels.max_duration_seconds == 180.0


def test_safe_zones_are_consumed_by_editorial():
    """editorial.protected_zones consume las safe zones del profile (Bloque M)."""
    from app.core.platform_profiles import PROFILES as _P

    profile = _P["tiktok"]  # safe_bottom=0.15
    zones = protected_zones(1080, 1920, captions=True, subject=False, profile=profile)
    caps_zone = zones[0]
    band_bottom = caps_zone["y"] + caps_zone["height"]
    assert band_bottom <= int(1920 * (1.0 - profile.safe_bottom)) + 1
    assert caps_zone["height"] == int(1920 * 0.28)


def test_editorial_zones_without_profile_match_legacy_behavior():
    zones = protected_zones(1080, 1920, captions=True, subject=False)
    assert zones[0]["y"] == int(1920 * 0.62)
    assert zones[0]["height"] == int(1920 * 0.28)


def test_to_dict_contract_is_nested():
    payload = PROFILES["instagram_reels"].to_dict()
    assert payload["frame"] == {"width": 1080, "height": 1920, "aspect": "9:16"}
    assert payload["fps"] == {"preferred": 30.0, "max": 60.0}
    assert payload["audio"]["target_lufs"] == -14.0
    assert payload["color"]["prefer_sdr"] is True
    assert set(payload["safe_zones"]) == {"top", "bottom", "left", "right"}
    assert "quality" in payload


def test_catalog_is_compact_for_ui():
    items = catalog()
    ids = [item["id"] for item in items]
    assert "instagram_reels" in ids and "tiktok" in ids
    for item in items:
        assert set(item) == {"id", "display_name", "aspect", "frame", "max_seconds"}


def test_youtube_allows_hdr_but_pipeline_prefers_sdr_elsewhere():
    assert PROFILES["youtube"].allow_hdr is True
    assert PROFILES["youtube"].prefer_sdr is False
    assert PROFILES["tiktok"].prefer_sdr is True
    assert PROFILES["tiktok"].allow_hdr is False
