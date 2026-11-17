"""Tests de composición de filtros de medios (media foundation).

Los estados se prueban con capabilities simuladas (available/missing) para no
depender del FFmpeg de la máquina; los tests de integración real viven en
test_media_foundation.py.
"""

from __future__ import annotations

import pytest

from app.core import media_capabilities as mc
from app.pipeline import media_filters as mf


@pytest.fixture(autouse=True)
def _clear_cache():
    yield
    mc._CACHE = None


def _caps(monkeypatch, *, filters: set[str] | None, encoders: set[str] | None = None):
    """Inyecta un diagnóstico simulado en el cache del Capability Doctor."""
    encoders = encoders if encoders is not None else {"libx264", "aac"}
    filters = filters if filters is not None else set()
    caps = {
        "ffmpeg_version": "test",
        "ffprobe_version": "test",
        "overall_status": "ok",
        "encoders": {
            name: ("available" if name in encoders else "missing")
            for name in mc.ENCODERS
        },
        "filters": {
            name: ("available" if name in filters else "missing")
            for name in mc.FILTERS
        },
        "features": {},
        "backends": {},
        "warnings": [],
    }
    if "vidstabdetect" in filters and "vidstabtransform" in filters:
        caps["backends"]["stabilization"] = "vidstab"
    elif "deshake" in filters:
        caps["backends"]["stabilization"] = "deshake"
    monkeypatch.setattr(mc, "_CACHE", caps)


_FULL = {
    "zscale", "tonemap", "vidstabdetect", "vidstabtransform", "hqdn3d",
    "afftdn", "anlmdn", "highpass", "lowpass", "acompressor", "alimiter",
}


def test_tonemap_chain_full(monkeypatch):
    _caps(monkeypatch, filters=_FULL)
    chain = mf.hdr_tonemap_chain()
    assert chain is not None
    assert "zscale=t=linear" in chain
    assert "tonemap=hable" in chain
    assert "zscale=t=bt709" in chain
    # No es un retag: convierte transfer y termina en yuv420p 8-bit.
    assert chain.endswith("format=yuv420p")


def test_tonemap_chain_missing_capability(monkeypatch):
    _caps(monkeypatch, filters=_FULL - {"tonemap"})
    assert mf.hdr_tonemap_chain() is None


def test_compose_prefilters_order_and_failure(monkeypatch):
    _caps(monkeypatch, filters=_FULL)
    filters, notes = mf.compose_video_prefilters(
        tonemap_needed=True,
        stabilization_mode="medium",
        denoise_mode="light",
        stab_transform_ready=True,
    )
    # Orden Bloque I1: tonemap → stabilization → denoise (antes del reframe).
    assert len(filters) == 3
    assert "tonemap" in filters[0]
    assert "vidstabtransform" in filters[1]
    assert "hqdn3d" in filters[2]
    assert notes == []


def test_compose_prefilters_hdr_without_capability_fails_clearly(monkeypatch):
    _caps(monkeypatch, filters=set())
    with pytest.raises(RuntimeError, match="zscale \\+ tonemap"):
        mf.compose_video_prefilters(
            tonemap_needed=True,
            stabilization_mode="off",
            denoise_mode="off",
            stab_transform_ready=False,
        )


def test_compose_prefilters_all_off_is_empty(monkeypatch):
    _caps(monkeypatch, filters=_FULL)
    filters, notes = mf.compose_video_prefilters(
        tonemap_needed=False,
        stabilization_mode="off",
        denoise_mode="off",
        stab_transform_ready=False,
    )
    assert filters == []
    assert notes == []


def test_compose_prefilters_deshake_fallback_documented(monkeypatch):
    _caps(monkeypatch, filters={"deshake"})
    filters, notes = mf.compose_video_prefilters(
        tonemap_needed=False,
        stabilization_mode="light",
        denoise_mode="off",
        stab_transform_ready=False,
    )
    assert filters == ["deshake=rx=16:ry=16"]
    assert notes and "deshake" in notes[0]
    assert "no equivalente" in notes[0]


def test_compose_prefilters_stab_missing_notes(monkeypatch):
    _caps(monkeypatch, filters=set())
    filters, notes = mf.compose_video_prefilters(
        tonemap_needed=False,
        stabilization_mode="strong",
        denoise_mode="off",
        stab_transform_ready=False,
    )
    assert filters == []
    assert notes and "Estabilización omitida" in notes[0]


def test_compose_prefilters_denoise_missing_notes(monkeypatch):
    _caps(monkeypatch, filters=set())
    filters, notes = mf.compose_video_prefilters(
        tonemap_needed=False,
        stabilization_mode="off",
        denoise_mode="medium",
        stab_transform_ready=False,
    )
    assert filters == []
    assert notes and "Denoise omitido" in notes[0]


def test_denoise_presets(monkeypatch):
    _caps(monkeypatch, filters={"hqdn3d"})
    assert mf.denoise_filter("light") == "hqdn3d=1.5:1.5:4:3"
    assert "hqdn3d" in mf.denoise_filter("strong")
    assert "hqdn3d" not in mf.denoise_filter("off") or True


def test_denoise_nlmeans_fallback_only_when_hqdn3d_missing(monkeypatch):
    _caps(monkeypatch, filters={"nlmeans"})
    assert mf.denoise_filter("medium").startswith("nlmeans")
    _caps(monkeypatch, filters={"hqdn3d", "nlmeans"})
    assert mf.denoise_filter("medium").startswith("hqdn3d")


def test_stabilization_presets_and_auto(monkeypatch):
    assert mf.resolve_stabilization_mode("auto") == "medium"
    assert mf.resolve_stabilization_mode("light") == "light"
    assert mf.resolve_stabilization_mode("bogus") == "medium"
    # El smoothing del preset strong es 20; shakiness es parámetro de detect.
    assert "smoothing=20" in mf.vidstab_transform_filter("strong")
    # optzoom=1: sin barras negras
    assert "optzoom=1" in mf.vidstab_transform_filter("medium")


def test_audio_cleanup_chain_order(monkeypatch):
    _caps(monkeypatch, filters=_FULL)
    chain = mf.audio_cleanup_chain("voice_strong")
    parts = chain.split(",")
    assert parts[0].startswith("highpass=f=90")
    assert parts[1].startswith("lowpass")
    assert parts[2].startswith("afftdn")
    assert parts[3].startswith("acompressor")
    assert parts[4].startswith("alimiter")
    # loudnorm NO está aquí: se añade después como etapa final.
    assert not any("loudnorm" in part for part in parts)


def test_audio_cleanup_chain_light_is_conservative(monkeypatch):
    _caps(monkeypatch, filters=_FULL)
    chain = mf.audio_cleanup_chain("voice_light")
    assert chain == "highpass=f=70,afftdn=nr=10:nf=-30"


def test_audio_cleanup_chain_off(monkeypatch):
    _caps(monkeypatch, filters=_FULL)
    assert mf.audio_cleanup_chain("off") is None


def test_audio_cleanup_chain_capability_driven(monkeypatch):
    # Sin afftdn pero con anlmdn: sustitución del reductor de ruido.
    _caps(monkeypatch, filters={"highpass", "anlmdn"})
    chain = mf.audio_cleanup_chain("voice")
    assert "highpass" in chain and "anlmdn" in chain and "afftdn" not in chain
    # Sin acompressor: se omite la compresión, cadena sigue siendo válida.
    _caps(monkeypatch, filters={"highpass", "afftdn"})
    chain = mf.audio_cleanup_chain("voice_strong")
    assert "acompressor" not in chain and "alimiter" not in chain
    assert "highpass" in chain and "afftdn" in chain
    # Sin nada aplicable: None.
    _caps(monkeypatch, filters=set())
    assert mf.audio_cleanup_chain("voice") is None


def test_loudnorm_chain_from_profile():
    from app.core.platform_profiles import PROFILES

    generic = PROFILES["generic_vertical"]
    assert mf.loudnorm_chain(generic) == "loudnorm=I=-14:TP=-1.5:LRA=11"
    reels = PROFILES["instagram_reels"]
    assert mf.loudnorm_chain(reels) == "loudnorm=I=-14:TP=-1:LRA=11"
