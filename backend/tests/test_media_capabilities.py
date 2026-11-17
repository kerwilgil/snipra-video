"""Tests del Capability Doctor (diagnóstico multimedia).

No dependen de que la máquina de CI tenga todos los filtros: los estados
available/missing se prueban con salidas simuladas de FFmpeg, y unknown con
comandos que fallan. Además hay un test de integración con el FFmpeg real
cuando existe (sin exigir capabilities concretas).
"""

from __future__ import annotations

import shutil

import pytest
from fastapi.testclient import TestClient

from app.core import media_capabilities as mc
from app.main import app

from tests.conftest import requires_ffmpeg

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

_SAMPLE_FILTERS = """Filters:
  T.. abench            A->A       Benchmark part of a filtergraph.
  TSC afftdn            A->A       Remove noise from audio using FFT.
 ... subtitles          V->V       Render text subtitles.
  T.. tonemap           V->V       Apply HDR to SDR tone mapping.
  TSC zscale            V->A        Zscale scaling.
  T.. hqdn3d            V->V       Apply a High Quality 3D Denoiser.
  T.. deshake           V->V       Stabilize shaky video (deprecated).
  T.. vidstabdetect     V->V       Extract global transformations.
  T.. vidstabtransform  V->V       Stabilize frames.
  T.. highpass          A->A       Pass frequencies above the threshold.
"""

_SAMPLE_FILTERS_NO_VIDSTAB = """Filters:
  TSC afftdn            A->A       Remove noise from audio using FFT.
 ... subtitles          V->V       Render text subtitles.
  T.. tonemap           V->V       Apply HDR to SDR tone mapping.
  TSC zscale            V->A        Zscale scaling.
  T.. hqdn3d            V->V       Apply a High Quality 3D Denoiser.
  T.. deshake           V->V       Stabilize shaky video (deprecated).
  T.. highpass          A->A       Pass frequencies above the threshold.
"""

_SAMPLE_FILTERS_NO_DENOISE = """Filters:
 ... subtitles          V->V       Render text subtitles.
  T.. highpass          A->A       Pass frequencies above the threshold.
"""

_SAMPLE_FILTERS_SLICE = """Filters:
 $  TSC loudnorm        A->A       Apply EBU R128 loudness normalization.
 $  T.. vidstabdetect   V->V       Extract global transformations.
 $  T.. vidstabtransform V->V       Stabilize frames.
"""

_SAMPLE_ENCODERS = """Encoders:
 V....D libx264         libx264 libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10
 V....D libx265         libx265 libx265 H.265 / HEVC
 A..... aac             AAC (Advanced Audio Coding) (codec aac)
 V....D prores_ks       prores_ks Apple ProRes (iCodec ProRes)
"""

_SAMPLE_ENCODERS_MINIMAL = """Encoders:
 V....D libx264         libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10
 A..... aac             AAC (Advanced Audio Coding) (codec aac)
"""


@pytest.fixture(autouse=True)
def _clear_cache():
    yield
    mc._CACHE = None


def test_parse_filters_output():
    names = mc._parse_filters_output(_SAMPLE_FILTERS)
    assert names is not None
    assert {"afftdn", "subtitles", "tonemap", "zscale", "hqdn3d"} <= names
    assert {"vidstabdetect", "vidstabtransform", "deshake", "highpass"} <= names


def test_parse_filters_output_handles_slice_marker():
    names = mc._parse_filters_output(_SAMPLE_FILTERS_SLICE)
    assert names is not None
    assert {"loudnorm", "vidstabdetect", "vidstabtransform"} <= names


def test_parse_encoders_output():
    names = mc._parse_encoders_output(_SAMPLE_ENCODERS)
    assert names is not None
    assert {"libx264", "libx265", "aac", "prores_ks"} <= names


def _fake_introspection(monkeypatch, *, version_ok=True, filters="", encoders=""):
    calls = []

    def fake(cmd, timeout=20):
        calls.append(list(cmd))
        joined = " ".join(cmd)
        if "-version" in joined:
            return (
                version_ok,
                "ffmpeg version 9.0.1-test Copyright" if version_ok else "",
            )
        if "-filters" in joined:
            return (bool(filters), filters)
        if "-encoders" in joined:
            return (bool(encoders), encoders)
        return False, ""

    monkeypatch.setattr(mc, "_run_introspection", fake)
    return calls


def test_capabilities_full_feature_set(monkeypatch):
    calls = _fake_introspection(
        monkeypatch,
        filters=_SAMPLE_FILTERS + " $ T.. loudnorm A->A x\n",
        encoders=_SAMPLE_ENCODERS,
    )
    caps = mc.get_media_capabilities(refresh=True)
    assert caps["overall_status"] == "ok"
    assert caps["features"]["hdr_to_sdr"] == mc.AVAILABLE
    assert caps["features"]["stabilization"] == mc.AVAILABLE
    assert caps["backends"]["stabilization"] == "vidstab"
    assert caps["features"]["video_denoise"] == mc.AVAILABLE
    assert caps["backends"]["video_denoise"] == "hqdn3d"
    assert caps["features"]["audio_cleanup"] == mc.AVAILABLE
    assert caps["features"]["subtitles"] == mc.AVAILABLE
    assert caps["features"]["social_export"] == mc.AVAILABLE
    assert caps["ffmpeg_version"] == "9.0.1-test"
    # Introspección completa: version + filters + encoders para cada binario.
    assert len(calls) == 4
    assert caps["encoders"]["libx265"] == mc.AVAILABLE
    assert caps["encoders"]["libsvtav1"] == mc.MISSING


def test_capabilities_deshake_fallback(monkeypatch):
    _fake_introspection(monkeypatch, filters=_SAMPLE_FILTERS_NO_VIDSTAB, encoders=_SAMPLE_ENCODERS_MINIMAL)
    caps = mc.get_media_capabilities(refresh=True)
    assert caps["features"]["stabilization"] == "deshake_fallback"
    assert caps["backends"]["stabilization"] == "deshake"
    assert any("vidstab" in warning for warning in caps["warnings"])
    assert caps["overall_status"] == "degraded"


def test_capabilities_missing_denoise(monkeypatch):
    _fake_introspection(monkeypatch, filters=_SAMPLE_FILTERS_NO_DENOISE, encoders=_SAMPLE_ENCODERS_MINIMAL)
    caps = mc.get_media_capabilities(refresh=True)
    assert caps["features"]["video_denoise"] == mc.MISSING
    assert caps["features"]["hdr_to_sdr"] == mc.MISSING
    assert caps["features"]["stabilization"] == mc.MISSING
    assert caps["overall_status"] == "degraded"


def test_capabilities_introspection_failure_is_unknown_not_missing(monkeypatch):
    """Si el comando de introspección falla, todo es UNKNOWN, nunca MISSING."""
    monkeypatch.setattr(mc, "_run_introspection", lambda cmd, timeout=20: (False, ""))
    caps = mc.get_media_capabilities(refresh=True)
    assert caps["overall_status"] == mc.UNKNOWN
    assert all(status == mc.UNKNOWN for status in caps["filters"].values())
    assert all(status == mc.UNKNOWN for status in caps["encoders"].values())
    assert all(status == mc.UNKNOWN for status in caps["features"].values())
    assert caps["warnings"]


def test_capabilities_cached_until_refresh(monkeypatch):
    calls = _fake_introspection(monkeypatch, filters=_SAMPLE_FILTERS, encoders=_SAMPLE_ENCODERS)
    mc.get_media_capabilities(refresh=True)
    count_after_first = len(calls)
    mc.get_media_capabilities()
    mc.get_media_capabilities()
    assert len(calls) == count_after_first
    mc.get_media_capabilities(refresh=True)
    assert len(calls) > count_after_first


def test_helpers(monkeypatch):
    _fake_introspection(monkeypatch, filters=_SAMPLE_FILTERS, encoders=_SAMPLE_ENCODERS)
    assert mc.has_filter("hqdn3d") is True
    assert mc.has_filter("does_not_exist") is False
    assert mc.has_encoder("libx264") is True
    assert mc.feature_status("hdr_to_sdr") == mc.AVAILABLE


@requires_ffmpeg
def test_capabilities_real_ffmpeg():
    """Integración con el binario real: solo exige contratos, no filtros concretos."""
    caps = mc.get_media_capabilities(refresh=True)
    assert caps["overall_status"] in ("ok", "degraded")
    assert caps["ffmpeg_version"]
    assert caps["ffprobe_version"]
    for status in caps["filters"].values():
        assert status in (mc.AVAILABLE, mc.MISSING)


@requires_ffmpeg
def test_api_system_media_capabilities_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/system/media-capabilities")
        assert response.status_code == 200
        payload = response.json()
        assert payload["overall_status"] in ("ok", "degraded", "unknown")
        assert set(payload["features"]) == {
            "hdr_to_sdr",
            "stabilization",
            "video_denoise",
            "audio_cleanup",
            "subtitles",
            "social_export",
        }
        # No expone rutas locales sensibles.
        assert "path" not in response.text.lower()


@requires_ffmpeg
def test_api_system_media_capabilities_refresh(monkeypatch):
    calls = _fake_introspection(
        monkeypatch, filters=_SAMPLE_FILTERS, encoders=_SAMPLE_ENCODERS_MINIMAL
    )
    with TestClient(app):
        mc.get_media_capabilities(refresh=True)
        before = len(calls)
        mc.get_media_capabilities()
        assert len(calls) == before
        mc.get_media_capabilities(refresh=True)
        assert len(calls) > before
