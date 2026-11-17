"""Tests de integración del pipeline de media foundation con FFmpeg real.

Cubren el pipeline unificado (Bloques E-I): HDR→SDR, estabilización,
denoise, limpieza de audio y la combinación completa con captions + reframe
+ delivery QC. Requieren FFmpeg con zscale/tonemap/vidstab/hqdn3d/afftdn;
si falta alguna capability el test se salta de forma justificada (no false
PASS): solo comprueba lo que el entorno realmente soporta.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app.core import config
from app.core import media_capabilities as mc
from app.core.media import probe
from app.pipeline.render import render_from_base
from app.providers.base import ClipCandidate

from tests.conftest import requires_ffmpeg

SEGMENTS = [
    {
        "start": 0.0,
        "end": 1.2,
        "text": "Frase uno para quemar.",
        "words": [{"word": "Frase", "start": 0.0, "end": 0.5}],
    },
    {
        "start": 1.2,
        "end": 2.4,
        "text": "Segunda frase del clip.",
        "words": [{"word": "Segunda", "start": 1.3, "end": 1.9}],
    },
]


def _ff(*args: str) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


@pytest.fixture(autouse=True)
def _real_capabilities():
    mc._CACHE = None
    yield
    mc._CACHE = None


@pytest.fixture(scope="module")
def mf_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("media_foundation")


def _make_sdr_base(path: Path, seconds: int = 3) -> Path:
    _ff(
        "-f", "lavfi", "-i", f"testsrc=size=480x270:rate=30:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(path),
    )
    return path


def _make_hdr_base(path: Path, transfer: str, seconds: int = 3) -> Path:
    _ff(
        "-f", "lavfi", "-i", f"testsrc=size=480x270:rate=30:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le",
        "-x265-params", f"colorprim=bt2020:transfer={transfer}:colormatrix=bt2020nc",
        "-c:a", "aac", "-shortest",
        str(path),
    )
    return path


def _make_shaky_base(path: Path, seconds: int = 3) -> Path:
    _ff(
        "-f", "lavfi", "-i", f"testsrc=size=480x270:rate=30:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-vf",
        "crop=420:210:x='30*sin(2*PI*t)':y='20*sin(2*PI*t*1.3)'",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(path),
    )
    return path


def _make_noisy_audio_base(path: Path, seconds: int = 3) -> Path:
    _ff(
        "-f", "lavfi", "-i", f"testsrc=size=480x270:rate=30:duration={seconds}",
        "-filter_complex",
        f"sine=frequency=440:duration={seconds}[a];"
        f"anoisesrc=d={seconds}:c=pink:r=44100[a:false];"
        f"[a][a:false]amix=inputs=2:duration=first[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(path),
    )
    return path


def _clip_dir(tmp_path: Path, job_id: str, title: str) -> Path:
    return tmp_path / "work" / job_id / "clips" / f"01-{title}"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@requires_ffmpeg
def test_render_all_off_matches_legacy_and_persists_qc(
    mf_dir, monkeypatch, tmp_path
):
    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    base = _make_sdr_base(mf_dir / "sdr_all_off.mp4")
    cand = ClipCandidate(start=0.0, end=3.0, title="Legacy Off", score=0.9)
    assets = render_from_base(
        "job-off",
        base,
        0,
        cand,
        ["9x16"],
        segments=SEGMENTS,
        media_options=None,
    )
    assert len(assets) == 1
    clip_dir = _clip_dir(tmp_path, "job-off", "legacy-off")
    # QC persistido y aprobado
    qc = _read_json(clip_dir / "delivery_qc-9x16.json")
    assert qc["passed"] is True
    assert qc["profile"] == "generic_vertical"
    assert qc["source_hdr_format"] is None
    # timeline expone el delivery QC
    timeline = _read_json(clip_dir / "timeline.json")
    entry = timeline["formats"][0]
    assert entry["delivery_qc"]["passed"] is True
    assert entry["destination_profile"] == "generic_vertical"
    # processing_config registra las opciones de medios
    processing = _read_json(clip_dir / "processing_config.json")
    media = processing["media"]
    assert media["stabilization"] == "off"
    assert media["video_denoise"] == "off"
    assert media["audio_cleanup"] == "off"
    assert media["tonemapped_to_sdr"] is False


@requires_ffmpeg
def test_render_hdr_pq_to_sdr(mf_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    if not (mc.has_filter("zscale") and mc.has_filter("tonemap")):
        pytest.skip("zscale/tonemap no disponibles en este FFmpeg")
    base = _make_hdr_base(mf_dir / "hdr_pq.mp4", "smpte2084")
    assert probe(base).hdr is True
    cand = ClipCandidate(start=0.0, end=3.0, title="HDR PQ", score=0.9)
    assets = render_from_base(
        "job-hdr-pq",
        base,
        0,
        cand,
        ["9x16"],
        segments=SEGMENTS,
        media_options={"color_handling": "auto"},
    )
    out = Path(assets[0].path)
    info = probe(out)
    # Salida SDR BT.709 8-bit: tone map real, no retag.
    assert info.hdr is False
    assert info.color_transfer == "bt709"
    assert info.pixel_format == "yuv420p"
    assert info.codec == "h264"
    qc = _read_json(_clip_dir(tmp_path, "job-hdr-pq", "hdr-pq") / "delivery_qc-9x16.json")
    assert qc["passed"] is True
    assert qc["source_hdr_format"] == "pq"
    assert qc["output_hdr_format"] == "sdr"


@requires_ffmpeg
def test_render_hdr_hlg_to_sdr(mf_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    if not (mc.has_filter("zscale") and mc.has_filter("tonemap")):
        pytest.skip("zscale/tonemap no disponibles en este FFmpeg")
    base = _make_hdr_base(mf_dir / "hdr_hlg.mp4", "arib-std-b67")
    assert probe(base).hdr_format == "hlg"
    cand = ClipCandidate(start=0.0, end=3.0, title="HDR HLG", score=0.9)
    assets = render_from_base(
        "job-hdr-hlg",
        base,
        0,
        cand,
        ["9x16"],
        segments=SEGMENTS,
        media_options={"color_handling": "sdr"},
    )
    info = probe(Path(assets[0].path))
    assert info.hdr is False
    assert info.color_transfer == "bt709"
    assert info.pixel_format == "yuv420p"


@requires_ffmpeg
def test_render_hdr_missing_capability_fails_clearly(mf_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    base = _make_hdr_base(mf_dir / "hdr_no_caps.mp4", "smpte2084")
    # Diagnóstico simulado sin zscale/tonemap.
    monkeypatch.setattr(mc, "_CACHE", {
        "ffmpeg_version": "t", "ffprobe_version": "t", "overall_status": "ok",
        "encoders": {}, "filters": {}, "features": {}, "backends": {},
        "warnings": [],
    })
    cand = ClipCandidate(start=0.0, end=3.0, title="HDR Sin Caps", score=0.9)
    with pytest.raises(RuntimeError, match="HDR detectado"):
        render_from_base(
            "job-hdr-nocaps",
            base,
            0,
            cand,
            ["9x16"],
            segments=None,
            media_options={"color_handling": "auto"},
        )


@requires_ffmpeg
def test_render_stabilization(mf_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    if mc.feature_status("stabilization") not in ("available", "deshake_fallback"):
        pytest.skip("vidstab/deshake no disponibles en este FFmpeg")
    base = _make_shaky_base(mf_dir / "shaky.mp4")
    cand = ClipCandidate(start=0.0, end=3.0, title="Estable", score=0.9)
    assets = render_from_base(
        "job-stab",
        base,
        0,
        cand,
        ["9x16"],
        segments=SEGMENTS,
        media_options={"stabilization": "medium"},
    )
    out = Path(assets[0].path)
    info = probe(out)
    # Output válido, duración aproximada, resolución final correcta.
    assert (info.width, info.height) == (1080, 1920)
    assert abs(info.duration - 3.0) < 1.0
    # Sin temporal transform file restante tras completar.
    clip_dir = _clip_dir(tmp_path, "job-stab", "estable")
    assert not (clip_dir / "stab_transforms.trf").exists()
    assert _read_json(clip_dir / "delivery_qc-9x16.json")["passed"] is True


@requires_ffmpeg
def test_render_stabilization_cleanup_on_failure(mf_dir, monkeypatch, tmp_path):
    """Si el render falla tras el análisis, el transform file se limpia igual."""
    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    if mc.feature_status("stabilization") != "available":
        pytest.skip("vidstab no disponible en este FFmpeg")
    base = _make_shaky_base(mf_dir / "shaky_fail.mp4", seconds=2)
    from unittest.mock import patch

    cand = ClipCandidate(start=0.0, end=2.0, title="Fallo Estable", score=0.9)
    with patch(
        "app.pipeline.render._encode_format", side_effect=RuntimeError("boom")
    ):
        with pytest.raises(RuntimeError, match="boom"):
            render_from_base(
                "job-stab-fail",
                base,
                0,
                cand,
                ["9x16"],
                segments=None,
                media_options={"stabilization": "light"},
            )
    clip_dir = _clip_dir(tmp_path, "job-stab-fail", "fallo-estable")
    assert not (clip_dir / "stab_transforms.trf").exists()


@requires_ffmpeg
def test_render_denoise(mf_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    if mc.feature_status("video_denoise") != "available":
        pytest.skip("hqdn3d/nlmeans no disponibles en este FFmpeg")
    base = _make_sdr_base(mf_dir / "noisy.mp4")
    cand = ClipCandidate(start=0.0, end=3.0, title="Denoise", score=0.9)
    assets = render_from_base(
        "job-denoise",
        base,
        0,
        cand,
        ["9x16"],
        segments=SEGMENTS,
        media_options={"video_denoise": "light"},
    )
    info = probe(Path(assets[0].path))
    assert (info.width, info.height) == (1080, 1920)
    assert info.hdr is not True
    assert _read_json(
        _clip_dir(tmp_path, "job-denoise", "denoise") / "delivery_qc-9x16.json"
    )["passed"] is True


@requires_ffmpeg
def test_render_audio_cleanup(mf_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    if mc.feature_status("audio_cleanup") != "available":
        pytest.skip("afftdn no disponible en este FFmpeg")
    base = _make_noisy_audio_base(mf_dir / "noisy_audio.mp4")
    cand = ClipCandidate(start=0.0, end=3.0, title="Audio Sucio", score=0.9)
    assets = render_from_base(
        "job-audio",
        base,
        0,
        cand,
        ["9x16"],
        segments=SEGMENTS,
        media_options={"audio_cleanup": "voice"},
    )
    out = Path(assets[0].path)
    info = probe(out)
    assert info.has_audio is True
    # El QC de loudness (loudnorm final tras el cleanup) debe pasar.
    qc = _read_json(
        _clip_dir(tmp_path, "job-audio", "audio-sucio") / "delivery_qc-9x16.json"
    )
    by_id = {check["id"]: check for check in qc["checks"]}
    assert by_id["loudness"]["status"] == "pass"
    assert qc["passed"] is True


@requires_ffmpeg
def test_render_audio_cleanup_without_audio_is_noop(mf_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    if mc.feature_status("audio_cleanup") != "available":
        pytest.skip("afftdn no disponible en este FFmpeg")
    silent = mf_dir / "no_audio.mp4"
    _ff(
        "-f", "lavfi", "-i", "testsrc=size=480x270:rate=30:duration=3",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-an", str(silent),
    )
    cand = ClipCandidate(start=0.0, end=3.0, title="Sin Audio", score=0.9)
    assets = render_from_base(
        "job-no-audio",
        silent,
        0,
        cand,
        ["9x16"],
        segments=None,
        media_options={"audio_cleanup": "voice_strong"},
    )
    info = probe(Path(assets[0].path))
    assert info.has_audio is False  # no-op válido: no falla, no inventa audio


@requires_ffmpeg
def test_full_combined_media_pipeline(mf_dir, monkeypatch, tmp_path):
    """SMOKE 6: HDR + estabilización + denoise + cleanup + captions + reframe + QC."""
    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    caps = mc.get_media_capabilities()
    needed = (
        caps["filters"].get("zscale") == "available"
        and caps["filters"].get("tonemap") == "available"
        and caps["backends"].get("stabilization") == "vidstab"
        and caps["filters"].get("hqdn3d") == "available"
        and caps["filters"].get("afftdn") == "available"
    )
    if not needed:
        pytest.skip("Este FFmpeg no tiene todas las capabilities del smoke combinado")

    # Base HDR con movimiento y audio con ruido.
    base = mf_dir / "combined.mp4"
    _ff(
        "-f", "lavfi", "-i", "testsrc=size=480x270:rate=30:duration=3",
        "-filter_complex",
        "crop=420:210:x='30*sin(2*PI*t)':y='20*sin(2*PI*t*1.3)'[v];"
        "sine=frequency=440:duration=3[a];"
        "anoisesrc=d=3:c=pink:r=44100[n];"
        "[a][n]amix=inputs=2:duration=first[aout]",
        "-map", "[v]", "-map", "[aout]",
        "-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le",
        "-x265-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
        "-c:a", "aac",
        str(base),
    )
    assert probe(base).hdr is True

    cand = ClipCandidate(start=0.0, end=3.0, title="Combinado", score=0.9)
    events: list[tuple[str, str]] = []
    assets = render_from_base(
        "job-combined",
        base,
        0,
        cand,
        ["9x16"],
        segments=SEGMENTS,
        media_options={
            "destination_profile": "tiktok",
            "color_handling": "auto",
            "stabilization": "medium",
            "video_denoise": "light",
            "audio_cleanup": "voice",
        },
        event_callback=lambda level, message: events.append((level, message)),
    )
    out = Path(assets[0].path)
    info = probe(out)
    assert (info.width, info.height) == (1080, 1920)
    assert info.hdr is False
    assert info.color_transfer == "bt709"
    assert info.pixel_format == "yuv420p"
    assert abs(info.duration - 3.0) < 1.0
    clip_dir = _clip_dir(tmp_path, "job-combined", "combinado")
    assert not (clip_dir / "stab_transforms.trf").exists()
    qc = _read_json(clip_dir / "delivery_qc-9x16.json")
    assert qc["passed"] is True
    assert qc["profile"] == "tiktok"
    timeline = _read_json(clip_dir / "timeline.json")
    assert timeline["formats"][0]["destination_profile"] == "tiktok"
