"""Tests del Delivery QC con FFmpeg real (outputs deliberadamente inválidos).

Cubre los estados PASS/WARN/FAIL del quality gate de salida contra el
platform profile, incluyendo HDR en destino SDR (washed-out guard).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.core.platform_profiles import PROFILES
from app.pipeline.delivery_qc import (
    FAIL,
    PASS,
    WARN,
    measure_loudness,
    run_delivery_qc,
)

from tests.conftest import requires_ffmpeg


def _ff(*args: str) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


@pytest.fixture(scope="module")
def qc_fixtures(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("delivery_qc")

    # Render correcto: 1080x1920 h264 yuv420p, loudnorm, faststart, aac.
    _ff(
        "-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440",
        "-t", "2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-c:a", "aac",
        "-shortest", "-movflags", "+faststart",
        str(d / "good.mp4"),
    )
    # Resolución incorrecta.
    _ff(
        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440",
        "-t", "2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-c:a", "aac", "-shortest", "-movflags", "+faststart",
        str(d / "wrong_res.mp4"),
    )
    # Pixel format incorrecto (yuv444p).
    _ff(
        "-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=30",
        "-t", "2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv444p",
        "-an", "-movflags", "+faststart",
        str(d / "wrong_pixfmt.mp4"),
    )
    # HDR PQ sin tonemap: el QC debe FALLAR (washed-out guard).
    _ff(
        "-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=30",
        "-t", "2",
        "-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le",
        "-x265-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
        "-an", "-movflags", "+faststart",
        str(d / "hdr_pq.mp4"),
    )
    # Loudness rota: audio a volumen altísimo sin normalizar.
    _ff(
        "-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440",
        "-t", "3",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-af", "volume=20dB",
        "-c:a", "aac", "-shortest", "-movflags", "+faststart",
        str(d / "too_loud.mp4"),
    )
    # Mono (WARN de channels) + sample rate 22050 (WARN).
    _ff(
        "-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=22050",
        "-t", "2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-ac", "1",
        "-c:a", "aac", "-shortest", "-movflags", "+faststart",
        str(d / "mono.mp4"),
    )
    # Sin faststart (moov al final).
    _ff(
        "-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440",
        "-t", "2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-c:a", "aac", "-shortest",
        str(d / "no_faststart.mp4"),
    )
    return d


@requires_ffmpeg
def test_qc_pass_on_good_render(qc_fixtures):
    result = run_delivery_qc(
        qc_fixtures / "good.mp4",
        "9x16",
        PROFILES["generic_vertical"],
        expected_duration=2.0,
        expected_has_audio=True,
    )
    assert result.passed is True, result.to_dict()
    by_id = {check.id: check for check in result.checks}
    assert by_id["resolution"].status == PASS
    assert by_id["codec"].status == PASS
    assert by_id["pixel_format"].status == PASS
    assert by_id["loudness"].status == PASS
    assert by_id["true_peak"].status in (PASS, WARN)
    assert by_id["color"].status == PASS
    assert by_id["faststart"].status == PASS
    assert by_id["audio_codec"].status == PASS


@requires_ffmpeg
def test_qc_fail_wrong_resolution(qc_fixtures):
    result = run_delivery_qc(
        qc_fixtures / "wrong_res.mp4",
        "9x16",
        PROFILES["generic_vertical"],
        expected_duration=2.0,
        expected_has_audio=True,
    )
    assert result.passed is False
    by_id = {check.id: check for check in result.checks}
    assert by_id["resolution"].status == FAIL
    assert by_id["aspect"].status == FAIL


@requires_ffmpeg
def test_qc_fail_wrong_pixfmt(qc_fixtures):
    result = run_delivery_qc(
        qc_fixtures / "wrong_pixfmt.mp4",
        "9x16",
        PROFILES["generic_vertical"],
        expected_duration=2.0,
        expected_has_audio=False,
    )
    assert result.passed is False
    by_id = {check.id: check for check in result.checks}
    assert by_id["pixel_format"].status == FAIL


@requires_ffmpeg
def test_qc_fail_hdr_output_on_sdr_profile(qc_fixtures):
    """HDR sin tonemap en destino SDR: FAIL mecánico, nunca PASS silencioso."""
    result = run_delivery_qc(
        qc_fixtures / "hdr_pq.mp4",
        "9x16",
        PROFILES["generic_vertical"],
        expected_duration=2.0,
        expected_has_audio=False,
        source_hdr_format="pq",
    )
    assert result.passed is False
    by_id = {check.id: check for check in result.checks}
    assert by_id["color"].status == FAIL
    assert result.output_hdr_format == "pq"
    assert result.source_hdr_format == "pq"


@requires_ffmpeg
def test_qc_fail_broken_loudness(qc_fixtures):
    result = run_delivery_qc(
        qc_fixtures / "too_loud.mp4",
        "9x16",
        PROFILES["generic_vertical"],
        expected_duration=3.0,
        expected_has_audio=True,
    )
    assert result.passed is False
    by_id = {check.id: check for check in result.checks}
    assert by_id["loudness"].status == FAIL
    assert by_id["true_peak"].status == FAIL


@requires_ffmpeg
def test_qc_warn_editorial_issues_do_not_fail(qc_fixtures):
    """Los WARN editoriales (mono, sample rate, faststart) no tumban el QC."""
    result = run_delivery_qc(
        qc_fixtures / "mono.mp4",
        "9x16",
        PROFILES["generic_vertical"],
        expected_duration=2.0,
        expected_has_audio=True,
    )
    by_id = {check.id: check for check in result.checks}
    assert by_id["channels"].status == WARN
    assert by_id["sample_rate"].status == WARN
    assert result.passed is True

    result = run_delivery_qc(
        qc_fixtures / "no_faststart.mp4",
        "9x16",
        PROFILES["generic_vertical"],
        expected_duration=2.0,
        expected_has_audio=True,
    )
    by_id = {check.id: check for check in result.checks}
    assert by_id["faststart"].status == WARN
    assert result.passed is True


@requires_ffmpeg
def test_qc_duration_max_is_warn(qc_fixtures):
    result = run_delivery_qc(
        qc_fixtures / "good.mp4",
        "9x16",
        PROFILES["instagram_post"],  # max 60s — no se excede con 2s
        expected_duration=2.0,
        expected_has_audio=True,
    )
    assert "duration_max" not in {check.id for check in result.checks}
    # Simulamos el aviso pidiendo duración esperada coherente con un máximo
    # menor: usar x_video-like comprobación manual con un perfil ajustado.
    from dataclasses import replace

    profile = replace(PROFILES["instagram_post"], max_duration_seconds=1.0)
    result = run_delivery_qc(
        qc_fixtures / "good.mp4",
        "9x16",
        profile,
        expected_duration=2.0,
        expected_has_audio=True,
    )
    by_id = {check.id: check for check in result.checks}
    assert by_id["duration_max"].status == WARN
    assert result.passed is True  # editorial: no bloquea


@requires_ffmpeg
def test_qc_missing_file_fails_cleanly(qc_fixtures):
    result = run_delivery_qc(
        qc_fixtures / "does_not_exist.mp4",
        "9x16",
        PROFILES["generic_vertical"],
        expected_duration=2.0,
        expected_has_audio=True,
    )
    assert result.passed is False
    assert result.checks[0].id == "file"
    assert result.checks[0].status == FAIL


@requires_ffmpeg
def test_measure_loudness_real(qc_fixtures):
    lufs, peak = measure_loudness(qc_fixtures / "good.mp4")
    assert lufs is not None
    assert abs(lufs - (-14.0)) < 2.0
    assert peak is not None
    assert peak < -0.5


def test_qc_safe_zone_warn(tmp_path):
    """Subtítulos que invaden el margen seguro: WARN editorial."""
    from app.pipeline.delivery_qc import _subtitle_safe_zone_status

    ass = tmp_path / "subs.ass"
    ass.write_text(
        "[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,64,0,0,0,0,0,0,0,0,100,100,0,0,1,3,1,2,"
        "40,40,50,1\n\n[Events]\n",
        encoding="utf-8",
    )
    tiktok = PROFILES["tiktok"]  # safe_bottom 0.15 → 288px requeridos
    check = _subtitle_safe_zone_status(ass, tiktok)
    assert check is not None
    assert check.status == WARN
    assert "margen seguro" in check.message

    ass_ok = tmp_path / "subs_ok.ass"
    ass_ok.write_text(
        "[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,64,0,0,0,0,0,0,0,0,100,100,0,0,1,3,1,2,"
        "40,40,290,1\n\n[Events]\n",
        encoding="utf-8",
    )
    check = _subtitle_safe_zone_status(ass_ok, tiktok)
    assert check is not None
    assert check.status == PASS
