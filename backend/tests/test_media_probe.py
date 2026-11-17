"""Tests del probe avanzado de medios (media foundation 1.0).

Estructura:
- Parser puro (parse_probe_json) con fixtures JSON realistas: cubre Dolby Vision
  (metadata) y todos los estados sin depender de binarios.
- Integración con FFmpeg real (requires_ffmpeg): SDR, HEVC 10-bit, 29.97fps,
  VFR, audio-only, sin audio, rotado, HDR PQ y HDR HLG.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from app.core.media import parse_probe_json, probe

from tests.conftest import requires_ffmpeg


def _ffmpeg_ok(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]


# ---------------------------------------------------------------------------
# Parser puro
# ---------------------------------------------------------------------------


def _video_stream(**overrides) -> dict:
    stream = {
        "codec_type": "video",
        "codec_name": "h264",
        "profile": "High",
        "width": 1920,
        "height": 1080,
        "pix_fmt": "yuv420p",
        "r_frame_rate": "30/1",
        "avg_frame_rate": "30/1",
        "bits_per_raw_sample": "8",
        "color_transfer": "bt709",
    }
    stream.update(overrides)
    return stream


def _payload(streams: list[dict], duration: str = "10.0") -> dict:
    return {"format": {"duration": duration}, "streams": streams}


def test_parse_sdr_h264_basic_fields():
    info = parse_probe_json(
        _payload([_video_stream(), {"codec_type": "audio", "codec_name": "aac"}])
    )
    assert (info.duration, info.width, info.height) == (10.0, 1920, 1080)
    assert info.has_audio is True
    assert info.codec == "h264"
    assert info.codec_profile == "High"
    assert info.pixel_format == "yuv420p"
    assert info.bit_depth == 8
    assert info.fps == 30.0
    assert (info.fps_num, info.fps_den) == (30, 1)
    assert info.variable_frame_rate_suspected is False
    assert info.hdr is False
    assert info.hdr_format == "sdr"
    assert info.dolby_vision is False


def test_parse_hdr_pq():
    info = parse_probe_json(
        _payload(
            [
                _video_stream(
                    codec_name="hevc",
                    pix_fmt="yuv420p10le",
                    bits_per_raw_sample="10",
                    color_transfer="smpte2084",
                    color_primaries="bt2020",
                    color_space="bt2020nc",
                )
            ]
        )
    )
    assert info.hdr is True
    assert info.hdr_format == "pq"
    assert info.bit_depth == 10


def test_parse_hdr_hlg():
    info = parse_probe_json(
        _payload([_video_stream(color_transfer="arib-std-b67")])
    )
    assert info.hdr is True
    assert info.hdr_format == "hlg"


def test_parse_wide_gamut_sdr_is_not_hdr():
    """BT.2020 con transfer no-HDR es wide gamut SDR, nunca HDR."""
    info = parse_probe_json(
        _payload(
            [
                _video_stream(
                    color_transfer="bt709",
                    color_primaries="bt2020",
                    color_space="bt2020nc",
                )
            ]
        )
    )
    assert info.hdr is False
    assert info.hdr_format == "sdr"
    assert info.color_primaries == "bt2020"


def test_parse_dolby_vision_side_data():
    """Dolby Vision se detecta solo si ffprobe expone el DOVI record."""
    stream = _video_stream(
        color_transfer="smpte2084",
        side_data_list=[
            {
                "side_data_type": "DOVI configuration record",
                "dv_profile": 8,
                "dv_level": 6,
                "rpu_present": 1,
                "el_present": 0,
                "bl_present": 1,
                "bl_compatibility_id": 1,
            }
        ],
    )
    info = parse_probe_json(_payload([stream]))
    assert info.dolby_vision is True
    assert info.dolby_vision_profile == 8
    assert info.hdr is True


def test_parse_rotation_display_matrix():
    stream = _video_stream(
        side_data_list=[
            {
                "side_data_type": "Display Matrix",
                "rotation": 90,
            }
        ]
    )
    info = parse_probe_json(_payload([stream]))
    assert info.rotation == 90


def test_parse_rotation_legacy_tag():
    info = parse_probe_json(
        _payload([_video_stream(tags={"rotate": "180"})])
    )
    assert info.rotation == 180


def test_parse_vfr_suspected():
    info = parse_probe_json(
        _payload([_video_stream(r_frame_rate="30/1", avg_frame_rate="45/2")])
    )
    assert info.fps == pytest.approx(22.5)
    assert info.variable_frame_rate_suspected is True


def test_parse_unknown_vfr_is_none():
    info = parse_probe_json(
        _payload([_video_stream(r_frame_rate="0/0", avg_frame_rate="0/0")])
    )
    assert info.fps is None
    assert info.variable_frame_rate_suspected is None


def test_parse_audio_only():
    info = parse_probe_json(
        _payload(
            [
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": "44100", "channels": 2, "channel_layout": "stereo", "bit_rate": "128000"}
            ]
        )
    )
    assert info.has_audio is True
    assert info.width == 0 and info.height == 0
    assert info.audio_stream_count == 1
    assert info.audio_codec == "aac"
    assert info.sample_rate == 44100
    assert info.channels == 2
    assert info.channel_layout == "stereo"
    assert info.audio_bitrate == 128000
    assert info.hdr is None


def test_parse_no_audio():
    info = parse_probe_json(_payload([_video_stream()]))
    assert info.has_audio is False
    assert info.audio_codec is None
    assert info.audio_stream_count == 0


def test_parse_missing_transfer_keeps_hdr_unknown():
    info = parse_probe_json(_payload([_video_stream(color_transfer=None)]))
    assert info.hdr is None
    assert info.hdr_format is None


def test_parse_video_bitrate_and_stream_count():
    info = parse_probe_json(
        _payload(
            [
                _video_stream(bit_rate="4500000"),
                {"codec_type": "audio", "codec_name": "aac"},
                {"codec_type": "audio", "codec_name": "ac3"},
            ]
        )
    )
    assert info.video_bitrate == 4500000
    assert info.audio_stream_count == 2


def test_probe_fixture_json_contract(tmp_path):
    """Contrato: el payload completo de VideoInfo es serializable."""
    info = parse_probe_json(_payload([_video_stream()]))
    data = json.loads(json.dumps(info.__dict__))
    assert data["width"] == 1920
    assert data["hdr_format"] == "sdr"


# ---------------------------------------------------------------------------
# Integración con FFmpeg real
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def probe_fixtures(tmp_path_factory) -> Path:
    """Genera fixtures pequeños y reproducibles para el probe."""
    d = tmp_path_factory.mktemp("probe_fixtures")

    def ff(*args: str) -> None:
        _ffmpeg_ok(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args])

    # SDR H.264 30fps con audio
    ff(
        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440",
        "-t", "2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(d / "sdr.mp4"),
    )
    # 29.97fps CFR
    ff(
        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30000/1001",
        "-t", "2", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(d / "ntsc.mp4"),
    )
    # HEVC 10-bit
    ff(
        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30",
        "-t", "2", "-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le",
        str(d / "hevc10.mp4"),
    )
    # HDR PQ (BT.2020 + SMPTE ST 2084)
    ff(
        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30",
        "-t", "2", "-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le",
        "-x265-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
        str(d / "hdr_pq.mp4"),
    )
    # HDR HLG
    ff(
        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30",
        "-t", "2", "-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le",
        "-x265-params", "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
        str(d / "hdr_hlg.mp4"),
    )
    # VFR: concat de tramos a 30 y 15 fps
    ff(
        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30",
        "-t", "1", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(d / "_vfr_a.mp4"),
    )
    ff(
        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=15",
        "-t", "1", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(d / "_vfr_b.mp4"),
    )
    (d / "_concat.txt").write_text(
        f"file '{(d / '_vfr_a.mp4').as_posix()}'\nfile '{(d / '_vfr_b.mp4').as_posix()}'\n",
        encoding="utf-8",
    )
    _ffmpeg_ok(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(d / "_concat.txt"),
            "-c", "copy", str(d / "vfr.mp4"),
        ]
    )
    # Rotado 90° (display matrix)
    _ffmpeg_ok(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-display_rotation", "90", "-i", str(d / "sdr.mp4"),
            "-c", "copy", str(d / "rotated.mp4"),
        ]
    )
    # Solo audio
    ff(
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
        "-t", "2", "-c:a", "aac", str(d / "audio_only.m4a"),
    )
    # Sin audio
    ff(
        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30",
        "-t", "2", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-an", str(d / "noaudio.mp4"),
    )
    return d


@requires_ffmpeg
def test_probe_real_sdr_h264(probe_fixtures):
    info = probe(probe_fixtures / "sdr.mp4")
    assert info.width == 320 and info.height == 180
    assert info.has_audio is True
    assert info.codec == "h264"
    assert info.fps == pytest.approx(30.0, abs=0.1)
    assert info.variable_frame_rate_suspected is False
    # El encode de prueba no declara transfer: HDR queda en desconocido.
    assert info.hdr is None
    assert info.dolby_vision is False
    assert info.bit_depth == 8


@requires_ffmpeg
def test_probe_real_2997fps(probe_fixtures):
    info = probe(probe_fixtures / "ntsc.mp4")
    assert info.fps == pytest.approx(29.97, abs=0.02)
    assert info.variable_frame_rate_suspected is False


@requires_ffmpeg
def test_probe_real_hevc_10bit(probe_fixtures):
    info = probe(probe_fixtures / "hevc10.mp4")
    assert info.codec == "hevc"
    assert info.codec_profile == "Main 10"
    assert info.bit_depth == 10
    assert info.pixel_format == "yuv420p10le"


@requires_ffmpeg
def test_probe_real_hdr_pq(probe_fixtures):
    info = probe(probe_fixtures / "hdr_pq.mp4")
    assert info.hdr is True
    assert info.hdr_format == "pq"
    assert info.color_transfer == "smpte2084"
    assert info.color_primaries == "bt2020"


@requires_ffmpeg
def test_probe_real_hdr_hlg(probe_fixtures):
    info = probe(probe_fixtures / "hdr_hlg.mp4")
    assert info.hdr is True
    assert info.hdr_format == "hlg"


@requires_ffmpeg
def test_probe_real_vfr(probe_fixtures):
    info = probe(probe_fixtures / "vfr.mp4")
    assert info.variable_frame_rate_suspected is True


@requires_ffmpeg
def test_probe_real_rotated(probe_fixtures):
    info = probe(probe_fixtures / "rotated.mp4")
    assert info.rotation == 90


@requires_ffmpeg
def test_probe_real_audio_only(probe_fixtures):
    info = probe(probe_fixtures / "audio_only.m4a")
    assert info.has_audio is True
    assert info.width == 0 and info.height == 0
    assert info.audio_codec == "aac"
    assert info.sample_rate == 44100
    assert info.channels == 1


@requires_ffmpeg
def test_probe_real_no_audio(probe_fixtures):
    info = probe(probe_fixtures / "noaudio.mp4")
    assert info.has_audio is False
    assert info.audio_stream_count == 0
