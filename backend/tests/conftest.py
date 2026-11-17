"""Fixtures de test: BD aislada + video de prueba generado con FFmpeg."""

import os
import tempfile
from pathlib import Path as _Path

# Aísla BD y almacenamiento ANTES de importar la app (config se cachea al importar).
# Each pytest invocation owns its database, including parallel clean-clone QA.
# A shared fixed directory lets another run's teardown delete live rows.
_TEST_STORAGE = _Path(tempfile.mkdtemp(prefix="snipra_tests_"))
os.environ["STORAGE_DIR"] = str(_TEST_STORAGE)
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_STORAGE / 'test.db').as_posix()}"

import shutil
import subprocess
from pathlib import Path

import pytest


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _has_vision() -> bool:
    try:
        import cv2
        import numpy
        return True
    except ImportError:
        return False


requires_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="FFmpeg no disponible")
requires_vision = pytest.mark.skipif(not _has_vision(), reason="Vision dependencies (opencv-python, numpy) no disponibles")


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """Genera un MP4 de 20s (patrón de test + tono) para validar el pipeline."""
    if not _has_ffmpeg():
        pytest.skip("FFmpeg no disponible")
    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=1280x720:rate=30:duration=20",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=20",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out


@pytest.fixture
def fake_segments() -> dict:
    """Documento de transcript sintético para probar detección sin Whisper."""
    segs = []
    t = 0.0
    for i in range(20):
        segs.append(
            {
                "start": round(t, 3),
                "end": round(t + 3, 3),
                "text": f"Frase número {i} sobre un tema interesante.",
                "words": [],
            }
        )
        t += 3
    return {"language": "es", "duration": t, "segments": segs}
