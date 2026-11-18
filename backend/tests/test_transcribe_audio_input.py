"""Regresion de cumplimiento: la transcripcion no depende de PyAV (GPL en proceso)."""

import wave

import numpy as np
import pytest

from app.pipeline.transcribe import load_wav_16k_mono


def _write_wav(path, samples, rate=16000, channels=1):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def test_load_wav_returns_normalized_float32(tmp_path):
    wav = tmp_path / "a.wav"
    _write_wav(wav, [0, 16384, -32768, 32767])
    audio = load_wav_16k_mono(wav)
    assert audio.dtype == np.float32
    assert audio.tolist() == pytest.approx([0.0, 0.5, -1.0, 32767 / 32768])


def test_load_wav_rejects_unexpected_format(tmp_path):
    wav = tmp_path / "b.wav"
    _write_wav(wav, [0, 0], rate=44100)
    with pytest.raises(ValueError):
        load_wav_16k_mono(wav)


@pytest.mark.parametrize("intervals", [None, [(10.0, 12.0)]])
def test_transcribe_job_passes_pcm_array_and_preserves_timestamps(monkeypatch, tmp_path, intervals):
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    from app.pipeline import transcribe

    monkeypatch.setattr(transcribe, "find_source", lambda job: tmp_path / "source.mp4")
    monkeypatch.setattr(transcribe, "job_dir", lambda job: tmp_path)

    def extract(src, out, **kwargs):
        _write_wav(out, [0, 16384, -32768])

    monkeypatch.setattr(transcribe, "extract_audio", extract)
    monkeypatch.setattr(transcribe, "extract_audio_segment", extract)
    word = SimpleNamespace(start=0.1, end=0.4, word="Hola")
    segment = SimpleNamespace(start=0.1, end=0.4, text=" Hola ", words=[word])
    model = SimpleNamespace(transcribe=Mock(return_value=(
        iter([segment]), SimpleNamespace(language="es", duration=2.0),
    )))
    monkeypatch.setattr(transcribe, "_get_model", lambda name: model)

    result = transcribe.transcribe_job("test", intervals=intervals)

    audio = model.transcribe.call_args.args[0]
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.tolist() == pytest.approx([0.0, 0.5, -1.0])
    assert model.transcribe.call_args.kwargs == {"word_timestamps": True, "vad_filter": True}
    offset = 10.0 if intervals else 0.0
    assert result["segments"] == [{
        "start": offset + 0.1, "end": offset + 0.4, "text": "Hola",
        "words": [{"start": offset + 0.1, "end": offset + 0.4, "word": "Hola"}],
    }]
    assert json.loads((tmp_path / "segments.json").read_text()) == {
        **result, "transcribed_intervals": [[10.0, 12.0]] if intervals else [],
    }
    assert not list(tmp_path.glob("*.wav"))
