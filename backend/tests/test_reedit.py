"""Tests unitarios del re-edit por transcript (sin FFmpeg)."""

from app.pipeline.extract import _audio_fade_filter
from app.pipeline.reedit import (
    compute_kept_ranges,
    detect_filler_ranges,
    detect_long_pause_ranges,
    detect_repeated_take_ranges,
    merge_ranges,
    remap_segments,
)


def _words(pairs, step=0.3, gap=0.05):
    """Construye words con start/end secuenciales a partir de una lista de tokens."""
    words = []
    t = 0.0
    for token in pairs:
        words.append({"word": token, "start": round(t, 3), "end": round(t + step - gap, 3)})
        t += step
    return words


def test_detect_filler_ranges_single_and_bigram():
    words = [
        {"word": "Bueno", "start": 0.0, "end": 0.4},
        {"word": "eh", "start": 0.5, "end": 0.7},
        {"word": "yo", "start": 0.8, "end": 1.0},
        {"word": "como", "start": 1.1, "end": 1.2},
        {"word": "que", "start": 1.2, "end": 1.5},
        {"word": "creo", "start": 1.6, "end": 2.0},
    ]
    ranges = detect_filler_ranges(words)
    assert (0.5, 0.7) in ranges  # 'eh'
    assert (1.1, 1.5) in ranges  # 'como que'


def test_filler_ranges_no_corta_conectores_dependientes_de_contexto():
    """'pues', 'o sea' y 'es decir' pueden ser muletilla o carga real (causa,
    contraste, aclaracion) -- sin contexto gramatical no se distingue, asi que
    ya no se quitan a ciegas. Mismo criterio que "so"/"like" en ingles: cuando
    hay duda, se conserva.
    """
    words = [
        {"word": "pues", "start": 0.0, "end": 0.3},
        {"word": "no", "start": 0.3, "end": 0.5},
        {"word": "se", "start": 0.5, "end": 0.7},
        {"word": "o", "start": 0.8, "end": 0.9},
        {"word": "sea", "start": 0.9, "end": 1.2},
        {"word": "es", "start": 1.3, "end": 1.4},
        {"word": "decir", "start": 1.4, "end": 1.7},
    ]
    assert detect_filler_ranges(words) == []


def test_detect_repeated_take_ranges_adjacent_repeat():
    words = _words(["el", "primero", "es", "el", "primero", "es", "que"])
    ranges = detect_repeated_take_ranges(words)
    assert (words[0]["start"], words[3]["start"]) in ranges


def test_detect_repeated_take_ranges_with_gap_word():
    words = _words(["el", "primero", "es", "este,", "el", "primero", "es", "que"])
    ranges = detect_repeated_take_ranges(words)
    assert (words[0]["start"], words[4]["start"]) in ranges


def test_detect_repeated_take_ranges_prefers_longest_phrase():
    words = _words(["yo", "creo", "que", "esto", "yo", "creo", "que", "esto", "funciona"])
    ranges = detect_repeated_take_ranges(words)
    assert (words[0]["start"], words[4]["start"]) in ranges
    # No debe quedar una segunda deteccion mas corta sobre las mismas palabras
    assert len(ranges) == 1


def test_detect_repeated_take_ranges_ignores_far_apart_repeats():
    # misma frase repetida, pero separada 10s: es una idea que vuelve, no una
    # toma fallida -- no debe recortarse.
    words = [
        {"word": "el", "start": 0.0, "end": 0.2},
        {"word": "primero", "start": 0.2, "end": 0.6},
        {"word": "es", "start": 0.6, "end": 0.8},
        {"word": "el", "start": 10.0, "end": 10.2},
        {"word": "primero", "start": 10.2, "end": 10.6},
        {"word": "es", "start": 10.6, "end": 10.8},
        {"word": "que", "start": 10.8, "end": 11.0},
    ]
    assert detect_repeated_take_ranges(words) == []


def test_detect_repeated_take_ranges_no_false_positive_on_distinct_speech():
    words = _words(["hoy", "vamos", "a", "hablar", "de", "autoconcepto", "y", "confianza"])
    assert detect_repeated_take_ranges(words) == []


def test_detect_long_pause_ranges_compresses_excess():
    words = [
        {"word": "Bueno", "start": 0.0, "end": 0.4},
        {"word": "creo", "start": 1.6, "end": 2.0},  # pausa de 1.2s
    ]
    ranges = detect_long_pause_ranges(words)
    # se conservan los primeros 0.3s de la pausa (0.4 a 0.7), se quita el resto
    assert ranges == [(0.7, 1.6)]


def test_detect_long_pause_ranges_ignores_short_pauses():
    words = [
        {"word": "Bueno", "start": 0.0, "end": 0.4},
        {"word": "creo", "start": 0.9, "end": 1.3},  # pausa de 0.5s, natural
    ]
    assert detect_long_pause_ranges(words) == []


def test_detect_long_pause_ranges_respects_custom_thresholds():
    # "solo pausas > 0.8s y deja al menos 0.3s" (umbral explicito del usuario)
    words = [
        {"word": "uno", "start": 0.0, "end": 0.2},
        {"word": "dos", "start": 1.5, "end": 1.7},
    ]
    ranges = detect_long_pause_ranges(words, min_pause=1.0, keep=0.5)
    assert ranges == [(0.7, 1.5)]


def test_detect_long_pause_ranges_skips_words_without_timing():
    words = [
        {"word": "uno", "start": 0.0, "end": 0.2},
        {"word": "dos"},  # sin start/end (ASR incompleto)
        {"word": "tres", "start": 2.0, "end": 2.2},
    ]
    assert detect_long_pause_ranges(words) == []


def test_merge_ranges_overlapping():
    assert merge_ranges([(1.0, 2.0), (1.5, 3.0), (5.0, 6.0)]) == [(1.0, 3.0), (5.0, 6.0)]


def test_compute_kept_ranges_removes_inner():
    kept = compute_kept_ranges(10.0, 20.0, [(12.0, 13.0), (15.0, 16.0)])
    assert kept == [(10.0, 12.0), (13.0, 15.0), (16.0, 20.0)]


def test_compute_kept_ranges_full_clip_when_no_removes():
    assert compute_kept_ranges(0.0, 5.0, []) == [(0.0, 5.0)]


def test_audio_fade_filter_uses_short_boundary_fades():
    fade = _audio_fade_filter(10.0)

    assert "afade=t=in:st=0:d=0.030" in fade
    assert "afade=t=out:st=9.970:d=0.030" in fade


def test_remap_segments_to_concatenated_timeline():
    segments = [
        {
            "start": 10.0,
            "end": 20.0,
            "text": "uno dos tres",
            "words": [
                {"word": "uno", "start": 10.0, "end": 11.0},
                {"word": "dos", "start": 12.2, "end": 13.0},  # cae en tramo eliminado
                {"word": "tres", "start": 16.5, "end": 17.0},
            ],
        }
    ]
    kept = compute_kept_ranges(10.0, 20.0, [(12.0, 13.0), (15.0, 16.0)])
    new_segments, total = remap_segments(segments, kept)
    words = new_segments[0]["words"]
    texts = [w["word"] for w in words]
    assert "dos" not in texts  # palabra en tramo eliminado se descarta
    assert words[0]["start"] == 0.0  # 'uno' arranca en 0 del nuevo timeline
    assert total == 8.0  # 10s - 2s eliminados
