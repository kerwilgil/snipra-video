"""Re-edición de clips por transcript: quitar muletillas / tramos y re-render.

Funciones puras (detección, rangos, remapeo) + concat FFmpeg de los tramos a conservar.
El re-render reusa render_from_base sobre el base concatenado.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.core.media import run

# Muletillas en español (palabra normalizada). Bigramas para multi-palabra.
#
# Solo sonidos de vacilacion PURA: nunca cargan significado, seguros de quitar
# siempre. "pues", "o sea" y "es decir" salieron de esta lista -- son
# conectores que a veces son muletilla y a veces cargan causa/contraste/
# aclaracion real ("pues no se" = duda genuina, no relleno), igual que "so" y
# "like" en ingles. Sin contexto gramatical, un matcher por token no puede
# distinguir el uso vacio del real, y el default cuando hay duda es conservar
# (mismo criterio que usa ChatCut en su guia de limpieza de habla). "como que"
# y "ya saben" quedan pendientes de revisar con el mismo criterio.
FILLER_TERMS = {"eh", "ehm", "em", "este", "mmm", "mm", "ah", "este,"}
FILLER_BIGRAMS = {("como", "que"), ("ya", "saben")}


def _norm(text: str) -> str:
    table = str.maketrans("áéíóúüñ", "aeiouun")
    return text.translate(table).lower().strip(".,!?;:()[]{}\"'")


def detect_filler_ranges(words: list[dict]) -> list[tuple[float, float]]:
    """Devuelve rangos (start, end) en tiempo absoluto de las muletillas detectadas."""
    ranges: list[tuple[float, float]] = []
    i = 0
    norms = [_norm(str(w.get("word", ""))) for w in words]
    while i < len(words):
        w = words[i]
        ws, we = w.get("start"), w.get("end")
        if ws is None or we is None:
            i += 1
            continue
        if i + 1 < len(words) and (norms[i], norms[i + 1]) in FILLER_BIGRAMS:
            nxt = words[i + 1]
            ranges.append((float(ws), float(nxt.get("end", we))))
            i += 2
            continue
        if norms[i] in FILLER_TERMS:
            ranges.append((float(ws), float(we)))
        i += 1
    return ranges


# pausas largas entre palabras: comprimir, no eliminar del todo -- una pausa de
# 0s suena atropellado, el video necesita aire para respirar entre ideas.
LONG_PAUSE_THRESHOLD_S = 0.8
PAUSE_KEEP_S = 0.3


def detect_long_pause_ranges(
    words: list[dict],
    min_pause: float = LONG_PAUSE_THRESHOLD_S,
    keep: float = PAUSE_KEEP_S,
) -> list[tuple[float, float]]:
    """Comprime cada pausa mayor a `min_pause` segundos a `keep` segundos.

    Devuelve el EXCESO de cada pausa como rango a eliminar (no la pausa
    completa): se dejan los primeros `keep` segundos despues de que termina la
    palabra anterior, y se corta el resto hasta que empieza la siguiente. Una
    pausa corta entre oraciones sigue sonando natural; solo el silencio muerto
    se acorta.
    """
    ranges: list[tuple[float, float]] = []
    for prev, nxt in zip(words, words[1:], strict=False):
        prev_end, next_start = prev.get("end"), nxt.get("start")
        if prev_end is None or next_start is None:
            continue
        prev_end, next_start = float(prev_end), float(next_start)
        if next_start - prev_end <= min_pause:
            continue
        remove_start = prev_end + keep
        if next_start > remove_start:
            ranges.append((remove_start, next_start))
    return ranges


# repeticion de toma: misma frase dicha dos veces seguidas (falso arranque),
# se conserva la segunda. Prueba primero la frase mas larga para no dejar
# restos de una repeticion larga tratada como si fueran varias cortas.
REPEAT_MAX_PHRASE_WORDS = 6
REPEAT_MAX_GAP_WORDS = 2
REPEAT_MAX_SEPARATION_S = 6.0


def detect_repeated_take_ranges(words: list[dict]) -> list[tuple[float, float]]:
    """Detecta frases repetidas ('el primero es... el primero es que...').

    Devuelve rangos (start, end) en tiempo absoluto: desde el inicio de la
    primera repeticion hasta el inicio de la segunda (la que se conserva).
    """
    norms = [_norm(str(w.get("word", ""))) for w in words]
    starts = [w.get("start") for w in words]
    n_words = len(words)
    ranges: list[tuple[float, float]] = []
    i = 0
    while i < n_words:
        found = None
        for n in range(REPEAT_MAX_PHRASE_WORDS, 1, -1):
            if i + n > n_words:
                continue
            base = norms[i : i + n]
            if not all(base):
                continue
            for gap in range(REPEAT_MAX_GAP_WORDS + 1):
                j = i + n + gap
                if j + n > n_words:
                    break
                if norms[j : j + n] != base:
                    continue
                if starts[i] is None or starts[j] is None:
                    continue
                if float(starts[j]) - float(starts[i]) > REPEAT_MAX_SEPARATION_S:
                    continue
                found = j
                break
            if found is not None:
                break
        if found is not None:
            ranges.append((float(starts[i]), float(starts[found])))
            i = found
        else:
            i += 1
    return ranges


def merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    ordered = sorted((min(a, b), max(a, b)) for a, b in ranges if b > a)
    for s, e in ordered:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


def compute_kept_ranges(
    clip_start: float, clip_end: float, remove: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Resta los tramos `remove` del rango [clip_start, clip_end]. Devuelve lo que se conserva."""
    kept: list[tuple[float, float]] = []
    cursor = clip_start
    for s, e in merge_ranges(remove):
        s = max(clip_start, min(clip_end, s))
        e = max(clip_start, min(clip_end, e))
        if s > cursor:
            kept.append((round(cursor, 3), round(s, 3)))
        cursor = max(cursor, e)
    if cursor < clip_end:
        kept.append((round(cursor, 3), round(clip_end, 3)))
    return [(s, e) for s, e in kept if e - s >= 0.05]


def _spans(kept: list[tuple[float, float]]) -> tuple[list[tuple[float, float, float]], float]:
    spans: list[tuple[float, float, float]] = []
    acc = 0.0
    for s, e in kept:
        spans.append((s, e, acc))
        acc += e - s
    return spans, round(acc, 3)


def source_ranges_for_edit(
    source: list[tuple[float, float]], kept: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Map kept intervals on the current concatenated timeline back to source."""
    spans, _ = _spans(source)
    result = []
    for left, right in kept:
        for start, end, offset in spans:
            a, b = max(left, offset), min(right, offset + end - start)
            if b > a:
                result.append((round(start + a - offset, 3), round(start + b - offset, 3)))
    return result


def remap_segments(
    segments: list[dict], kept: list[tuple[float, float]]
) -> tuple[list[dict], float]:
    """Remapea words al timeline concatenado (0..total). Descarta words fuera de lo conservado."""
    spans, total = _spans(kept)

    def locate(t: float) -> tuple[float, float, float] | None:
        for s, e, ns in spans:
            if s <= t <= e:
                return s, e, ns
        return None

    new_segments: list[dict] = []
    for seg in segments:
        new_words: list[dict] = []
        for w in seg.get("words") or []:
            ws, we = w.get("start"), w.get("end")
            if ws is None or we is None:
                continue
            span = locate((float(ws) + float(we)) / 2)
            if span is None:
                continue
            s, e, ns = span
            new_words.append(
                {
                    "word": w.get("word", ""),
                    "start": round(ns + (max(float(ws), s) - s), 3),
                    "end": round(ns + (min(float(we), e) - s), 3),
                }
            )
        if new_words:
            new_segments.append(
                {
                    "start": new_words[0]["start"],
                    "end": new_words[-1]["end"],
                    "text": " ".join(str(x["word"]).strip() for x in new_words).strip(),
                    "words": new_words,
                }
            )
    return new_segments, total


def build_concat_base(src: str | Path, kept: list[tuple[float, float]], out: str | Path) -> Path:
    """Extrae cada tramo conservado y los concatena en un único base mp4."""
    from app.pipeline.extract import extract_clip

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent

    parts: list[Path] = []
    for idx, (s, e) in enumerate(kept):
        part = work / f"{out_path.stem}-part{idx}.mp4"
        extract_clip(src, s, e, part)
        parts.append(part)

    if len(parts) == 1:
        parts[0].replace(out_path)
        return out_path

    list_file = work / f"{out_path.stem}-concat.txt"
    list_file.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    cmd = [
        settings.ffmpeg_path,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file.name,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        out_path.name,
    ]
    run(cmd, timeout=1800, cwd=work)
    for p in parts:
        p.unlink(missing_ok=True)
    list_file.unlink(missing_ok=True)
    return out_path
