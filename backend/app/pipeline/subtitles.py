"""Generacion de subtitulos .ass estilo social a partir del transcript."""

from pathlib import Path

MAX_WORDS_PER_LINE = 4
MAX_CHARS_PER_LINE = 24
MAX_GAP = 0.9
MIN_LINE_SECONDS = 0.8
MAX_LINE_SECONDS = 3.5

KEYWORD_TERMS = {
    "no",
    "nunca",
    "nadie",
    "jamas",
    "error",
    "errores",
    "secreto",
    "secretos",
    "truco",
    "trucos",
    "clave",
    "claves",
    "importante",
    "mira",
    "atencion",
    "ojo",
    "facil",
    "rapido",
    "gratis",
    "mejor",
    "peor",
    "brutal",
    "increible",
    "millones",
    "millon",
    "dinero",
    "exito",
    "imparable",
    "siempre",
    "todo",
    "todos",
    "ahora",
    "nuevo",
}

# keyword normalizada -> emoji. Inyecta max 1 por linea (solo estilos con emoji:True).
# Requiere fuente con cobertura de emoji (Segoe UI Emoji via fontconfig en Windows).
EMOJI_MAP = {
    "dinero": "💰",
    "millones": "💰",
    "millon": "💰",
    "rico": "💰",
    "exito": "🚀",
    "imparable": "🔥",
    "brutal": "🔥",
    "fuego": "🔥",
    "increible": "🤯",
    "secreto": "🤫",
    "secretos": "🤫",
    "atencion": "⚠️",
    "ojo": "👀",
    "mira": "👀",
    "error": "❌",
    "errores": "❌",
    "miedo": "😱",
    "amor": "❤️",
    "tiempo": "⏰",
    "ganar": "🏆",
    "idea": "💡",
    "clave": "🔑",
    "claves": "🔑",
}

STYLE_PRESETS = {
    "viral_bold": {
        "primary": r"&H00FFFFFF",
        "active": r"&H001FD2FF&",
        "keyword": r"&H001FD2FF&",
        "back": r"&H00000000",
        "scale": 1.05,
        "box": 1,
        "font": "Montserrat",
        "outline_scale": 1.55,
        "shadow_scale": 1.4,
        "emoji": False,
        "animate": True,
    },
    "clean": {
        "primary": r"&H00FFFFFF",
        "active": r"&H00FFFFFF&",
        "keyword": r"&H00FFFFFF&",
        "back": r"&HAA000000",
        "scale": 0.88,
        "box": 1,
        "emoji": False,
    },
    "gaming": {
        "primary": r"&H00FFFFFF",
        "active": r"&H0000F5FF&",
        "keyword": r"&H0000F5FF&",
        "back": r"&HAA080808",
        "scale": 1.04,
        "box": 3,
    },
    "corporate": {
        "primary": r"&H00FFFFFF",
        "active": r"&H00FFFFFF&",
        "keyword": r"&H001FD2FF&",
        "back": r"&H99000000",
        "scale": 0.82,
        "box": 3,
        "font": "Arial",
        "animate": False,
        "emoji": False,
    },
    # aliases para jobs/ajustes antiguos
    "opus": {
        "alias": "viral_bold",
    },
    "podcast": {
        "alias": "viral_bold",
    },
    "bold": {
        "alias": "viral_bold",
    },
    "minimal": {
        "primary": r"&H00FFFFFF",
        "active": r"&H00FFFFFF&",
        "keyword": r"&H00FFFFFF&",
        "back": r"&H00000000",
        "scale": 0.82,
        "box": 1,
        "animate": False,
        "emoji": False,
    },
    "neon": {
        "primary": r"&H00FFFFFF",
        "active": r"&H0000F5FF&",
        "keyword": r"&H00FF4DE3&",
        "back": r"&H99080808",
        "scale": 1.02,
        "box": 3,
        "animate": True,
    },
}

HIGHLIGHT_COLORS = {
    "yellow": "#FFD21F",
    "cyan": "#00D4FF",
    "green": "#37E26B",
    "red": "#FF3B30",
    "pink": "#FF4FD8",
    "white": "#FFFFFF",
}

DEFAULT_STYLE_CONFIG = {
    "style": "viral_bold",
    "size": 1.0,
    "position": "lower_middle",
    "text_color": "#FFFFFF",
    "highlight_color": "#FFD21F",
    "secondary_highlight_color": "#00D4FF",
    "stroke_color": "#000000",
    "stroke_width": 8,
    "shadow_enabled": True,
    "background_box": False,
    "uppercase": True,
    "emoji_enabled": False,
    "highlight_max_words": 3,
}


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _collect_words(segments: list[dict], start: float, end: float) -> list[dict]:
    words: list[dict] = []
    for seg in segments:
        for w in seg.get("words", []) or []:
            ws, we = w.get("start"), w.get("end")
            if ws is None or we is None:
                continue
            if we <= start or ws >= end:
                continue
            text = (w.get("word") or "").strip()
            if not text or not _is_clean_word(text):
                continue
            words.append(
                {
                    "start": max(0.0, ws - start),
                    "end": max(0.0, min(we, end) - start),
                    "text": text,
                }
            )
    return words


def _fallback_segments(segments: list[dict], start: float, end: float) -> list[dict]:
    out = []
    for seg in segments:
        ss, se = seg.get("start"), seg.get("end")
        if ss is None or se is None or se <= start or ss >= end:
            continue
        text = (seg.get("text") or "").strip()
        if text:
            out.append(
                {
                    "start": max(0.0, ss - start),
                    "end": max(0.0, min(se, end) - start),
                    "text": text,
                }
            )
    return out


def _chunk_words(words: list[dict]) -> list[dict]:
    lines: list[dict] = []
    buf: list[dict] = []
    for w in words:
        current_text = " ".join([b["text"] for b in buf] + [w["text"]])
        should_cut = (
            len(buf) >= MAX_WORDS_PER_LINE
            or len(current_text) > MAX_CHARS_PER_LINE
            or (buf and (w["start"] - buf[-1]["end"]) > MAX_GAP)
        )
        if buf and should_cut:
            lines.append(_join(buf))
            buf = []
        buf.append(w)
    if buf:
        lines.append(_join(buf))
    return lines


def _join(buf: list[dict]) -> dict:
    return {
        "start": buf[0]["start"],
        "end": buf[-1]["end"],
        "text": " ".join(b["text"] for b in buf).strip(),
        "words": list(buf),
    }


def _escape_ass(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", " ")
        .strip()
    )


def _normalize(text: str) -> str:
    replacements = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")
    return text.translate(replacements).lower().strip(".,!?;:()[]{}\"'")


_CLEAN_RE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789áéíóúüñÁÉÍÓÚÜÑ.,!?;:()'-/ "
)


def _is_clean_word(text: str) -> bool:
    """Devuelve False para tokens basura del modelo Whisper base (:::, |||, }}}…)."""
    if not text:
        return False
    clean_count = sum(1 for ch in text if ch in _CLEAN_RE_CHARS)
    return clean_count / len(text) >= 0.6


def _is_keyword(text: str) -> bool:
    normalized = _normalize(text)
    return normalized in KEYWORD_TERMS or any(ch.isdigit() for ch in normalized)


def _line_emoji(words: list[dict]) -> str:
    """Primer emoji que matchee alguna palabra de la linea (o cadena vacia)."""
    for w in words:
        emoji = EMOJI_MAP.get(_normalize(str(w.get("text", ""))))
        if emoji:
            return emoji
    return ""


def _style(style: str | None) -> dict:
    preset = STYLE_PRESETS.get((style or "viral_bold").lower(), STYLE_PRESETS["viral_bold"])
    if "alias" in preset:
        return STYLE_PRESETS[preset["alias"]]
    return preset


def _hex_to_ass(hex_color: str, default: str) -> str:
    value = (hex_color or "").strip().lstrip("#")
    if len(value) != 6:
        return default
    try:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
    except ValueError:
        return default
    return f"&H00{b:02X}{g:02X}{r:02X}&"


def normalize_style_config(config: dict | None = None, style: str | None = None) -> dict:
    out = dict(DEFAULT_STYLE_CONFIG)
    if style:
        out["style"] = style
    if config:
        out.update({k: v for k, v in config.items() if k in out})
    if out["style"] in {"opus", "podcast", "bold"}:
        out["style"] = "viral_bold"
    if out["highlight_color"] in HIGHLIGHT_COLORS:
        out["highlight_color"] = HIGHLIGHT_COLORS[out["highlight_color"]]
    out["highlight_max_words"] = max(1, min(3, int(out.get("highlight_max_words") or 3)))
    out["stroke_width"] = max(0, min(14, int(out.get("stroke_width") or 8)))
    out["size"] = max(0.65, min(1.4, float(out.get("size") or 1.0)))
    return out


def _word_text(
    words: list[dict],
    active_index: int,
    highlight_size: int,
    preset: dict,
    style_config: dict,
) -> str:
    highlighted = 0
    max_highlights = int(style_config.get("highlight_max_words", 3))
    parts = []
    for i, word in enumerate(words):
        raw_text = str(word["text"])
        text = _escape_ass(raw_text.upper() if style_config.get("uppercase", True) else raw_text)
        is_keyword = _is_keyword(raw_text) and highlighted < max_highlights
        if i == active_index:
            color = preset["keyword"] if is_keyword else preset["active"]
            if is_keyword:
                highlighted += 1
            parts.append(rf"{{\c{color}\fs{highlight_size}\b1}}{text}{{\rDefault}}")
        elif is_keyword:
            highlighted += 1
            parts.append(rf"{{\c{preset['keyword']}\b1}}{text}{{\rDefault}}")
        else:
            parts.append(text)
    return " ".join(parts)


def _fallback_lines(segments: list[dict], start: float, end: float) -> list[dict]:
    lines = []
    for ln in _fallback_segments(segments, start, end):
        parts = [part for part in str(ln["text"]).split() if part.strip()]
        if not parts:
            continue
        # reparte timings uniformemente en el span (sin word-level real); asi cada pseudo-palabra
        # tiene su slot distinto y no se solapan al renderizar
        span = max(0.3, float(ln["end"]) - float(ln["start"]))
        step = span / len(parts)
        words = [
            {
                "start": float(ln["start"]) + i * step,
                "end": float(ln["start"]) + (i + 1) * step,
                "text": part,
            }
            for i, part in enumerate(parts)
        ]
        lines.extend(_chunk_words(words))
    return lines


def build_ass(
    segments: list[dict],
    clip_start: float,
    clip_end: float,
    width: int,
    height: int,
    style: str = "viral_bold",
    style_config: dict | None = None,
) -> str:
    """Devuelve el contenido .ass o cadena vacia si no hay texto en el rango."""
    words = _collect_words(segments, clip_start, clip_end)
    lines = _chunk_words(words) if words else _fallback_lines(segments, clip_start, clip_end)
    if not lines:
        return ""

    cfg = normalize_style_config(style_config, style)
    preset = dict(_style(str(cfg.get("style"))))
    preset["primary"] = _hex_to_ass(str(cfg.get("text_color")), preset["primary"])
    preset["active"] = _hex_to_ass(str(cfg.get("highlight_color")), preset["active"])
    preset["keyword"] = _hex_to_ass(str(cfg.get("highlight_color")), preset["keyword"])
    preset["box"] = 3 if cfg.get("background_box") else 1
    is_vertical = height > width
    fontsize = max(
        28,
        int(height * (0.047 if is_vertical else 0.055) * preset["scale"] * cfg["size"]),
    )
    highlight_size = int(fontsize * 1.12)
    outline = max(2, int(cfg["stroke_width"] * (height / 1920) * preset.get("outline_scale", 1.0)))
    shadow = (
        max(0, int(height * 0.0015 * preset.get("shadow_scale", 1.0)))
        if cfg.get("shadow_enabled")
        else 0
    )
    margin_v = int(height * (0.15 if is_vertical else 0.11))
    margin_h = int(width * 0.07)
    alignment = 2
    if cfg.get("position") == "middle":
        alignment = 5
        margin_v = 0
    elif cfg.get("position") == "top":
        alignment = 8
        margin_v = int(height * 0.10)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{preset.get('font', 'Arial Black')},{fontsize},"
        f"{preset['primary']},&H000000FF,"
        f"{_hex_to_ass(str(cfg.get('stroke_color')), '&H00101010&')},{preset['back']},"
        f"-1,0,0,0,100,100,0,0,{preset['box']},{outline},{shadow},2,"
        f"{margin_h},{margin_h},{margin_v},1\n".replace(",2,", f",{alignment},")
        + "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    emojis_on = bool(cfg.get("emoji_enabled")) and preset.get("emoji", True)
    raw_events: list[tuple[float, float, str]] = []
    for ln in lines:
        line_words = ln.get("words") or []
        emoji = _line_emoji(line_words) if emojis_on and line_words else ""
        suffix = f" {emoji}" if emoji else ""
        if line_words:
            # la linea se sostiene al menos MIN_LINE_SECONDS; cada palabra ocupa hasta el
            # inicio de la siguiente (rellena huecos para que el subtitulo no parpadee)
            line_end = min(
                line_words[0]["start"] + MAX_LINE_SECONDS,
                max(line_words[-1]["end"], line_words[0]["start"] + MIN_LINE_SECONDS),
            )
            for i, word in enumerate(line_words):
                w_start = float(word["start"])
                w_end = float(line_words[i + 1]["start"]) if i + 1 < len(line_words) else line_end
                text = _word_text(line_words, i, highlight_size, preset, cfg) + suffix
                raw_events.append((w_start, w_end, text))
        else:
            raw = str(ln["text"])
            text = _escape_ass(raw.upper() if cfg.get("uppercase", True) else raw)
            raw_events.append((float(ln["start"]), float(ln["end"]), text))

    # post-proceso: ordenar y acotar el fin de cada evento al inicio del siguiente.
    # Garantiza CERO solape (nunca 2 subtitulos a la vez, sin importar timestamps de origen).
    raw_events.sort(key=lambda e: e[0])
    events = []
    for i, (start, end, text) in enumerate(raw_events):
        if i + 1 < len(raw_events):
            end = min(end, raw_events[i + 1][0])
        if end - start < 0.05:  # < 50 ms → demasiado corto para leer, se descarta
            continue
        events.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}")
    return header + "\n".join(events) + "\n"


def write_ass(content: str, out_path: str | Path) -> Path | None:
    if not content:
        return None
    p = Path(out_path)
    p.write_text(content, encoding="utf-8")
    return p
