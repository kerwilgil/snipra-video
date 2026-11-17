"""Construcción del prompt de detección de momentos y parseo de la respuesta JSON."""

import json
import re

from app.providers.base import ClipCandidate, DetectParams, EditorialCard

SYSTEM_PROMPT = """\
Eres un editor experto de contenido viral para TikTok, Reels y YouTube Shorts.
Recibes la transcripción con timestamps de un video largo y eliges los mejores
momentos para convertir en clips cortos independientes.

══ REGLAS ABSOLUTAS (sin excepciones) ══

1. start y end deben coincidir EXACTAMENTE con timestamps de la transcripción.

2. Cada clip debe ser una UNIDAD NARRATIVA COMPLETA con tres partes:
   • INICIO — empieza justo después del cierre (◆) de la frase anterior. El hablante
     abre una idea nueva; el espectador no necesita contexto previo para entenderla.
   • DESARROLLO — la idea se explica con argumentos, historia, datos o ejemplo.
   • CIERRE — termina OBLIGATORIAMENTE en una línea marcada con ◆ (punto, !, ?).
     NUNCA termines en una línea sin ◆. Si el segmento dice "para refrigerar la"
     o "que tiene nuestro" u otra frase incompleta → ESE NO ES UN END VÁLIDO.
     Avanza hasta la siguiente línea ◆ que cierre la oración.

3. Clip auto-contenido: un espectador que no vio el video entiende de qué trata.

4. CUMPLE la duración mínima y máxima indicadas. Para cumplirlas, amplía el end
   hasta el siguiente ◆ dentro del rango, nunca cortes antes.

5. Los marcadores [--- TRANSICIÓN (Xs sin voz) ---] indican silencio largo: intro
   animada del programa, title card, corte de cámara mayor, publicidad, capítulo.
   Un clip NUNCA puede cruzar una TRANSICIÓN. start y end deben estar en el mismo
   bloque continuo de habla entre dos marcadores. Si el único material viral cruza
   una transición, elige otro momento; NUNCA incluyas el marcador dentro del clip.

══ CRITERIOS DE VIRALIDAD (aplica dentro de las reglas anteriores) ══

- Gancho fuerte: afirmación sorprendente, dato impactante, pregunta retórica.
- Pico emocional: revelación, humor, frustración, inspiración, indignación.
- Frase citable: algo que la gente quiere guardar o compartir.
- Remate potente: la última línea del clip impacta y cierra con fuerza.
- Sin contexto largo: el gancho entra en los primeros 3 segundos.

══ MODOS AUTOMÁTICOS ══

Podcast / Entrevista (2+ personas, micrófono visible, debate):
  Prioriza UNA idea bien redondeada: setup → argumento → remate.
  La idea central debe quedar COMPLETAMENTE explicada dentro del clip.
  El remate es la última ◆ del clip; debe ser una conclusión, no una transición.
  Con 3+ hablantes: un clip NO puede terminar con una pregunta abierta, un
  desafío sin resolver o una frase de transición ("y eso es porque...", "claro
  pero...", "exactamente, aunque..."). El intercambio debe estar cerrado: el
  hablante que abrió la idea u otro la remata con conclusión. Si se necesitan
  más segundos para cerrar, amplía el end hasta el ◆ de cierre real.

Vlog / Directo a cámara (una persona, estilo personal):
  Busca: anécdota completa con inicio y desenlace, insight personal fuerte,
  humor auténtico, reacción genuina o momento relatable.
  Gancho = declaración personal sorprendente o situación cotidiana en los
  primeros 3 s que haga al espectador identificarse. Remate cierra la historia.

Gaming (gameplay, kills, reacciones en tiempo real):
  Prioriza acción, tensión, kill streaks, death screens, victorias/derrotas.
  El gancho debe entrar en los primeros 3 s de acción real; evita intros largas.
  Si hay commentary, el clip debe tener TANTO la acción como la reacción.

Tutorial / Educativo → pasos concretos, errores comunes, antes/después.
Música / Ambiental → corta por energía o cambio de escena, sin subtítulos.

══ AUTO-VERIFICACIÓN (obligatoria antes de añadir cada clip al JSON) ══

  ✓ ¿El start es el inicio de una oración nueva (después de una ◆ previa)?
  ✓ ¿El end cae en una línea marcada con ◆?
  ✓ ¿La idea central está completamente explicada dentro del clip?
  ✓ ¿Se entendería sin ver el resto del video?
  ✓ ¿El clip NO cruza ningún marcador [--- TRANSICIÓN ---]?

Si alguna respuesta es NO → ajusta start/end hasta que todas sean SÍ.

Devuelve EXCLUSIVAMENTE un array JSON válido, sin texto adicional ni markdown.

Formato de cada elemento:
{"start": float, "end": float, "title": str, "hook": str, "score": float (0..1),
 "reason": str, "hashtags": [str], "description": str,
 "editorial_cards": [{"kind": "contrast|consequence|derived", "at": float,
 "duration": 2..6, "kicker": str, "text": str, "secondary": str, "evidence": str}]}
- title: título corto y atractivo.
- hook: la primera frase que engancha en los primeros segundos.
- score: potencial viral estimado (0-1, siendo 1 altísimo).
- reason: por qué este momento funciona (breve).
- hashtags: 3-6 hashtags relevantes (con o sin #).
- description: caption lista para publicar (1-2 frases con gancho)."""


def chunk_segments(
    segments: list[dict], max_chars: int = 18000, max_chunks: int = 16
) -> list[list[dict]]:
    """Parte los segmentos en bloques para no truncar transcripciones largas.

    El tamaño de bloque se ajusta para no exceder max_chunks llamadas al LLM en videos
    muy largos (un video de 3h cabe en ~8-10 bloques).
    """
    if not segments:
        return []
    total = sum(len(s.get("text") or "") + 18 for s in segments)
    size = max(max_chars, -(-total // max_chunks))  # ceil(total / max_chunks)

    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_len = 0
    for seg in segments:
        line_len = len(seg.get("text") or "") + 18
        if current and current_len + line_len > size:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(seg)
        current_len += line_len
    if current:
        chunks.append(current)
    return chunks


_SENTENCE_FINAL = frozenset(".!?")
_GAP_TRANSITION = 2.5  # segundos de silencio que marcan transición de escena


def _format_transcript(segments: list[dict], max_chars: int = 40000) -> str:
    """Formatea la transcripción con marcadores ◆ (cierre de oración) y TRANSICIÓN (gap largo).

    ◆ = el LLM puede poner el end de un clip aquí (oración completa).
    [--- TRANSICIÓN ---] = silencio ≥ 2.5s; el LLM NUNCA puede cruzar este punto.
    """
    lines = []
    prev_end: float | None = None
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        start = float(s.get("start", 0))
        end = float(s.get("end", 0))
        if prev_end is not None and (start - prev_end) >= _GAP_TRANSITION:
            gap = start - prev_end
            lines.append(f"[--- TRANSICIÓN ({gap:.1f}s sin voz) ---]")
        marker = " ◆" if text[-1] in _SENTENCE_FINAL else ""
        lines.append(f"[{start:.1f}-{end:.1f}] {text}{marker}")
        prev_end = end
    body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[:max_chars] + "\n[... transcripción truncada ...]"
    return body


def build_user_prompt(segments: list[dict], params: DetectParams) -> str:
    transcript = _format_transcript(segments)
    lang = params.language or "el idioma original"
    content_type = params.processing_mode or "auto"
    gaming_labels = {
        "gaming_auto": "gaming automatico: propone eventos breves y distintos para decidir el edit",
        "gaming_hook": "gaming hook: protege accion, desenlace y repeticion",
        "gaming_highlights": "gaming highlights: propone microjugadas distintas para un montaje",
    }
    content_type = gaming_labels.get(content_type, content_type)
    cards = (
        "\nTarjetas semánticas: propón como máximo 3 por clip y sólo cuando ayuden a "
        "entender un contraste, una consecuencia o una conclusión derivada. Cada tarjeta "
        "debe incluir evidence copiada de la transcripción; no inventes cifras, nombres ni "
        "hechos. Si no hay sustento claro devuelve editorial_cards: [].\n"
        if params.semantic_cards
        else "\nNo propongas tarjetas: devuelve editorial_cards: [] en cada clip.\n"
    )
    return (
        f"Idioma del video: {lang}.\n"
        f"Modo de procesamiento: {content_type}.\n"
        f"Quiero exactamente {params.num_clips} clips (o menos si no hay material suficiente).\n"
        f"Duración por clip: MÍNIMO {params.min_seconds:.0f}s, MÁXIMO {params.max_seconds:.0f}s. "
        f"Un clip de {int(params.min_seconds / 2)}s es INVÁLIDO — debe tener al menos "
        f"{params.min_seconds:.0f}s de contenido.\n\n"
        "Transcripción (◆ = cierre de oración; el end DEBE caer en ◆ | "
        "[--- TRANSICIÓN ---] = corte de escena, NUNCA cruzar):\n"
        f"{transcript}\n"
        f"{cards}\n"
        "Devuelve solo el array JSON con los clips."
    )


def parse_candidates(text: str) -> list[ClipCandidate]:
    """Extrae y valida el array JSON de la respuesta del modelo (tolerante a fences)."""
    cleaned = text.strip()
    # quita fences de markdown ```json ... ```
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    # intenta aislar el array si hay texto alrededor
    if not cleaned.startswith("["):
        match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)

    data = json.loads(cleaned)
    if not isinstance(data, list):
        raise ValueError("La respuesta no es un array JSON")

    out: list[ClipCandidate] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            out.append(
                ClipCandidate(
                    start=float(item.get("start", 0)),
                    end=float(item.get("end", 0)),
                    title=str(item.get("title", "")).strip(),
                    hook=str(item.get("hook", "")).strip(),
                    score=float(item.get("score", 0.5)),
                    reason=str(item.get("reason", "")).strip(),
                    hashtags=_clean_hashtags(item.get("hashtags")),
                    description=str(item.get("description", "")).strip(),
                    editorial_cards=_clean_cards(item.get("editorial_cards")),
                )
            )
        except (TypeError, ValueError):
            continue
    return out


def _clean_cards(value: object) -> list[EditorialCard]:
    if not isinstance(value, list):
        return []
    cards: list[EditorialCard] = []
    for item in value[:3]:
        if not isinstance(item, dict):
            continue
        try:
            cards.append(EditorialCard.model_validate(item))
        except (TypeError, ValueError):
            continue
    return cards


def _clean_hashtags(value: object) -> list[str]:
    """Normaliza hashtags: acepta lista o string, antepone '#', sin espacios ni vacios."""
    if isinstance(value, str):
        raw = re.split(r"[\s,]+", value)
    elif isinstance(value, list):
        raw = [str(v) for v in value]
    else:
        return []
    tags: list[str] = []
    for item in raw:
        tag = re.sub(r"\s+", "", item.strip())
        if not tag:
            continue
        tag = "#" + tag.lstrip("#")
        if tag != "#" and tag not in tags:
            tags.append(tag)
    return tags[:8]
