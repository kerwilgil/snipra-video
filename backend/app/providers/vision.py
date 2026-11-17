"""Re-ranking visual de candidatos con un modelo multimodal (visión).

El detector de texto (LLM sobre el transcript) elige un pool de candidatos.
Este módulo extrae unos pocos fotogramas de cada candidato y deja que el
modelo CON VISIÓN puntúe qué tan atractivo/viral es lo que se VE (acción,
expresividad, momentos llamativos) — algo que el análisis de solo texto no
capta. Reordena el pool combinando score textual + visual.

Diseño: barato y a prueba de fallos. Si FFmpeg o la API fallan, se devuelven
los candidatos sin cambios (nunca rompe el job).
"""

from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
import time
from collections.abc import Callable

from app.core.config import settings
from app.core.sanitize import sanitize_error_text
from app.providers.base import ClipCandidate, ScoreContribution
from app.providers.structured_visual import (
    ANALYSIS_VERSION,
    FRAME_SAMPLING_VERSION,
    PROMPT_VERSION,
    SCORING_PROFILE_VERSION,
    StructuredVisualCache,
    deterministic_visual_score,
    inference_cache_key,
    parse_structured_response,
)

logger = logging.getLogger(__name__)


def _score_structured_shadow(provider: str, model: str, frames: list[bytes], title: str) -> str:
    """Provider adapter for P1.1; kept separate so shadow cannot affect legacy scoring."""
    if provider != "codex":
        raise RuntimeError(f"Structured visual provider unsupported: {provider}")
    import tempfile
    from pathlib import Path

    from app.providers.cli_subscription import _CliError, resolve_provider_cli
    from app.providers.structured_visual import structured_prompt

    exe = resolve_provider_cli("codex")
    if exe is None:
        raise _CliError("El CLI de Codex no está instalado.")
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        paths: list[str] = []
        for index, frame in enumerate(frames):
            path = directory / f"structured-frame-{index}.jpg"
            path.write_bytes(frame)
            paths.append(str(path.resolve()))
        output = directory / "structured-output.json"
        process = subprocess.run(
            [
                exe,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-rules",
                "--model",
                model,
                "-o",
                str(output),
                structured_prompt(title),
                "-i",
                *paths,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=300,
            cwd=directory,
        )
        if process.returncode != 0 or not output.exists():
            raise _CliError(f"Codex structured vision failed: {(process.stderr or '')[:200]}")
        return output.read_text(encoding="utf-8")


def _append_structured_shadow(
    candidate: ClipCandidate, frames: list[bytes], provider: str, model: str, video_path: str = ""
) -> ClipCandidate:
    """Best-effort telemetry only: never changes score, reason, or selection fields."""
    if not getattr(settings, "enable_structured_visual_shadow", False):
        return candidate
    try:
        timestamps = [round(candidate.start + candidate.duration * i / 5, 3) for i in range(1, 5)]
        fingerprint = f"{video_path}|{len(frames)}"
        key = inference_cache_key(
            fingerprint, candidate.start, candidate.end, timestamps, provider, model
        )
        cache = StructuredVisualCache(settings.storage_dir / "mediashorts.db")
        cached = cache.get(key)
        cache_hit = cached is not None
        if cached is None:
            started = time.monotonic()
            raw = _score_structured_shadow(provider, model, frames, candidate.title)
            features = parse_structured_response(raw)
            cached = {
                "raw_response": raw,
                "features": features.model_dump(),
                "request_duration_ms": round((time.monotonic() - started) * 1000),
            }
            cache.put(key, cached)
        features = parse_structured_response(json.dumps(cached["features"]))
        metadata = {
            "structured_event_features": features.model_dump(),
            "deterministic_visual_score": deterministic_visual_score(features),
            "legacy_vlm_score": candidate.score,
            "provider": provider,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "analysis_version": ANALYSIS_VERSION,
            "frame_sampling_version": FRAME_SAMPLING_VERSION,
            "scoring_profile_version": SCORING_PROFILE_VERSION,
            "cache_hit": cache_hit,
            "cache_key": key,
            "request_duration_ms": cached.get("request_duration_ms", 0),
        }
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning(
            "Structured visual shadow failed for '%s': %s",
            candidate.title, sanitize_error_text(str(exc)),
        )
        return candidate
    entry = ScoreContribution(
        signal="structured_visual_shadow",
        stage="experimental_p1_1",
        score_before=candidate.score,
        contribution=0.0,
        score_after=candidate.score,
        metadata=metadata,
    )
    return candidate.model_copy(update={"score_breakdown": [*candidate.score_breakdown, entry]})

# NVIDIA NO esta aca a proposito. Sigue siendo un proveedor de TEXTO valido; lo
# que no sirve es su modelo multimodal gratuito (llama-3.2-90b-vision-instruct).
# Dos fallos medidos sobre frames reales:
#   1) Contexto: 32k tokens y la imagen viaja en base64 DENTRO del prompt, a
#      ~0.95 tokens/byte. 4 frames de 512px = 55.755 tokens -> 400 siempre, en
#      gameplay Y en podcast. Fallaba en silencio porque este modulo nunca
#      rompe el job.
#   2) Calidad: aun reducido a 1 frame de 336px, alucina. Sobre un frame de
#      Call of Duty respondio "un hombre lanzando un balon"; sobre uno de
#      podcast (un hombre ante un microfono) respondio "una multitud en un
#      parque, de todas las edades". Una puntuacion inventada es peor que
#      ninguna: desordena el ranking en vez de mejorarlo.
# Para reactivarlo basta agregarlo al frozenset; el scorer y sus limites siguen
# implementados abajo.
VISION_PROVIDERS = frozenset({"claude", "openai", "claude_code", "codex"})
FRAMES_PER_CLIP = 4
FRAME_WIDTH = 512

PROVIDER_FRAME_LIMITS = {
    "nvidia": (1, 336),
}
DEFAULT_FRAME_LIMITS = (FRAMES_PER_CLIP, FRAME_WIDTH)


VISION_WEIGHT = 0.45  # peso del score visual frente al textual
# Modelo multimodal GRATIS en NVIDIA NIM. Los modelos de texto del provider
# (llama-3.3, nemotron, deepseek) NO ven imágenes, por eso se fuerza este.
NVIDIA_VISION_MODEL = "auto"  # Compatibility alias; resolve against the live catalog.
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def frame_limits(provider: str) -> tuple[int, int]:
    """(n_frames, ancho) admisibles por proveedor."""
    return PROVIDER_FRAME_LIMITS.get((provider or "").lower(), DEFAULT_FRAME_LIMITS)


_PROMPT = (
    "Eres editor de clips virales para TikTok/Reels/Shorts. Estos son fotogramas "
    'en orden de UN clip candidato titulado: "{title}". Evalúa qué tan atractivo '
    "es VISUALMENTE para enganchar en los primeros segundos: acción, expresividad, "
    "gestos, momentos llamativos, claridad de la escena. Ignora el audio (no lo tienes). "
    "Si muestra muerte propia/killcam recibida, menú, marcador o sólo desplazamiento sin "
    "acción, el score DEBE ser <= 0.25 aunque haya medallas. peak_frame es el fotograma "
    "1-4 donde ocurre la mejor acción visible; outcome debe ser kill, victory, death, "
    "menu, scoreboard, idle o neutral. Responde SOLO JSON: "
    '{{"score": <0.0-1.0>, "reason": "<motivo muy breve>", '
    '"peak_frame": <1-4>, "outcome": "<resultado>"}}.'
)

# Los candidatos de la Etapa 1 nacen de picos de audio y cambios de escena, no
# del transcript, asi que el redactor de titulos (que lee texto) los deja mudos:
# salian como "(sin titulo)" y con hashtags genericos. Como la vision YA esta
# mirando esos fotogramas para puntuarlos, pedirle el copy en la misma llamada
# no cuesta una llamada extra.
_PROMPT_COPY = (
    "Eres editor de clips virales para TikTok/Reels/Shorts. Estos son fotogramas "
    "en orden de UN clip de gameplay/video SIN transcripción disponible: no hay "
    "texto, solo lo que se VE. Evalúa qué tan atractivo es visualmente para "
    "enganchar en los primeros segundos y ESCRIBE el copy a partir de la imagen. "
    "El título debe describir lo que ocurre, en español, sin inventar hechos que "
    "no se vean. Si muestra muerte propia/killcam recibida, menú, marcador o sólo "
    "desplazamiento sin acción, el score DEBE ser <= 0.25. peak_frame es el "
    "fotograma 1-4 con la mejor acción visible; outcome debe ser kill, victory, "
    "death, menu, scoreboard, idle o neutral. Responde SOLO JSON: "
    '{{"score": <0.0-1.0>, "reason": "<motivo muy breve>", '
    '"peak_frame": <1-4>, "outcome": "<resultado>", '
    '"title": "<título en español, máx 60 caracteres>", '
    '"hook": "<frase de apertura, máx 40 caracteres>", '
    '"hashtags": ["#uno", "#dos", "#tres"]}}.'
)


def _extract_frames(
    src: str,
    start: float,
    end: float,
    n: int = FRAMES_PER_CLIP,
    width: int = FRAME_WIDTH,
) -> list[bytes]:
    """Saca n fotogramas JPG (en memoria) repartidos dentro de [start, end]."""
    dur = max(0.1, end - start)
    frames: list[bytes] = []
    for i in range(1, n + 1):
        t = start + dur * i / (n + 1)
        cmd = [
            settings.ffmpeg_path,
            "-ss",
            f"{max(0.0, t):.3f}",
            "-i",
            src,
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-2",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout:
            frames.append(proc.stdout)
    return frames


def _parse_copy(text: str) -> dict:
    """Extrae title/hook/hashtags del JSON de vision. {} si no hay o es basura."""
    match = _JSON_RE.search(text or "")
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    out: dict = {}
    title = str(data.get("title") or "").strip()
    hook = str(data.get("hook") or "").strip()
    if title:
        out["title"] = title[:120]
    if hook:
        out["hook"] = hook[:120]
    tags = data.get("hashtags")
    if isinstance(tags, list):
        clean = [("#" + str(t).lstrip("#").strip()) for t in tags if str(t).lstrip("#").strip()]
        if clean:
            out["hashtags"] = clean[:8]
    return out


def _prompt_for(title: str) -> str:
    """Sin titulo -> se le pide tambien el copy; con titulo -> solo puntuar."""
    if (title or "").strip():
        return _PROMPT.format(title=title)
    return _PROMPT_COPY


def _parse_score(text: str) -> tuple[float, str]:
    match = _JSON_RE.search(text or "")
    if not match:
        return 0.5, ""
    try:
        data = json.loads(match.group(0))
        score = float(data.get("score", 0.5))
        return max(0.0, min(1.0, score)), str(data.get("reason", "")).strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.5, ""


def _parse_visual_response(text: str) -> tuple[float, str, dict]:
    """Parsea una respuesta visual valida; salida vacia/basura es un fallo real."""
    match = _JSON_RE.search(text or "")
    if not match:
        raise ValueError("El proveedor de vision no devolvio JSON.")
    try:
        data = json.loads(match.group(0))
        if "score" not in data:
            raise ValueError("La respuesta visual no incluye score.")
        score = float(data["score"])
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("La respuesta del proveedor de vision no es valida.") from exc
    outcome = str(data.get("outcome") or "").strip().lower()
    if outcome in {"death", "menu", "scoreboard", "idle"}:
        score = min(score, 0.25)
    reason = str(data.get("reason", "")).strip()
    if outcome:
        reason = (
            f"{reason}; resultado visual: {outcome}" if reason else f"resultado visual: {outcome}"
        )
    try:
        peak_frame = int(data.get("peak_frame"))
    except (TypeError, ValueError):
        peak_frame = 0
    if 1 <= peak_frame <= FRAMES_PER_CLIP:
        marker = f"pico visual frame {peak_frame}/{FRAMES_PER_CLIP}"
        reason = f"{reason}; {marker}" if reason else marker
    return (
        max(0.0, min(1.0, score)),
        reason,
        _parse_copy(match.group(0)),
    )


def _score_claude(model: str, frames: list[bytes], title: str) -> tuple[float, str]:
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    content: list[dict] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(f).decode(),
            },
        }
        for f in frames
    ]
    content.append({"type": "text", "text": _prompt_for(title)})
    resp = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return _parse_visual_response(text)


def _score_claude_code(model: str, frames: list[bytes], title: str) -> tuple[float, str]:
    """Scoring visual con el CLI de Claude Code (suscripcion, sin API key).

    El CLI no acepta imagenes por stdin: hay que escribirlas a disco y
    referenciarlas con @ruta en el prompt (verificado contra el CLI real).
    """
    import tempfile
    from pathlib import Path

    from app.providers.cli_subscription import (
        _CliError,
        _extract_claude_result,
        resolve_provider_cli,
    )

    exe = resolve_provider_cli("claude_code")
    if exe is None:
        raise _CliError("El CLI de Claude Code no esta instalado o no esta en el PATH.")

    with tempfile.TemporaryDirectory() as tmp:
        refs: list[str] = []
        for idx, data in enumerate(frames):
            path = Path(tmp) / f"frame{idx}.jpg"
            path.write_bytes(data)
            refs.append(f"@{path.name}")
        prompt = _prompt_for(title) + "\n\nFotogramas: " + " ".join(refs)
        cmd = [exe, "-p", prompt, "--model", model, "--output-format", "json"]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=180,
            cwd=tmp,  # cwd = carpeta de los frames, para que @frame0.jpg resuelva
        )
        if proc.returncode != 0:
            raise _CliError(
                f"Claude Code CLI fallo ({proc.returncode}): {(proc.stderr or '')[:200]}"
            )
        text = _extract_claude_result(proc.stdout)
    return _parse_visual_response(text)


def _score_codex(model: str, frames: list[bytes], title: str) -> tuple[float, str]:
    """Scoring visual con el CLI de Codex (suscripcion ChatGPT, sin API key).

    Codex acepta imagenes de forma nativa con `-i`. Dos detalles verificados
    contra el CLI real: `-i` es variadico, asi que el prompt tiene que ir ANTES
    (si no, se traga el prompt como si fuera otro archivo); y hay que cerrarle
    stdin o se queda esperando entrada indefinidamente.
    """
    import tempfile
    from pathlib import Path

    from app.providers.cli_subscription import _CliError, resolve_provider_cli

    exe = resolve_provider_cli("codex")
    if exe is None:
        raise _CliError("El CLI de Codex no esta instalado o no esta en el PATH.")

    with tempfile.TemporaryDirectory() as tmp:
        paths: list[str] = []
        for idx, data in enumerate(frames):
            path = Path(tmp) / f"frame{idx}.jpg"
            path.write_bytes(data)
            paths.append(str(path.resolve()))
        out_path = Path(tmp) / "codex-vision.txt"
        prompt = _prompt_for(title)
        cmd = [
            exe,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-rules",
            "--model",
            model,
            "-o",
            str(out_path),
            prompt,
            "-i",
            *paths,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,  # medido: ~75s por llamada, muy por encima de Claude Code
            cwd=tmp,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            raise _CliError(f"Codex CLI fallo ({proc.returncode}): {(proc.stderr or '')[:200]}")
        if not out_path.exists():
            raise _CliError("Codex CLI no escribio ninguna respuesta visual.")
        out = out_path.read_text(encoding="utf-8")
    return _parse_visual_response(out)


def _score_openai_compatible(
    client, model: str, frames: list[bytes], title: str
) -> tuple[float, str]:
    """Scoring vía API estilo OpenAI chat (sirve para OpenAI y NVIDIA NIM)."""
    content: list[dict] = [{"type": "text", "text": _prompt_for(title)}]
    for f in frames:
        b64 = base64.b64encode(f).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    resp = client.chat.completions.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": content}],
    )
    raw = resp.choices[0].message.content or ""
    return _parse_visual_response(raw)


def _score_openai(model: str, frames: list[bytes], title: str) -> tuple[float, str]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    return _score_openai_compatible(client, model, frames, title)


def _score_nvidia(model: str, frames: list[bytes], title: str) -> tuple[float, str]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.nvidia_api_key, base_url=settings.nvidia_base_url)
    # ignora el model de texto del job: usa el modelo multimodal configurado de NIM
    from app.providers.nvidia_catalog import resolve_model

    vision_model = resolve_model(getattr(settings, "nvidia_vision_model", ""), vision=True)
    return _score_openai_compatible(client, vision_model, frames, title)


def rerank_by_vision(
    candidates: list[ClipCandidate],
    video_path: str,
    *,
    provider: str,
    model: str,
    progress_callback: Callable[[float, str], None] | None = None,
    event_callback: Callable[[str, str], None] | None = None,
) -> list[ClipCandidate]:
    """Reordena candidatos combinando su score textual con un score visual del modelo.

    No-op seguro si el proveedor no tiene visión, si no hay candidatos, o ante
    cualquier error de FFmpeg/API (devuelve la lista original).
    """
    provider = (provider or "").lower()
    if provider not in VISION_PROVIDERS or not candidates:
        return candidates

    scorer = {
        "claude": _score_claude,
        "openai": _score_openai,
        "nvidia": _score_nvidia,
        "claude_code": _score_claude_code,
        "codex": _score_codex,
    }[provider]
    n_frames, width = frame_limits(provider)
    scored: list[ClipCandidate] = []
    any_scored = False
    total = len(candidates)
    for index, c in enumerate(candidates):
        if progress_callback:
            progress_callback(
                index / total,
                f"Visión IA: procesando candidato {index + 1}/{total}",
            )
        try:
            frames = _extract_frames(video_path, c.start, c.end, n=n_frames, width=width)
            if not frames:
                scored.append(c)
                continue
            try:
                result = scorer(model, frames, c.title)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Re-rank visual falló para '%s' (%s). Se omite.",
                    c.title, sanitize_error_text(str(exc)),
                )
                if provider == "codex" and index == 0:
                    from app.providers.cli_subscription import resolve_provider_cli

                    if resolve_provider_cli("claude_code") is not None:
                        if event_callback:
                            event_callback(
                                "warning",
                                "Visión Codex falló en el primer candidato; se detuvieron "
                                "las llamadas restantes y se usa Claude Code (suscripción) "
                                "como fallback visual.",
                            )
                        return rerank_by_vision(
                            candidates,
                            video_path,
                            provider="claude_code",
                            model=VISION_DEFAULT_MODELS["claude_code"],
                            progress_callback=progress_callback,
                            event_callback=event_callback,
                        )
                scored.append(c)
                # Un error del CLI de suscripción suele ser global (argumentos,
                # autenticación o cuota), no del fotograma. No repetir el mismo
                # fallo caro para todos los candidatos.
                if provider in SUBSCRIPTION_VISION:
                    scored.extend(candidates[index + 1 :])
                    break
                continue
            # Los scorers devuelven (score, reason) o (score, reason, copy). Se
            # aceptan ambas formas para no romper a quien parchee el scorer.
            visual, reason = result[0], result[1]
            copy = result[2] if len(result) > 2 and isinstance(result[2], dict) else {}
            any_scored = True
            combined = round((1 - VISION_WEIGHT) * c.score + VISION_WEIGHT * visual, 3)
            new_reason = c.reason
            if reason:
                suffix = f"visual: {reason}"
                new_reason = f"{c.reason}; {suffix}" if c.reason else suffix
            update = {
                "score": combined,
                "reason": new_reason,
                "score_breakdown": [
                    *c.score_breakdown,
                    ScoreContribution(
                        signal="vision_rerank",
                        stage="vision.rerank_by_vision",
                        score_before=c.score,
                        contribution=round(combined - c.score, 3),
                        score_after=combined,
                        note=reason or "sin motivo del proveedor",
                    ),
                ],
            }
            # El copy solo se aplica si el clip venia MUDO: si el detector de texto
            # ya redacto un titulo, sabe mas que la vision (leyo lo que se dijo).
            if copy and not (c.title or "").strip():
                if copy.get("title"):
                    update["title"] = copy["title"]
                if copy.get("hook"):
                    update["hook"] = copy["hook"]
                if copy.get("hashtags"):
                    update["hashtags"] = copy["hashtags"]
            legacy_candidate = c.model_copy(update=update)
            scored.append(
                _append_structured_shadow(legacy_candidate, frames, provider, model, video_path)
            )
        finally:
            if progress_callback:
                progress_callback(
                    (index + 1) / total,
                    f"Visión IA: candidato {index + 1}/{total} completado",
                )

    if not any_scored:
        return candidates
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored


# --- Seleccion de proveedor/modelo de vision -----------------------------------

VISION_DEFAULT_MODELS = {
    "claude_code": "claude-sonnet-5",
    "codex": "gpt-5.6-terra",
    "claude": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
}

SUBSCRIPTION_VISION = frozenset({"claude_code", "codex"})

# Modelos por proveedor, para que el selector muestre SOLO los del proveedor de
# vision elegido. Antes se ofrecia la union de todos: aparecia claude-sonnet-5
# dos veces (claude_code y claude apuntan al mismo modelo) y salia gpt-4o-mini
# incluso habiendo elegido codex.
VISION_PROVIDER_MODELS = {
    "claude_code": ["claude-sonnet-5", "claude-opus-5"],
    # gpt-5.6-{luna,terra,sol} son las variantes reales del catalogo de Codex,
    # no nombres inventados (ver app/providers/cli_subscription.py).
    "codex": ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"],
    "claude": ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    # Free-tier NIM multimodal real. Solo VIA (grounding), nunca rerank.
    "nvidia": [],  # Discovered through /settings/nvidia-models, never a fixed list.
}

# Lo que se OFRECE en Ajustes: solo suscripcion y free-tier. Las APIs de pago
# siguen soportadas si se fijan a proposito, pero no se listan: no deben poder
# elegirse por descuido, porque son consumos con tarjeta.
VISION_PROVIDER_CHOICES = ("auto", "claude_code", "codex", "nvidia")


def vision_models_for(provider: str) -> list[str]:
    """Modelos ofrecibles para un proveedor de vision concreto."""
    return list(VISION_PROVIDER_MODELS.get((provider or "").lower(), ()))


def resolve_vision(text_provider: str, text_model: str, *, via: bool = False) -> tuple[str, str]:
    """(proveedor, modelo) de vision segun Ajustes.

    `vision_provider="auto"` sigue al proveedor de TEXTO. Eso importa por costo:
    si el usuario eligio una suscripcion (claude_code/codex), la vision no debe
    saltar sola a una API de pago. Una API solo se usa si esta elegida a
    proposito, igual que en la deteccion de texto.

    `via=False` (re-rank de clips): NO acepta nvidia (alucina en scoring).
    `via=True`  (grounding VIA): nvidia sí entra, pero SOLO configurado a
    proposito en Ajustes (vision_provider="nvidia", free-tier): auto nunca lo
    elige, y nunca resuelve una API pagada escondida detras de una suscripcion.
    """
    from app.core.config import settings

    configured = (getattr(settings, "vision_provider", "") or "auto").strip().lower()
    provider = (text_provider or "").lower() if configured == "auto" else configured

    if provider == "nvidia":
        # VIA visual con NIM: solo explícito (via=True y vision_provider=nvidia).
        # El modelo visual NIM es distinto del de texto: los de texto no ven.
        if not via or configured != "nvidia":
            return "", ""
        from app.providers.nvidia_catalog import resolve_model

        model = resolve_model(getattr(settings, "nvidia_vision_model", ""), vision=True)
        return "nvidia", model

    if provider not in VISION_PROVIDERS:
        return "", ""

    model = (getattr(settings, "vision_model", "") or "").strip()
    if not model:
        # Con "auto" y el mismo proveedor, reusar el modelo del job evita pedir
        # uno que la suscripcion no tenga habilitado.
        if configured == "auto" and text_model:
            model = text_model
        else:
            model = VISION_DEFAULT_MODELS.get(provider, "")
    return provider, model
