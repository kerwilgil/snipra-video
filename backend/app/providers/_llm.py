"""Flujo común de detección por LLM: prompt → completar → parsear (con reintentos).

Para videos largos parte la transcripción en bloques (chunks) y analiza cada uno, así el
modelo evalúa TODO el video y no solo el principio (evita truncar la transcripción).
"""

import math
from collections.abc import Callable

from app.providers.base import ClipCandidate, DetectParams
from app.providers.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    chunk_segments,
    parse_candidates,
)

CompleteFn = Callable[[str, str], str]


def run_detection(
    complete: CompleteFn,
    segments: list[dict],
    params: DetectParams,
    retries: int = 1,
) -> list[ClipCandidate]:
    if not segments:
        return []

    chunks = chunk_segments(segments)
    if len(chunks) <= 1:
        raw = _detect_chunk(complete, segments, params, retries)
        min_dur = params.min_seconds * 0.5
        valid = [c for c in raw if c.duration >= min_dur]
        if not valid and raw:
            raise RuntimeError(
                f"LLM devolvió {len(raw)} clips pero todos son demasiado cortos "
                f"(< {min_dur:.0f}s). El fallback naive se encargará."
            )
        return valid

    # videos largos: pide candidatos por bloque y junta el pool; detect_clips ordena y recorta
    per_chunk = max(2, math.ceil(params.num_clips / len(chunks)) + 1)
    sub_params = params.model_copy(update={"num_clips": per_chunk})

    pool: list[ClipCandidate] = []
    failures = 0
    for chunk in chunks:
        try:
            pool.extend(_detect_chunk(complete, chunk, sub_params, retries=0))
        except Exception:  # noqa: BLE001
            failures += 1

    if not pool and failures:
        raise RuntimeError(f"Detección por bloques falló en los {failures} bloques")
    # descarta clips que el LLM devolvió con duración claramente por debajo del mínimo
    min_dur = params.min_seconds * 0.5
    valid = [c for c in pool if c.duration >= min_dur]
    if not valid and pool:
        raise RuntimeError(
            f"LLM devolvió {len(pool)} clips pero todos son demasiado cortos "
            f"(< {min_dur:.0f}s). El fallback naive se encargará."
        )
    return valid


def _detect_chunk(
    complete: CompleteFn,
    segments: list[dict],
    params: DetectParams,
    retries: int,
) -> list[ClipCandidate]:
    user = build_user_prompt(segments, params)
    last_err: Exception | None = None
    for _attempt in range(retries + 1):
        try:
            raw = complete(SYSTEM_PROMPT, user)
            candidates = parse_candidates(raw)
            if candidates:
                return candidates
            last_err = ValueError("El modelo no devolvió clips válidos")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        # refuerza la instrucción en el reintento
        user += "\n\nIMPORTANTE: responde SOLO con el array JSON, sin texto ni markdown."

    raise RuntimeError(f"Detección por LLM falló tras {retries + 1} intentos: {last_err}")
