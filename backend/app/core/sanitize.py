"""Sanitización de texto externo antes de exponerlo en errores/logs visibles.

Una excepción de un SDK o el stderr de un CLI puede contener la API key que
se le pasó (mensajes de auth, eco de headers, etc.). Nunca devolver eso al
cliente ni a la UI. Esta función quita los valores de clave CONFIGURADOS en
esta instalación y los sustituye por [REDACTED].

No oculta solo patrones genéricos: reemplaza exactamente las credenciales
efectivamente cargadas, que es lo que puede filtrarse de verdad.
"""

from __future__ import annotations

REDACTED = "[REDACTED]"

# Campos de Settings cuyo valor es un secreto.
_SECRET_FIELDS = (
    "anthropic_api_key",
    "openai_api_key",
    "nvidia_api_key",
)


def _configured_secrets() -> list[str]:
    """Valores de clave efectivamente configurados (no vacíos). Sin imprimirlos."""
    try:
        from app.core.config import settings
    except Exception:  # noqa: BLE001 - sanear no debe fallar
        return []
    secrets: list[str] = []
    for field in _SECRET_FIELDS:
        value = getattr(settings, field, "") or ""
        value = str(value).strip()
        # Umbral generoso para no redactar "1" o placeholders triviales.
        if len(value) >= 4:
            secrets.append(value)
    return secrets


def sanitize_error_text(text: str) -> str:
    """Elimina secretos configurados del texto; conserva el contexto restante."""
    if not text:
        return text or ""
    result = text
    for secret in _configured_secrets():
        if secret in result:
            result = result.replace(secret, REDACTED)
    return result


def sanitize_traceback(exc: BaseException) -> str:
    """Traceback completo de `exc` como string, con secretos redactados.

    Para sustituir `traceback.print_exc()` y `logger.exception(...)` cuando la
    excepción proviene de un SDK/CLI externo: el log conserva tipo, stack y
    contexto, pero nunca las API keys configuradas.
    """
    import traceback as _tb

    raw = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
    return sanitize_error_text(raw)
