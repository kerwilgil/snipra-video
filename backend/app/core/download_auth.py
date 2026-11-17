"""Configuración de sesiones locales; nunca contiene valores de cookies."""

from pathlib import Path

COOKIE_MODES = {"auto", "none", "chrome", "brave", "edge", "firefox", "cookies_file"}


def cookie_mode(browser: str, path: str = "") -> str:
    raw = browser.strip()
    value = raw.lower()
    if value == "detect":
        return "auto"
    # Compatibilidad con instalaciones anteriores que solo configuraban un archivo.
    if value in COOKIE_MODES:
        return value
    return raw or ("cookies_file" if path.strip() else "auto")


def validate_cookie_path(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    path = Path(value)
    if (
        any(ord(char) < 32 for char in value)
        or value.startswith(("\\\\", "//"))
        or not path.is_absolute()
        or path.suffix.lower() != ".txt"
        or ":" in value[len(path.drive):]
    ):
        raise ValueError("Utiliza una ruta absoluta local a un archivo cookies.txt.")
    try:
        if not path.is_file():
            raise ValueError("El archivo cookies.txt no existe o no es un archivo local.")
        with path.open("rb"):
            pass  # Verifica acceso, sin leer ni mostrar el contenido.
    except OSError:
        raise ValueError("No se puede leer el archivo cookies.txt configurado.") from None
    return value
