"""Ingesta del video fuente: archivo local subido o URL (yt-dlp)."""

from __future__ import annotations

import ipaddress
import platform
import re
import shutil
import socket
from collections.abc import Callable
from datetime import UTC, datetime
from os import environ
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import settings
from app.core.download_auth import cookie_mode, validate_cookie_path
from app.core.media import VideoInfo, probe

_BROWSER_COOKIE_RE = re.compile(
    r"(?x)"
    r"(?P<name>[^+:]+)"
    r"(?:\s*\+\s*(?P<keyring>[^:]+))?"
    r"(?:\s*:\s*(?!:)(?P<profile>.+?))?"
    r"(?:\s*::\s*(?P<container>.+))?"
)

_BOT_CHALLENGE_MARKERS = (
    "sign in to confirm",
    "not a bot",
    "use --cookies-from-browser",
    "use --cookies",
)

_AUTH_REQUIRED_MARKERS = _BOT_CHALLENGE_MARKERS + (
    "login required",
    "private video",
    "members-only",
    "authentication required",
    "login_required",
    "login page",
    "log in to access",
    "logged-in",
    "instagram sent an empty media response",
    # X/Twitter devuelve este mismo mensaje generico tanto para tweets sin
    # video como para tweets con video restringido a cuentas logueadas
    # (contenido sensible, cuenta protegida). Sin cookies, el API de invitado
    # de X directamente no expone el media de esos tweets. Se trata como
    # "requiere auth" para que se reintente con las cookies del navegador
    # antes de rendirse: si de verdad no hay video, el reintento simplemente
    # falla igual sin costo extra relevante.
    "no video could be found in this tweet",
)

_DIRECT_MEDIA_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".m3u8"}
_OPENAI_PAGE_HOSTS = {"openai.com", "chatgpt.com"}
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}

# Reintentos acotados SOLO para el 403 puntual de YouTube al pedir video data
# (no metadata, no auth challenge). Confirmado empiricamente 2026-08-14: el
# mismo video publico, mismo cliente/formato, sin cookies, falla con 403
# ~1 de cada 3 intentos tanto con yt-dlp 2026.06.09 como 2026.07.04, y tanto
# con los clientes forzados de Snipra como con los defaults de yt-dlp -- es
# el CDN de Google rechazando esa URL firmada puntual (edge/expiracion), no
# un problema de version, de cliente ni de sesion. Una extraccion nueva (URL
# firmada nueva) resuelve el caso casi siempre; los "retries" internos de
# yt-dlp no ayudan aqui porque reintentan la MISMA URL firmada que YouTube
# ya rechazo, no vuelven a extraer.
_YOUTUBE_403_MAX_ATTEMPTS = 3

# Kick pasa a UUIDv7 en la URL del VOD: kick.com/<canal>/videos/<uuid>. Ese uuid
# es el del LIVESTREAM, no el del Video, y el extractor kick:vod de yt-dlp lo pega
# tal cual en /api/v1/video/<uuid> -> 404 "No query results for model [Video]".
# El puente es el listado del canal, que trae livestream.start_time y video.uuid:
# el timestamp va codificado en los primeros 48 bits del UUIDv7, asi que se empareja
# por fecha de inicio. Las URLs viejas traen un UUIDv4 (el del Video) y no se tocan.
_KICK_VOD_RE = re.compile(
    r"^(?P<prefix>https?://(?:www\.)?kick\.com/(?P<channel>[\w-]+)/videos/)"
    r"(?P<uuid>[\da-f]{8}-(?:[\da-f]{4}-){3}[\da-f]{12})(?P<suffix>[/?#].*)?$",
    re.IGNORECASE,
)
_KICK_CHANNEL_VIDEOS_API = "https://kick.com/api/v2/channels/{channel}/videos"
# start_time y created_at vienen en segundos; el UUIDv7 en milisegundos. Un margen
# corto absorbe el redondeo sin llegar a confundir dos directos distintos.
_KICK_START_TOLERANCE_S = 15.0

_COOKIE_LOAD_MARKERS = (
    "cookies.txt",
    "netscape format",
    "could not copy chrome cookie database",
    "could not copy cookie",
    "failed to load cookies",
    "failed to decrypt",
    "decrypt with dpapi",
    "dpapi",
    "permission denied",
    "could not find",
    "cookies database",
)


class VideoDownloadError(RuntimeError):
    """Error de descarga legible para mostrar en el job."""


def job_dir(job_id: str) -> Path:
    d = settings.work_dir / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_upload(job_id: str, filename: str, fileobj) -> Path:
    """Guarda un archivo subido como source.<ext> dentro del work dir del job."""
    ext = Path(filename).suffix or ".mp4"
    dest = job_dir(job_id) / f"source{ext}"
    with dest.open("wb") as out:
        shutil.copyfileobj(fileobj, out)
    return dest


def _parse_browser_cookie_source(value: str) -> tuple[str, str | None, str | None, str | None]:
    raw = value.strip()
    match = _BROWSER_COOKIE_RE.fullmatch(raw)
    if match is None:
        raise VideoDownloadError(
            "YT_DLP_COOKIES_FROM_BROWSER no es valido. Usa auto, brave, edge, firefox "
            "o un perfil como brave:Default."
        )
    browser, keyring, profile, container = match.group("name", "keyring", "profile", "container")
    from yt_dlp.cookies import SUPPORTED_BROWSERS

    if browser.lower() not in SUPPORTED_BROWSERS:
        raise VideoDownloadError("Selecciona Chrome, Brave, Edge o Firefox en Ajustes.")
    return browser.lower(), profile, keyring.upper() if keyring else None, container


def _installed_browser_cookie_sources() -> list[tuple[str, str | None, str | None, str | None]]:
    # Snipra corre en Windows y macOS (ver Start-Snipra.* / Snipra.app), pero cada
    # uno guarda los perfiles del navegador en una ruta totalmente distinta. Antes
    # esto solo miraba LOCALAPPDATA/APPDATA (Windows): en macOS esas variables no
    # existen, la deteccion siempre devolvia una lista vacia y el reintento con
    # cookies del navegador nunca se activaba, sin importar el sitio.
    sources: list[tuple[str, str | None, str | None, str | None]] = []

    if platform.system() == "Darwin":
        support = Path.home() / "Library/Application Support"
        chromium_profiles = [
            ("brave", "Default", support / "BraveSoftware/Brave-Browser/Default/Cookies"),
            ("edge", "Default", support / "Microsoft Edge/Default/Cookies"),
            ("chrome", "Default", support / "Google/Chrome/Default/Cookies"),
        ]
        firefox_root = support / "Firefox/Profiles"
    elif platform.system() == "Windows":
        # Nunca resolver variables ausentes contra el cwd.
        if not environ.get("LOCALAPPDATA") or not environ.get("APPDATA"):
            return sources
        local = Path(environ["LOCALAPPDATA"])
        roaming = Path(environ["APPDATA"])
        chromium_profiles = [
            (
                "brave",
                "Default",
                local / "BraveSoftware/Brave-Browser/User Data/Default/Network/Cookies",
            ),
            ("edge", "Default", local / "Microsoft/Edge/User Data/Default/Network/Cookies"),
            ("chrome", "Default", local / "Google/Chrome/User Data/Default/Network/Cookies"),
        ]
        firefox_root = roaming / "Mozilla/Firefox/Profiles"
    else:
        config = Path(environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        chromium_profiles = [
            ("brave", "Default", config / "BraveSoftware/Brave-Browser/Default/Cookies"),
            ("edge", "Default", config / "microsoft-edge/Default/Cookies"),
            ("chrome", "Default", config / "google-chrome/Default/Cookies"),
        ]
        firefox_root = Path.home() / ".mozilla/firefox"

    for browser, _profile, cookies_path in chromium_profiles:
        depth = 2 if cookies_path.parent.name == "Network" else 1
        root = cookies_path.parents[depth]
        try:
            profiles = sorted(root.iterdir(), key=lambda p: (p.name != "Default", p.name))
            for candidate in profiles:
                if candidate.name != "Default" and not candidate.name.startswith("Profile "):
                    continue
                if any((candidate / suffix).is_file() for suffix in ("Network/Cookies", "Cookies")):
                    sources.append((browser, candidate.name, None, None))
                if sum(source[0] == browser for source in sources) >= 4:
                    break
        except OSError:
            continue

    try:
        for profile in sorted(firefox_root.iterdir()):
            if (profile / "cookies.sqlite").is_file():
                sources.append(("firefox", str(profile), None, None))
            if sum(source[0] == "firefox" for source in sources) >= 4:
                break
    except OSError:
        pass

    return sources


def _browser_cookie_sources() -> list[tuple[str, str | None, str | None, str | None]]:
    configured = cookie_mode(settings.yt_dlp_cookies_from_browser, settings.yt_dlp_cookies_path)
    if configured in {"none", "cookies_file"}:
        return []
    if configured == "auto":
        return _installed_browser_cookie_sources()
    return [_parse_browser_cookie_source(configured)]


ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]


_QUALITY_HEIGHTS = {
    "720": 720,
    "1080": 1080,
    "1440": 1440,
    "2k": 1440,
    "2160": 2160,
    "4k": 2160,
}


def _format_for_quality(quality: str) -> str:
    """Selector de formato yt-dlp segun la calidad pedida ('auto' = mejor disponible).

    Prueba H.264 (avc1) primero en cada nivel: YouTube y Facebook sirven AV1 cada
    vez mas seguido incluso a resoluciones normales, y aunque es un codec valido
    dentro de MP4, WhatsApp y varios reproductores lo rechazan en silencio (el
    archivo "se ve" correcto -- contenedor MP4, extension correcta -- pero no
    reproduce). Para "Crear clips" esto no cambia nada: el render reencodea a
    libx264 de todas formas. Para "Descargar original" (sin reencode, por diseno)
    el codec de origen ES el codec final, asi que aqui si importa. Si la fuente
    solo tiene AV1/VP9 a esa calidad, cae a esos sin error -- nunca bloquea la
    descarga, solo prefiere H.264 cuando esta disponible.
    """
    q = (quality or "1080").strip().lower()
    if q in {"auto", "best", "max", "maxima"}:
        return "bv*[vcodec^=avc1]+ba/bv*+ba/b"
    h = _QUALITY_HEIGHTS.get(q, 1080)
    # El fallback final SIN tope de altura ("b" a secas) solo debe usarse si de
    # verdad no hay nada mejor: antes de llegar ahi se intenta bv*+ba sin tope
    # (streams separados, cualquier resolucion) para no caer directo al 360p
    # combinado mas viejo que YouTube ofrece cuando el tope <=h no encuentra
    # match entre los formatos combinados legacy.
    return (
        f"bv*[vcodec^=avc1][height<={h}]+ba/bv*[height<={h}]+ba/"
        f"b[height<={h}]/bv*+ba/b"
    )


def requested_quality_height(quality: str) -> int | None:
    """Altura (px) que implica la calidad pedida, o None si es 'auto'/sin techo definido.

    Es SOLO el techo pedido, no una garantia: la fuente puede no ofrecer nada tan alto
    (region, restricciones anti-bot del sitio, etc.) y yt-dlp cae a lo mejor disponible
    por debajo del techo sin que eso sea un error. Se usa para avisar en el log del job
    cuando lo conseguido queda muy por debajo de lo pedido, en vez de quedar en silencio.
    """
    q = (quality or "").strip().lower()
    if q in {"", "auto", "best", "max", "maxima"}:
        return None
    return _QUALITY_HEIGHTS.get(q)


_AUDIO_FORMATS = {"mp3", "wav", "m4a", "opus", "flac"}


class _PrivateYdlLogger:
    """yt-dlp puede imprimir stderr con cookies, URLs firmadas o rutas de perfiles."""

    def debug(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


def is_instagram_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme.lower() in {"http", "https"} and (
        (parsed.hostname or "").lower() in {"instagram.com", "www.instagram.com", "m.instagram.com"}
    )


def normalize_instagram_url(url: str) -> str:
    # Los links compartidos conservan query/fragment; yt-dlp resuelve redirects.
    if not is_instagram_url(url):
        return url
    parsed = urlparse(url)
    if parsed.hostname.lower() == "m.instagram.com":
        netloc = parsed.netloc.lower().replace("m.instagram.com", "www.instagram.com")
        return parsed._replace(netloc=netloc).geturl()
    return url


def _normalize_audio_format(audio_format: str | None) -> str:
    fmt = (audio_format or settings.audio_format or "mp3").strip().lower()
    return fmt if fmt in _AUDIO_FORMATS else "mp3"


def _common_ydl_options(
    out_tmpl: str,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict:
    opts = {
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _PrivateYdlLogger(),
        "cachedir": False,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
        # YouTube exige un PO Token para los streams DASH del cliente 'ios' y
        # bloquea varios formatos https del cliente 'android' (SABR) sin el.
        # Sin ellos, yt-dlp se queda solo con el itag 18 (640x360 combinado)
        # y ahi termina la descarga aunque se pida 1080p. tv_embedded y
        # android_vr todavia exponen el DASH completo (hasta 4K) sin PO Token.
        "extractor_args": {"youtube": {"player_client": ["tv_embedded", "android_vr", "web"]}},
        "concurrent_fragment_downloads": 4,
    }
    if progress_callback is not None:
        opts["progress_hooks"] = [_download_progress_hook(progress_callback, cancel_check)]
    # Con FFMPEG_PATH absoluto (app empaquetada: Contents/Helpers) yt-dlp debe saber dónde está
    # ffmpeg para fusionar vídeo y audio; no lo busca en el PATH del launcher.
    ffmpeg_path = Path(settings.ffmpeg_path)
    if ffmpeg_path.is_absolute() and ffmpeg_path.parent.is_dir():
        opts["ffmpeg_location"] = str(ffmpeg_path.parent)
    return opts


def _base_ydl_options(
    out_tmpl: str,
    progress_callback: ProgressCallback | None = None,
    quality: str | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict:
    opts = _common_ydl_options(out_tmpl, progress_callback, cancel_check)
    opts["format"] = _format_for_quality(quality or settings.download_quality)
    opts["merge_output_format"] = "mp4"
    return opts


def _base_audio_ydl_options(
    out_tmpl: str,
    progress_callback: ProgressCallback | None = None,
    audio_format: str | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict:
    """Solo audio (bestaudio), convertido con ffmpeg al formato pedido. No baja video."""
    opts = _common_ydl_options(out_tmpl, progress_callback, cancel_check)
    opts["format"] = "bestaudio/best"
    opts["postprocessors"] = [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": _normalize_audio_format(audio_format),
            "preferredquality": "192",
        }
    ]
    return opts


def _download_progress_hook(
    progress_callback: ProgressCallback,
    cancel_check: CancelCheck | None = None,
):
    def hook(data: dict) -> None:
        if cancel_check is not None and cancel_check():
            raise VideoDownloadError("Job cancelado durante la descarga")
        status = data.get("status")
        if status == "finished":
            progress_callback(1.0, "Descarga completada")
            return
        if status != "downloading":
            return
        downloaded = data.get("downloaded_bytes") or 0
        total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
        if total:
            pct = max(0.0, min(1.0, downloaded / total))
            progress_callback(pct, f"Descargando {round(pct * 100)}%")

    return hook


def _ydl_options(out_tmpl: str, progress_callback: ProgressCallback | None = None) -> dict:
    variants = _ydl_option_variants(out_tmpl, progress_callback)
    return variants[1][1] if len(variants) > 1 else variants[0][1]


def _ydl_option_variants(
    out_tmpl: str,
    progress_callback: ProgressCallback | None = None,
    quality: str | None = None,
    cancel_check: CancelCheck | None = None,
    audio_format: str | None = None,
    source_url: str = "",
) -> list[tuple[str, dict]]:
    # audio_format viene con un valor (aunque sea "") solo cuando el caller pide
    # audio-only; None significa modo video normal. Evita que un "" quality de
    # video se confunda con "pedir audio".
    def build_base() -> dict:
        if audio_format is not None:
            opts = _base_audio_ydl_options(out_tmpl, progress_callback, audio_format, cancel_check)
        else:
            opts = _base_ydl_options(out_tmpl, progress_callback, quality, cancel_check)
        if is_instagram_url(source_url):
            opts.update(noplaylist=True, playlistend=1)
        return opts

    # Otros providers conservan el intento público inicial. Instagram prioriza
    # sesiones locales según la configuración explícita y termina con acceso público.
    variants: list[tuple[str, dict]] = [("sin cookies", build_base())]
    mode = cookie_mode(settings.yt_dlp_cookies_from_browser, settings.yt_dlp_cookies_path)
    if mode == "none":
        return variants

    for browser_source in _browser_cookie_sources():
        opts = build_base()
        opts["cookiesfrombrowser"] = browser_source
        variants.append((browser_source[0], opts))

    if settings.yt_dlp_cookies_path.strip() or mode == "cookies_file":
        opts = build_base()
        opts["cookiefile"] = settings.yt_dlp_cookies_path.strip()
        variants.append(("cookies.txt", opts))

    if is_instagram_url(source_url):
        variants = variants[1:] + variants[:1]

    return variants


def _is_cookie_load_error(error: Exception) -> bool:
    lower = str(error).lower()
    return any(marker in lower for marker in _COOKIE_LOAD_MARKERS)


def _needs_cookie_retry(error: Exception) -> bool:
    lower = str(error).lower()
    return any(marker in lower for marker in _AUTH_REQUIRED_MARKERS)


def _is_youtube_video_data_403(url: str, error: Exception) -> bool:
    """403 de YouTube pidiendo el video data, sin evidencia de que haga falta login.

    Distinto de _needs_cookie_retry: ahi hay texto explicito de challenge/bot/login.
    Aqui yt-dlp encontro el formato y la URL firmada, pero el CDN de Google
    rechazo esa URL puntual. Cookies no arreglan esto -- una extraccion nueva si.
    """
    if (urlparse(url).hostname or "").lower() not in _YOUTUBE_HOSTS:
        return False
    lower = str(error).lower()
    return "403" in lower and "forbidden" in lower and not _needs_cookie_retry(error)


def _attempt_download(opts: dict, url: str) -> Exception | None:
    """Descarga con reintentos acotados solo para el 403 transitorio de YouTube.

    Cada intento vuelve a extraer info y a firmar formatos desde cero: la URL
    firmada que YouTube rechazo con 403 casi siempre es cosa de ese intento
    puntual, no del formato ni del cliente elegido (ver _YOUTUBE_403_MAX_ATTEMPTS).
    Cualquier otro error corta en el primer intento -- reintentar ahi solo
    tardaria mas en llegar al mismo resultado.
    """
    import yt_dlp

    last_exc: Exception | None = None
    for attempt in range(_YOUTUBE_403_MAX_ATTEMPTS):
        try:
            if "cookiefile" in opts:
                try:
                    if not validate_cookie_path(opts["cookiefile"]):
                        raise ValueError("Configura la ruta de cookies.txt en Ajustes.")
                except ValueError as exc:
                    return VideoDownloadError(str(exc))
            with yt_dlp.YoutubeDL(opts) as ydl:
                # yt-dlp reescribe cookiefile al cerrar. Solo consumimos la sesión;
                # nunca persistimos nuevas cookies ni modificamos el archivo del usuario.
                ydl.save_cookies = lambda: None
                result = ydl.download([url])
                if result:
                    return VideoDownloadError("El downloader no completó la descarga.")
            return None
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            is_last_attempt = attempt + 1 >= _YOUTUBE_403_MAX_ATTEMPTS
            if is_last_attempt or not _is_youtube_video_data_403(url, exc):
                return exc
    return last_exc


def _is_openai_page_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host not in _OPENAI_PAGE_HOSTS:
        return False
    return Path(parsed.path).suffix.lower() not in _DIRECT_MEDIA_SUFFIXES


def _uuid7_timestamp_ms(value: str) -> int | None:
    """Milisegundos codificados en un UUIDv7, o None si no es v7."""
    hexed = value.replace("-", "").lower()
    if len(hexed) != 32 or hexed[12] != "7":
        return None
    try:
        return int(hexed[:12], 16)
    except ValueError:
        return None


def _kick_parse_time(value: object) -> float | None:
    """Convierte '2026-08-06 02:36:19' o ISO-8601 de la API de Kick a epoch UTC."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _kick_fetch_channel_videos(channel: str) -> list[dict]:
    """Listado de VODs del canal. Requiere impersonacion TLS: Cloudflare filtra el resto."""
    url = _KICK_CHANNEL_VIDEOS_API.format(channel=channel)
    try:
        from curl_cffi import requests as curl_requests
    except ImportError as exc:
        raise VideoDownloadError(
            "Falta curl-cffi para hablar con Kick (Cloudflare rechaza las peticiones sin "
            "impersonacion TLS). Instala las dependencias del backend con 'uv sync'."
        ) from exc

    try:
        response = curl_requests.get(url, impersonate="chrome", timeout=30)
    except Exception as exc:  # noqa: BLE001 - curl_cffi levanta su propia jerarquia
        raise VideoDownloadError(f"No se pudo consultar los VODs de Kick: {exc}") from exc
    if response.status_code == 404:
        raise VideoDownloadError(f"Kick no encontro el canal '{channel}'. Revisa la URL.")
    if response.status_code != 200:
        raise VideoDownloadError(
            f"Kick respondio {response.status_code} al listar los VODs de '{channel}'."
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise VideoDownloadError(
            "Kick devolvio una respuesta ilegible al listar los VODs."
        ) from exc
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _kick_video_uuid(item: dict) -> str | None:
    video = item.get("video")
    if not isinstance(video, dict):
        return None
    uuid_value = video.get("uuid")
    return uuid_value if isinstance(uuid_value, str) and uuid_value else None


def resolve_kick_vod_url(url: str) -> str:
    """Traduce una URL de VOD de Kick con UUID de livestream al UUID de video que usa yt-dlp.

    Devuelve la URL intacta si no es un VOD de Kick o si ya trae el UUID del video.
    """
    match = _KICK_VOD_RE.match(url.strip())
    if match is None:
        return url
    livestream_uuid = match.group("uuid")
    started_ms = _uuid7_timestamp_ms(livestream_uuid)
    if started_ms is None:
        return url

    channel = match.group("channel")
    started_at = started_ms / 1000
    best_uuid: str | None = None
    best_delta = _KICK_START_TOLERANCE_S
    for item in _kick_fetch_channel_videos(channel):
        video_uuid = _kick_video_uuid(item)
        if not video_uuid:
            continue
        for key in ("start_time", "created_at"):
            moment = _kick_parse_time(item.get(key))
            if moment is None:
                continue
            delta = abs(moment - started_at)
            if delta <= best_delta:
                best_delta = delta
                best_uuid = video_uuid

    if best_uuid is None:
        raise VideoDownloadError(
            f"Kick ya no expone ese VOD de '{channel}'. Kick borra los VODs pasados unos dias "
            "y solo deja los recientes del canal; si el directo sigue publicado, abre la pestana "
            "Videos del canal y pega el enlace del VOD que aparezca ahi."
        )
    return f"{match.group('prefix')}{best_uuid}{match.group('suffix') or ''}"


def validate_download_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise VideoDownloadError("Usa una URL publica http(s) de una pagina o archivo de video.")
    if _is_openai_page_url(url):
        raise VideoDownloadError(
            "Las paginas de ChatGPT/OpenAI no exponen un archivo de video para yt-dlp. "
            "Descarga el video desde ChatGPT y subelo a Snipra, o pega una URL publica directa "
            "del archivo (.mp4, .mov o .webm)."
        )
    if settings.allow_private_downloads:
        return

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise VideoDownloadError("La URL contiene un puerto invalido.") from exc

    try:
        literal = ipaddress.ip_address(parsed.hostname)
        addresses = {literal}
    except ValueError:
        try:
            resolved = socket.getaddrinfo(
                parsed.hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except OSError as exc:
            raise VideoDownloadError("No se pudo resolver el servidor de la URL.") from exc
        try:
            addresses = {ipaddress.ip_address(item[4][0]) for item in resolved}
        except ValueError as exc:
            raise VideoDownloadError(
                "El servidor de la URL devolvio una direccion invalida."
            ) from exc

    if not addresses or any(not address.is_global for address in addresses):
        raise VideoDownloadError(
            "La URL apunta a la red local o privada. "
            "Usa una URL publica o activa ALLOW_PRIVATE_DOWNLOADS solo para una LAN de confianza."
        )


def _download_error_message(
    error: Exception, attempts: list[str] | None = None, source_url: str = "https://youtube.com"
) -> str:
    lower = str(error).lower()
    # No incorporar labels arbitrarios/perfiles a mensajes enviados al job.
    safe_attempts = [label for label in (attempts or []) if label in {
        "sin cookies", "cookies.txt", "chrome", "brave", "edge", "firefox"
    }]
    attempted = f" Intentos: {' | '.join(safe_attempts)}." if safe_attempts else ""
    if is_instagram_url(source_url) and _needs_cookie_retry(error):
        return (
            "Instagram requiere una sesión iniciada para acceder a este contenido. "
            "Selecciona en Ajustes un navegador donde hayas iniciado sesión o utiliza "
            "cookies.txt. La sesión debe tener acceso al contenido."
            f"{attempted}"
        )
    if _is_cookie_load_error(error):
        return (
            "No se pudo leer la sesión o el archivo cookies.txt configurado. "
            "El navegador puede bloquear o cifrar sus cookies; Snipra no intenta saltarse ese "
            "bloqueo. Selecciona otro navegador con sesión iniciada o registra un cookies.txt "
            "local existente y legible en Ajustes."
            f"{attempted}"
        )
    if "unsupported url" in lower:
        return (
            "Esta URL no expone un video descargable para Snipra. Usa una URL publica compatible "
            "o pega el enlace directo al archivo de video (.mp4, .mov o .webm)."
            f"{attempted}"
        )
    if _is_youtube_video_data_403(source_url, error):
        return (
            "YouTube rechazo el stream de video (HTTP 403). Snipra ya reintento la "
            "descarga automaticamente: esto casi siempre es un rechazo puntual del "
            "servidor de YouTube al firmar esa URL, no una sesion faltante. Si "
            "sigue fallando tras varios intentos puede ser una restriccion real del "
            "video (privado, con edad restringida o solo para miembros) -- ahi si "
            f"hacen falta cookies (YT_DLP_COOKIES_PATH).{attempted}"
        )
    if any(marker in lower for marker in _BOT_CHALLENGE_MARKERS):
        host = (urlparse(source_url).hostname or "").lower()
        is_youtube = host in _YOUTUBE_HOSTS
        if not is_youtube:
            return (
                "El sitio requiere autenticación o una sesión válida para descargar este "
                "contenido. Configura cookies de navegador para yt-dlp o proporciona un archivo "
                f"cookies.txt.{attempted}"
            )
        if settings.yt_dlp_cookies_path or settings.yt_dlp_cookies_from_browser:
            return (
                "YouTube rechazo la descarga incluso usando cookies. Verifica que tu sesion del "
                "navegador tenga acceso al video, que YouTube abra sin captcha y cierra Brave/Edge "
                f"antes de procesar si Windows mantiene bloqueadas las cookies.{attempted}"
            )
        return (
            "YouTube pidio confirmar que no eres un bot. Configura en backend/.env "
            "YT_DLP_COOKIES_FROM_BROWSER=auto, o usa YT_DLP_COOKIES_PATH con un "
            f"archivo cookies.txt exportado desde tu navegador.{attempted}"
        )
    if "no video could be found in this tweet" in lower:
        return (
            "X/Twitter no expuso ningun video en ese tweet incluso probando con las "
            "cookies de tu navegador. O el tweet de verdad no tiene video (es imagen, "
            "GIF, o el video esta en un tweet citado/retweeteado, no en el que "
            "pegaste), o esta restringido a cuentas logueadas (contenido sensible, "
            "cuenta protegida) y la sesion de tu navegador no tiene acceso a el. "
            f"Verifica el tweet abierto y logueado en el navegador que Snipra usa "
            f"para cookies.{attempted}"
        )
    if _needs_cookie_retry(error):
        return (
            "El sitio requiere una sesión iniciada con acceso al contenido. "
            f"Selecciona un navegador o cookies.txt en Ajustes.{attempted}"
        )
    return (
        "yt-dlp no pudo descargar el video. Comprueba el enlace, la conexión y que el "
        f"contenido sea accesible para tu sesión.{attempted}"
    )


def _source_candidates(job_id: str, *, audio: bool = False) -> list[Path]:
    pattern = "source_audio.*" if audio else "source.*"
    return sorted(
        path for path in job_dir(job_id).glob(pattern)
        if path.is_file() and path.suffix.lower() not in {
            ".part", ".ytdl", ".json", ".jpg", ".jpeg", ".png", ".vtt", ".srt"
        }
    )


def download_url(
    job_id: str,
    url: str,
    progress_callback: ProgressCallback | None = None,
    quality: str | None = None,
    cancel_check: CancelCheck | None = None,
) -> Path:
    """Descarga un video desde una URL usando yt-dlp."""
    validate_download_url(url)
    url = normalize_instagram_url(resolve_kick_vod_url(url))
    out_tmpl = str(job_dir(job_id) / "source.%(ext)s")
    attempts: list[str] = []
    last_error: Exception | None = None
    cookie_error: Exception | None = None
    for label, opts in _ydl_option_variants(
        out_tmpl, progress_callback, quality, cancel_check, source_url=url
    ):
        if cancel_check is not None and cancel_check():
            raise VideoDownloadError("Job cancelado antes de descargar")
        exc = _attempt_download(opts, url)
        if exc is None:
            last_error = None
            break
        last_error = exc
        attempts.append(label)
        if cookie_error is None and _is_cookie_load_error(exc):
            cookie_error = exc
        if not (_is_cookie_load_error(exc) or _needs_cookie_retry(exc)):
            break
    if last_error is not None:
        cause = cookie_error or last_error
        raise VideoDownloadError(_download_error_message(cause, attempts, url)) from None

    candidates = _source_candidates(job_id)
    if not candidates:
        raise VideoDownloadError("yt-dlp no produjo ningún archivo de video completo.")
    return candidates[0]


def download_audio_url(
    job_id: str,
    url: str,
    progress_callback: ProgressCallback | None = None,
    audio_format: str | None = None,
    cancel_check: CancelCheck | None = None,
) -> Path:
    """Descarga SOLO el audio (sin video) y lo convierte al formato pedido con ffmpeg."""
    validate_download_url(url)
    url = normalize_instagram_url(resolve_kick_vod_url(url))
    out_tmpl = str(job_dir(job_id) / "source_audio.%(ext)s")
    attempts: list[str] = []
    last_error: Exception | None = None
    cookie_error: Exception | None = None
    for label, opts in _ydl_option_variants(
        out_tmpl, progress_callback, None, cancel_check, audio_format=audio_format, source_url=url
    ):
        if cancel_check is not None and cancel_check():
            raise VideoDownloadError("Job cancelado antes de descargar")
        exc = _attempt_download(opts, url)
        if exc is None:
            last_error = None
            break
        last_error = exc
        attempts.append(label)
        if cookie_error is None and _is_cookie_load_error(exc):
            cookie_error = exc
        if not (_is_cookie_load_error(exc) or _needs_cookie_retry(exc)):
            break
    if last_error is not None:
        cause = cookie_error or last_error
        raise VideoDownloadError(_download_error_message(cause, attempts, url)) from None

    candidates = _source_candidates(job_id, audio=True)
    if not candidates:
        raise VideoDownloadError("yt-dlp no produjo ningún archivo de audio completo.")
    return candidates[0]


def find_audio_source(job_id: str) -> Path:
    candidates = _source_candidates(job_id, audio=True)
    if not candidates:
        raise FileNotFoundError(f"No hay audio para el job {job_id}")
    return candidates[0]


def find_source(job_id: str) -> Path:
    candidates = _source_candidates(job_id)
    if not candidates:
        raise FileNotFoundError(f"No hay source para el job {job_id}")
    return candidates[0]


def probe_source(job_id: str) -> VideoInfo:
    return probe(find_source(job_id))


def probe_audio_source(job_id: str) -> VideoInfo:
    return probe(find_audio_source(job_id))
