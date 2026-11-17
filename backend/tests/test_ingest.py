"""Tests de ingesta URL sin descargar contenido real."""

import socket

import pytest

from app.core import config
from app.pipeline import ingest


@pytest.fixture(autouse=True)
def _resolve_test_urls_as_public(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))],
    )


def test_ydl_options_include_browser_cookies(monkeypatch):
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_from_browser", "brave:Default")
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_path", "")

    opts = ingest._ydl_options("source.%(ext)s")

    assert opts["cookiesfrombrowser"] == ("brave", "Default", None, None)
    assert "cookiefile" not in opts


def test_ydl_options_include_cookie_file(monkeypatch, tmp_path):
    cookie_file = tmp_path / "cookies.txt"
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_from_browser", "")
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_path", str(cookie_file))

    opts = ingest._ydl_options("source.%(ext)s")

    assert opts["cookiefile"] == str(cookie_file)
    assert "cookiesfrombrowser" not in opts


def test_bot_challenge_message_points_to_cookies(monkeypatch):
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_from_browser", "")
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_path", "")

    msg = ingest._download_error_message(RuntimeError("Sign in to confirm you're not a bot"))

    assert "YT_DLP_COOKIES_FROM_BROWSER=auto" in msg


def test_auth_challenge_messages_are_classified_by_source_host(monkeypatch):
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_from_browser", "")
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_path", "")
    error = RuntimeError("Sign in to confirm you're not a bot")
    instagram = ingest._download_error_message(error, source_url="https://www.instagram.com/reel/abc")
    youtube = ingest._download_error_message(error, source_url="https://youtu.be/abc")
    generic = ingest._download_error_message(error, source_url="https://example.com/video")
    assert "Instagram requiere" in instagram and "YouTube" not in instagram
    assert "YouTube pidio" in youtube
    assert "El sitio requiere autenticación" in generic and "YouTube" not in generic


def test_normal_download_error_is_not_reclassified_as_authentication():
    message = ingest._download_error_message(RuntimeError("HTTP Error 404: Not Found"), source_url="https://www.instagram.com/p/abc")
    assert message.startswith("yt-dlp no pudo descargar")


def test_is_youtube_video_data_403_matches_youtube_hosts_only():
    err = RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
    assert ingest._is_youtube_video_data_403("https://www.youtube.com/watch?v=abc", err)
    assert ingest._is_youtube_video_data_403("https://youtu.be/abc", err)
    assert not ingest._is_youtube_video_data_403("https://example.com/video", err)


def test_is_youtube_video_data_403_ignores_auth_challenge_text():
    # un 403 con evidencia de bot-challenge es otro caso (ahi cookies si ayudan);
    # lo maneja _needs_cookie_retry, no el reintento transitorio.
    err = RuntimeError("Sign in to confirm you're not a bot (403)")
    assert not ingest._is_youtube_video_data_403("https://www.youtube.com/watch?v=abc", err)


def test_youtube_403_message_is_specific_and_not_generic_cookie_advice(monkeypatch):
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_from_browser", "")
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_path", "")
    error = RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden")

    msg = ingest._download_error_message(error, source_url="https://www.youtube.com/watch?v=abc")

    assert "YouTube rechazo el stream" in msg
    assert "Sign in" not in msg


def test_non_youtube_403_is_not_reclassified():
    error = RuntimeError("HTTP Error 403: Forbidden")
    message = ingest._download_error_message(error, source_url="https://example.com/video")
    assert message.startswith("yt-dlp no pudo descargar")


def test_format_for_quality_maps_resolution():
    assert "height<=720" in ingest._format_for_quality("720")
    assert "height<=1080" in ingest._format_for_quality("1080")
    assert "height<=1440" in ingest._format_for_quality("1440")
    assert "height<=1440" in ingest._format_for_quality("2k")
    # auto = mejor disponible (sin tope de altura); prueba H.264 primero
    assert ingest._format_for_quality("auto") == "bv*[vcodec^=avc1]+ba/bv*+ba/b"


def test_ydl_options_receive_job_quality_limit():
    opts = ingest._ydl_option_variants("source.%(ext)s", quality="2160")[0][1]

    assert "height<=2160" in opts["format"]


def test_normalize_audio_format_accepts_known_values_and_falls_back():
    assert ingest._normalize_audio_format("wav") == "wav"
    assert ingest._normalize_audio_format("MP3") == "mp3"
    # formato desconocido o vacio -> mp3 (nunca revienta el job por un valor raro)
    assert ingest._normalize_audio_format("ogg-vorbis-legacy") == "mp3"
    assert ingest._normalize_audio_format(None) == "mp3"


def test_base_audio_ydl_options_extract_audio_no_video():
    opts = ingest._base_audio_ydl_options("source_audio.%(ext)s", audio_format="flac")

    assert opts["format"] == "bestaudio/best"
    assert "merge_output_format" not in opts  # no arma un video, no aplica
    pp = opts["postprocessors"][0]
    assert pp["key"] == "FFmpegExtractAudio"
    assert pp["preferredcodec"] == "flac"


def test_ydl_option_variants_audio_mode_uses_audio_options():
    # audio_format=None (explicito, no el default posicional) activa el modo audio
    variants = ingest._ydl_option_variants("source_audio.%(ext)s", audio_format="opus")
    label, opts = variants[0]
    assert label == "sin cookies"
    assert opts["format"] == "bestaudio/best"
    assert opts["postprocessors"][0]["preferredcodec"] == "opus"


def test_ydl_option_variants_default_stays_video_mode():
    # sin pasar audio_format, sigue siendo el flujo de video de siempre (regresion)
    variants = ingest._ydl_option_variants("source.%(ext)s", quality="1080")
    _, opts = variants[0]
    assert opts["format"] != "bestaudio/best"
    assert "postprocessors" not in opts


def test_format_for_quality_has_uncapped_fallback_before_legacy_combo():
    # Regresion: sin este escalon intermedio (streams separados sin tope de
    # altura), el selector caia directo al combo legacy 'b' (itag 18, 640x360)
    # en cuanto el combo con tope de altura no encontraba match -- aunque
    # hubiera streams separados de mejor calidad disponibles.
    fmt = ingest._format_for_quality("1080")
    parts = fmt.split("/")
    assert parts[0] == "bv*[vcodec^=avc1][height<=1080]+ba"
    assert "bv*[height<=1080]+ba" in parts  # cualquier codec, mismo tope, si no hay H.264
    assert "bv*+ba" in parts  # escalon sin tope, antes del ultimo recurso 'b'
    assert parts.index("bv*+ba") < len(parts) - 1  # nunca es el ultimo recurso
    assert parts[-1] == "b"  # el ultimo recurso sigue siendo el combo legacy


def test_base_ydl_options_use_po_token_free_clients():
    # tv_embedded/android_vr exponen el DASH completo (hasta 4K) sin pedir PO
    # Token; 'ios'/'android' solos quedan limitados por YouTube a itag 18
    # (640x360) sin el. Ver commit 6f8e91f.
    opts = ingest._base_ydl_options("source.%(ext)s")
    clients = opts["extractor_args"]["youtube"]["player_client"]
    assert "tv_embedded" in clients
    assert "android_vr" in clients
    assert "ios" not in clients


def test_requested_quality_height_maps_known_values():
    assert ingest.requested_quality_height("1080") == 1080
    assert ingest.requested_quality_height("2160") == 2160
    assert ingest.requested_quality_height("4k") == 2160
    assert ingest.requested_quality_height("2k") == 1440


def test_requested_quality_height_none_for_auto_or_unset():
    # 'auto' o vacio = sin techo definido: no hay base para comparar y avisar
    assert ingest.requested_quality_height("auto") is None
    assert ingest.requested_quality_height("") is None
    assert ingest.requested_quality_height(None) is None
    assert ingest.requested_quality_height("best") is None


def test_dpapi_error_classified_as_cookie_load():
    # 'Failed to decrypt with DPAPI' debe tratarse como fallo de cookies (no fatal),
    # para que download_url siga al intento "sin cookies" en videos publicos.
    assert ingest._is_cookie_load_error(RuntimeError("ERROR: Failed to decrypt with DPAPI"))
    assert ingest._is_cookie_load_error(RuntimeError("Could not copy cookie database"))


def test_cookie_lock_error_is_preserved_over_later_auth_error(monkeypatch, tmp_path):
    import yt_dlp

    monkeypatch.setattr(ingest, "job_dir", lambda job_id: tmp_path)
    monkeypatch.setattr(
        ingest, "_ydl_option_variants", lambda *a, **k: [("brave:Default", {}), ("firefox", {})]
    )
    calls = {"value": 0}

    class FakeYDL:
        def __init__(self, opts):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def download(self, urls):
            calls["value"] += 1
            if calls["value"] == 1:
                raise RuntimeError("Could not copy Chrome cookie database")
            raise RuntimeError("Instagram sent an empty media response")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    with pytest.raises(ingest.VideoDownloadError, match="no intenta saltarse ese bloqueo"):
        ingest.download_url("job1", "https://www.instagram.com/reel/abc")


def test_openai_page_url_explains_how_to_import_video():
    with pytest.raises(ingest.VideoDownloadError, match="ChatGPT/OpenAI"):
        ingest.validate_download_url("https://openai.com/chatgpt-work/?video=1208185418")


def test_download_url_rejects_loopback_and_private_dns(monkeypatch):
    monkeypatch.setattr(config.settings, "allow_private_downloads", False)

    with pytest.raises(ingest.VideoDownloadError, match="red local o privada"):
        ingest.validate_download_url("http://127.0.0.1/video.mp4")

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.25", 443))
        ],
    )
    with pytest.raises(ingest.VideoDownloadError, match="red local o privada"):
        ingest.validate_download_url("https://media.example/video.mp4")


def test_download_url_accepts_public_dns_and_private_opt_in(monkeypatch):
    monkeypatch.setattr(config.settings, "allow_private_downloads", False)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))],
    )
    ingest.validate_download_url("https://media.example/video.mp4")

    monkeypatch.setattr(config.settings, "allow_private_downloads", True)
    ingest.validate_download_url("http://127.0.0.1/video.mp4")


def test_ydl_variants_try_public_download_before_cookies(monkeypatch):
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_path", "C:/cookies.txt")
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_from_browser", "")

    labels = [label for label, _ in ingest._ydl_option_variants("source.%(ext)s")]

    assert labels == ["sin cookies", "cookies.txt"]


def test_download_url_retries_cookies_only_after_auth_challenge(monkeypatch, tmp_path):
    import yt_dlp

    monkeypatch.setattr(ingest, "job_dir", lambda job_id: tmp_path)
    monkeypatch.setattr(
        ingest,
        "_ydl_option_variants",
        lambda *a, **k: [("sin cookies", {}), ("edge", {})],
    )
    calls = {"n": 0}

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def download(self, urls):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Sign in to confirm you're not a bot")
            (tmp_path / "source.mp4").write_text("x", encoding="utf-8")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    result = ingest.download_url("job1", "https://www.youtube.com/watch?v=abc")
    assert result.name == "source.mp4"
    assert calls["n"] == 2


def test_download_url_falls_back_to_no_cookies(monkeypatch, tmp_path):
    import yt_dlp

    monkeypatch.setattr(ingest, "job_dir", lambda job_id: tmp_path)
    monkeypatch.setattr(
        ingest, "_ydl_option_variants", lambda *a, **k: [("edge", {}), ("sin cookies", {})]
    )
    calls = {"n": 0}

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def download(self, urls):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("ERROR: Failed to decrypt with DPAPI")
            (tmp_path / "source.mp4").write_text("x", encoding="utf-8")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    result = ingest.download_url("job1", "https://www.youtube.com/watch?v=abc")
    assert result.name == "source.mp4"
    assert calls["n"] == 2  # edge fallo DPAPI -> continuo -> sin cookies exito


def test_download_url_retries_transient_youtube_403_before_switching_variant(monkeypatch, tmp_path):
    import yt_dlp

    monkeypatch.setattr(ingest, "job_dir", lambda job_id: tmp_path)
    monkeypatch.setattr(ingest, "_ydl_option_variants", lambda *a, **k: [("sin cookies", {})])
    calls = {"n": 0}

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def download(self, urls):
            calls["n"] += 1
            if calls["n"] < ingest._YOUTUBE_403_MAX_ATTEMPTS:
                raise RuntimeError(
                    "ERROR: unable to download video data: HTTP Error 403: Forbidden"
                )
            (tmp_path / "source.mp4").write_text("x", encoding="utf-8")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    result = ingest.download_url("job1", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert result.name == "source.mp4"
    # se agoto justo al ultimo intento permitido, sin cambiar de variante (una sola en el mock)
    assert calls["n"] == ingest._YOUTUBE_403_MAX_ATTEMPTS


def test_download_url_gives_up_after_max_youtube_403_retries(monkeypatch, tmp_path):
    import yt_dlp

    monkeypatch.setattr(ingest, "job_dir", lambda job_id: tmp_path)
    monkeypatch.setattr(ingest, "_ydl_option_variants", lambda *a, **k: [("sin cookies", {})])
    calls = {"n": 0}

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def download(self, urls):
            calls["n"] += 1
            raise RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    with pytest.raises(ingest.VideoDownloadError) as excinfo:
        ingest.download_url("job1", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert calls["n"] == ingest._YOUTUBE_403_MAX_ATTEMPTS
    assert "YouTube rechazo el stream" in str(excinfo.value)


def test_download_url_does_not_retry_non_youtube_403(monkeypatch, tmp_path):
    import yt_dlp

    monkeypatch.setattr(ingest, "job_dir", lambda job_id: tmp_path)
    monkeypatch.setattr(ingest, "_ydl_option_variants", lambda *a, **k: [("sin cookies", {})])
    calls = {"n": 0}

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def download(self, urls):
            calls["n"] += 1
            raise RuntimeError("HTTP Error 403: Forbidden")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    with pytest.raises(ingest.VideoDownloadError):
        ingest.download_url("job1", "https://example.com/video.mp4")
    assert calls["n"] == 1  # host no es YouTube -> sin reintento transitorio


def test_auto_browser_sources_prefers_installed_brave_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest.platform, "system", lambda: "Windows")

    local = tmp_path / "local"
    brave_cookies = local / "BraveSoftware/Brave-Browser/User Data/Default/Network/Cookies"
    brave_cookies.parent.mkdir(parents=True)
    brave_cookies.write_text("cookies", encoding="utf-8")

    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_from_browser", "auto")
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_path", "")

    assert ingest._browser_cookie_sources() == [("brave", "Default", None, None)]


def test_auto_browser_sources_prefers_installed_brave_macos(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ingest.Path, "home", lambda: tmp_path)

    support = tmp_path / "Library/Application Support"
    brave_cookies = support / "BraveSoftware/Brave-Browser/Default/Cookies"
    brave_cookies.parent.mkdir(parents=True)
    brave_cookies.write_text("cookies", encoding="utf-8")

    monkeypatch.setattr(config.settings, "yt_dlp_cookies_from_browser", "auto")
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_path", "")

    assert ingest._browser_cookie_sources() == [("brave", "Default", None, None)]


# --- Kick: UUID de livestream en la URL -> UUID de video que espera yt-dlp -------------

_KICK_LIVESTREAM_URL = "https://kick.com/kerwil/videos/019fd4ed-c0b8-7aba-9c06-5f6a1477fcf9"
_KICK_VIDEO_UUID = "1513b8e8-59c8-40f5-9ebc-b31eb6fdf51b"


def _kick_listing(start_time="2026-08-06 02:36:19", created_at="2026-08-06 02:36:24"):
    return [
        {
            "id": 120882332,
            "start_time": start_time,
            "created_at": created_at,
            "video": {"uuid": _KICK_VIDEO_UUID},
        }
    ]


def test_resolve_kick_vod_url_swaps_livestream_uuid(monkeypatch):
    monkeypatch.setattr(ingest, "_kick_fetch_channel_videos", lambda channel: _kick_listing())

    resolved = ingest.resolve_kick_vod_url(_KICK_LIVESTREAM_URL)

    assert resolved == f"https://kick.com/kerwil/videos/{_KICK_VIDEO_UUID}"


def test_resolve_kick_vod_url_picks_matching_stream(monkeypatch):
    otro = {
        "start_time": "2026-08-05 21:00:00",
        "created_at": "2026-08-05 21:00:05",
        "video": {"uuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"},
    }
    monkeypatch.setattr(
        ingest, "_kick_fetch_channel_videos", lambda channel: [otro, *_kick_listing()]
    )

    resolved = ingest.resolve_kick_vod_url(_KICK_LIVESTREAM_URL)

    assert resolved.endswith(_KICK_VIDEO_UUID)


def test_resolve_kick_vod_url_keeps_legacy_video_uuid(monkeypatch):
    def _explode(channel):
        raise AssertionError("no debe consultar la API para un UUID de video")

    monkeypatch.setattr(ingest, "_kick_fetch_channel_videos", _explode)
    url = "https://kick.com/xqc/videos/5c697a87-afce-4256-b01f-3c8fe71ef5cb"

    assert ingest.resolve_kick_vod_url(url) == url


def test_resolve_kick_vod_url_ignores_other_sites(monkeypatch):
    def _explode(channel):
        raise AssertionError("no debe consultar la API para URLs que no son de Kick")

    monkeypatch.setattr(ingest, "_kick_fetch_channel_videos", _explode)
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    assert ingest.resolve_kick_vod_url(url) == url


def test_resolve_kick_vod_url_explains_expired_vod(monkeypatch):
    monkeypatch.setattr(
        ingest,
        "_kick_fetch_channel_videos",
        lambda channel: _kick_listing("2026-07-01 10:00:00", "2026-07-01 10:00:05"),
    )

    with pytest.raises(ingest.VideoDownloadError) as excinfo:
        ingest.resolve_kick_vod_url(_KICK_LIVESTREAM_URL)

    assert "kerwil" in str(excinfo.value)
