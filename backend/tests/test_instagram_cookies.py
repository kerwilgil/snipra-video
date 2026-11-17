"""Sesiones sintéticas: nunca usa un perfil real ni cookies de una cuenta."""

import socket
import traceback

import pytest

from app.core import config, runtime_settings
from app.core.download_auth import cookie_mode, validate_cookie_path
from app.pipeline import ingest


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_from_browser", "auto")
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_path", "")
    monkeypatch.setattr(ingest, "job_dir", lambda _: tmp_path)
    monkeypatch.setattr(ingest, "_installed_browser_cookie_sources", lambda: [])
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))
    ])


@pytest.mark.parametrize("path", [
    "reel/abc", "reels/abc", "p/abc", "tv/abc", "stories/test/12345",
    "share/reel/abc", "share/p/abc", "test/reel/abc",
])
@pytest.mark.parametrize("host", ["instagram.com", "www.instagram.com", "m.instagram.com"])
def test_instagram_detection(host, path):
    url = f"https://{host}/{path}?igsh=test#fragment"
    assert ingest.is_instagram_url(url)
    normalized = ingest.normalize_instagram_url(url)
    assert "?igsh=test#fragment" in normalized


@pytest.mark.parametrize("url", [
    "https://youtu.be/test", "https://youtube.com/watch?v=test", "https://tiktok.com/@test/video/1",
    "https://x.com/test/status/1", "https://twitter.com/test/status/1", "https://facebook.com/reel/1",
    "https://instagram.com.evil.test/reel/1", "https://evil.test/instagram.com/reel/1",
    "ftp://instagram.com/p/1",
])
def test_other_providers_unchanged(url):
    assert not ingest.is_instagram_url(url)
    assert ingest.normalize_instagram_url(url) == url
    assert ingest._ydl_option_variants("source.%(ext)s", source_url=url)[0][0] == "sin cookies"


@pytest.mark.parametrize("browser", ["chrome", "brave", "edge", "firefox"])
def test_explicit_browser_options(monkeypatch, browser):
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_from_browser", browser)
    variants = ingest._ydl_option_variants("source.%(ext)s", source_url="https://instagram.com/reel/abc")
    assert [label for label, _ in variants] == [browser, "sin cookies"]
    assert variants[0][1]["cookiesfrombrowser"] == (browser, None, None, None)
    assert "cookiefile" not in variants[0][1]


def test_auto_order_file_fallback(monkeypatch, tmp_path):
    file = tmp_path / "cookies.txt"
    file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_path", str(file))
    monkeypatch.setattr(ingest, "_installed_browser_cookie_sources", lambda: [
        (name, "Default", None, None) for name in ("brave", "edge", "chrome", "firefox")
    ])
    variants = ingest._ydl_option_variants("source.%(ext)s", source_url="https://instagram.com/stories/test/1")
    assert [label for label, _ in variants] == ["brave", "edge", "chrome", "firefox", "cookies.txt", "sin cookies"]
    assert variants[-2][1]["cookiefile"] == str(file)
    assert all(not ("cookiefile" in opts and "cookiesfrombrowser" in opts) for _, opts in variants)


def test_none_disables_all_cookies(monkeypatch):
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_from_browser", "none")
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_path", "C:/should-not-be-read.cookies.txt")
    assert [label for label, _ in ingest._ydl_option_variants("out")] == ["sin cookies"]


def test_file_mode_excludes_browser(monkeypatch, tmp_path):
    file = tmp_path / "cookies.txt"
    file.touch()
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_from_browser", "cookies_file")
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_path", str(file))
    assert ingest._ydl_options("out")["cookiefile"] == str(file)
    assert cookie_mode("", str(file)) == "cookies_file"
    assert cookie_mode("brave:Default") == "brave:Default"


def test_browser_discovery_windows_profiles(monkeypatch, tmp_path):
    # Usa la implementación real sin acceder a perfiles del equipo.
    monkeypatch.setattr(ingest.platform, "system", lambda: "Windows")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    for suffix in [
        "local/BraveSoftware/Brave-Browser/User Data/Profile 1/Network/Cookies",
        "local/Microsoft/Edge/User Data/Default/Network/Cookies",
        "local/Google/Chrome/User Data/Default/Cookies",
        "roaming/Mozilla/Firefox/Profiles/test/cookies.sqlite",
    ]:
        file = tmp_path / suffix
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    sources = _REAL_DISCOVERY()
    assert [s[0] for s in sources] == ["brave", "edge", "chrome", "firefox"]
    assert sources[0][1] == "Profile 1"


_REAL_DISCOVERY = ingest._installed_browser_cookie_sources


def test_browser_not_installed_and_env_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest.platform, "system", lambda: "Windows")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "absent"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "absent"))
    assert _REAL_DISCOVERY() == []
    monkeypatch.delenv("LOCALAPPDATA")
    assert _REAL_DISCOVERY() == []


@pytest.mark.parametrize("path", ["relative.cookies.txt", "https://test/cookies.txt", "\\\\server\\cookies.txt", "C:\\cookies.txt\nsecret"])
def test_unsafe_file_path_rejected(path):
    with pytest.raises(ValueError):
        validate_cookie_path(path)


def test_missing_file_falls_back_without_creating_it(monkeypatch, tmp_path):
    file = tmp_path / "missing.cookies.txt"
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_from_browser", "cookies_file")
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_path", str(file))
    import yt_dlp
    class FakeYdl:
        def __init__(self, opts):
            assert "cookiefile" not in opts
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def download(self, urls):
            (tmp_path / "source.mp4").touch()
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYdl)
    assert ingest.download_url("job", "https://instagram.com/p/test").name == "source.mp4"
    assert not file.exists()


@pytest.mark.parametrize("message", [
    "You need to log in to access this content", "This content is unreachable. Login required",
    "Instagram sent an empty media response", "The webpage request was redirected to the login page",
    "login_required", "private video", "authentication required",
])
def test_auth_classification_and_story_message(message):
    error = RuntimeError(message)
    assert ingest._needs_cookie_retry(error)
    result = ingest._download_error_message(error, source_url="https://instagram.com/stories/test/1")
    assert "Instagram requiere una sesión" in result
    assert "no existe" not in result


@pytest.mark.parametrize("audio", [False, True])
def test_auto_bounded_fallback_and_safe_traceback(monkeypatch, tmp_path, capsys, audio):
    import yt_dlp
    monkeypatch.setattr(ingest, "_installed_browser_cookie_sources", lambda: [
        ("chrome", "Profile 1", None, None), ("firefox", "C:/secret-profile", None, None)
    ])
    calls = []
    class FakeYdl:
        def __init__(self, opts):
            self.opts = opts
            calls.append(opts)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.save_cookies()
        def save_cookies(self):
            raise AssertionError("Nunca guardar cookies")
        def download(self, urls):
            self.opts["logger"].error("Cookie: sessionid=synthetic-secret C:/secret-profile")
            if len(calls) == 1:
                raise RuntimeError("Failed to decrypt with DPAPI C:/secret-profile")
            raise RuntimeError("login required Cookie: sessionid=synthetic-secret")
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYdl)
    with pytest.raises(ingest.VideoDownloadError) as exc:
        if audio:
            ingest.download_audio_url("job", "https://instagram.com/stories/test/1", audio_format="mp3")
        else:
            ingest.download_url("job", "https://instagram.com/stories/test/1")
    assert len(calls) == 3
    text = "".join(traceback.format_exception(exc.value)) + capsys.readouterr().err
    assert "synthetic-secret" not in text and "secret-profile" not in text


def test_general_errors_and_attempt_labels_are_allowlisted():
    result = ingest._download_error_message(
        RuntimeError("HTTP 404 Cookie: sessionid=synthetic-secret /Users/private/profile"),
        ["chrome:C:/private/profile", "firefox"], "https://instagram.com/p/abc"
    )
    assert "synthetic-secret" not in result and "private" not in result
    assert "firefox" in result and "requiere" not in result


@pytest.mark.parametrize("path", ["reel/abc", "p/abc", "stories/test/123"])
def test_download_media_with_accessible_session(monkeypatch, tmp_path, path):
    import yt_dlp
    monkeypatch.setattr(config.settings, "yt_dlp_cookies_from_browser", "chrome")
    class FakeYdl:
        def __init__(self, opts):
            assert opts["cookiesfrombrowser"] == ("chrome", None, None, None)
            assert opts["noplaylist"] is True and opts["playlistend"] == 1
            assert "height<=1080" in opts["format"]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def download(self, urls):
            assert urls == [f"https://instagram.com/{path}"]
            (tmp_path / "source.mp4").touch()
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYdl)
    assert ingest.download_url("job", f"https://instagram.com/{path}", quality="1080").name == "source.mp4"


def test_cookie_file_read_only_with_real_ytdlp(monkeypatch, tmp_path):
    import yt_dlp
    file = tmp_path / "cookies.txt"
    initial = b"# Netscape HTTP Cookie File\n"
    file.write_bytes(initial)
    def fake_download(ydl, urls):
        assert len(ydl.cookiejar) == 0
    monkeypatch.setattr(yt_dlp.YoutubeDL, "download", fake_download)
    assert ingest._attempt_download({"cookiefile": str(file), "logger": ingest._PrivateYdlLogger()}, "https://instagram.com/p/a") is None
    assert file.read_bytes() == initial


@pytest.mark.parametrize("extension", ["flv", "webm"])
def test_incomplete_file_is_never_a_download_result(monkeypatch, tmp_path, extension):
    monkeypatch.setattr(ingest, "_attempt_download", lambda *a: None)
    (tmp_path / "source.mp4.part").touch()
    with pytest.raises(ingest.VideoDownloadError, match="archivo de video completo"):
        ingest.download_url("job", "https://instagram.com/p/a")
    with pytest.raises(FileNotFoundError):
        ingest.find_source("job")
    # Conserva contenedores de otros providers (por ejemplo FLV).
    filename = f"source.{extension}"
    (tmp_path / filename).touch()
    assert ingest.download_url("job", "https://example.com/a").name == filename
    assert ingest.find_source("job").name == filename


def test_settings_store_only_reference_and_reject_missing_file(monkeypatch, tmp_path):
    from fastapi import HTTPException
    from app.api import settings as settings_api
    file = tmp_path / "cookies.txt"
    file.write_text("# synthetic marker not a session", encoding="utf-8")
    persisted = tmp_path / "app_settings.json"
    monkeypatch.setattr(runtime_settings, "RUNTIME_SETTINGS_PATH", persisted)
    monkeypatch.setattr(settings_api, "get_settings", lambda: {
        "yt_dlp_cookies_from_browser": config.settings.yt_dlp_cookies_from_browser,
        "yt_dlp_cookies_path": config.settings.yt_dlp_cookies_path,
    })
    result = settings_api.update_settings(settings_api.SettingsUpdate(
        yt_dlp_cookies_from_browser="COOKIES_FILE", yt_dlp_cookies_path=str(file)))
    assert result["yt_dlp_cookies_from_browser"] == "cookies_file"
    assert "synthetic marker" not in str(result) + persisted.read_text()
    with pytest.raises(HTTPException) as exc:
        settings_api.update_settings(settings_api.SettingsUpdate(yt_dlp_cookies_path=str(tmp_path / "absent.cookies.txt")))
    assert exc.value.status_code == 400
