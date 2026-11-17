"""Tests de canales trackeados: alta vía API, idempotencia, contrato de extracción y salud."""

from __future__ import annotations

import socket
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.db import engine
from app.main import app
from app.models import Job, TrackedChannel, TrackedChannelVideo
from app.pipeline import channel_watch, ingest
from app.pipeline.channel_watch import ExtractionResult, ExtractionStatus


@pytest.fixture(autouse=True)
def _resolve_test_urls_as_public(monkeypatch):
    monkeypatch.setattr(
        ingest.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))],
    )


def _fake_result(
    video_ids: list[str],
    status: ExtractionStatus = ExtractionStatus.SUCCESS,
    channel_title: str = "Test Channel",
    canonical_url: str = "https://www.youtube.com/channel/UCtest",
) -> ExtractionResult:
    entries = [
        {"id": vid, "url": f"https://www.youtube.com/watch?v={vid}", "title": f"Video {vid}"}
        for vid in video_ids
    ]
    return ExtractionResult(
        status=status if entries else ExtractionStatus.EMPTY,
        entries=entries,
        channel_title=channel_title,
        canonical_url=canonical_url,
    )


# ==============================================================================
# Matrix A: Successful channel with entries
# ==============================================================================
def test_matrix_a_successful_channel_with_entries(monkeypatch):
    with TestClient(app) as c:
        created = c.post("/api/channels", json={"url": "https://www.youtube.com/@channel_a"})
        channel_id = created.json()["id"]

    monkeypatch.setattr(
        channel_watch,
        "_extract_flat_entries",
        lambda url: _fake_result(["vidA1", "vidA2"]),
    )

    result = channel_watch.check_channel(channel_id)
    assert result["checked"] is True
    assert result["success"] is True
    assert result["videos_seen"] == 2
    assert result["error_code"] is None

    with Session(engine) as session:
        ch = session.get(TrackedChannel, channel_id)
        assert ch is not None
        assert ch.last_check_status == "healthy"
        assert ch.last_check_error is None
        assert ch.consecutive_failures == 0
        assert ch.videos_seen_last_check == 2
        assert ch.new_videos_last_check == 2
        assert ch.last_success_at is not None
        assert ch.canonical_url == "https://www.youtube.com/channel/UCtest"


# ==============================================================================
# Matrix B: Successful channel with zero new entries
# ==============================================================================
def test_matrix_b_successful_channel_zero_new_entries(monkeypatch):
    with TestClient(app) as c:
        created = c.post("/api/channels", json={"url": "https://www.youtube.com/@channel_b"})
        channel_id = created.json()["id"]

    # 1er check con vidB1
    monkeypatch.setattr(
        channel_watch,
        "_extract_flat_entries",
        lambda url: _fake_result(["vidB1"]),
    )
    first = channel_watch.check_channel(channel_id)
    assert first["success"] is True

    # 2do check con la misma lista (0 videos nuevos)
    second = channel_watch.check_channel(channel_id)
    assert second["checked"] is True
    assert second["success"] is True
    assert second["first_check"] is False
    assert second["videos_seen"] == 1
    assert second["new_videos"] == 0
    assert second["jobs_created"] == 0

    with Session(engine) as session:
        ch = session.get(TrackedChannel, channel_id)
        assert ch.last_check_status == "healthy"
        assert ch.consecutive_failures == 0
        assert ch.new_videos_last_check == 0


# ==============================================================================
# Matrix C: Extraction timeout/error
# ==============================================================================
def test_matrix_c_extraction_timeout_or_error(monkeypatch):
    with TestClient(app) as c:
        created = c.post("/api/channels", json={"url": "https://www.youtube.com/@channel_c"})
        channel_id = created.json()["id"]

    # 1er check exitoso para fijar last_success_at
    monkeypatch.setattr(
        channel_watch,
        "_extract_flat_entries",
        lambda url: _fake_result(["vidC1"]),
    )
    first = channel_watch.check_channel(channel_id)
    assert first["success"] is True

    with Session(engine) as session:
        initial_success_at = session.get(TrackedChannel, channel_id).last_success_at
        assert initial_success_at is not None

    # 2do check falla por timeout
    monkeypatch.setattr(
        channel_watch,
        "_extract_flat_entries",
        lambda url: ExtractionResult(
            status=ExtractionStatus.ERROR,
            error_code="TIMEOUT",
            error_message="Tiempo de espera agotado al consultar el canal.",
        ),
    )
    second = channel_watch.check_channel(channel_id)
    assert second["checked"] is True
    assert second["success"] is False
    assert second["error_code"] == "TIMEOUT"
    assert "Tiempo de espera agotado" in second["error_message"]

    with Session(engine) as session:
        ch = session.get(TrackedChannel, channel_id)
        assert ch.last_check_status == "error"
        assert ch.consecutive_failures == 1
        assert "Tiempo de espera agotado" in ch.last_check_error
        # Preserva el last_success_at anterior
        assert ch.last_success_at == initial_success_at


# ==============================================================================
# Matrix D: Malformed / unsupported URL or SSRF rejection
# ==============================================================================
def test_matrix_d_ssrf_rejection_marks_error(monkeypatch):
    with TestClient(app) as c:
        created = c.post("/api/channels", json={"url": "https://www.youtube.com/@channel_d"})
        channel_id = created.json()["id"]

    def _reject(url: str) -> None:
        raise ingest.VideoDownloadError("la URL apunta a la red local o privada")

    monkeypatch.setattr(channel_watch.ingest, "validate_download_url", _reject)

    result = channel_watch.check_channel(channel_id)
    assert result["checked"] is True
    assert result["success"] is False
    assert result["error_code"] == "SSRF_REJECTED"

    with Session(engine) as session:
        ch = session.get(TrackedChannel, channel_id)
        assert ch.last_check_status == "error"
        assert ch.consecutive_failures == 1


# ==============================================================================
# Matrix E: First check registers seen without jobs
# ==============================================================================
def test_matrix_e_first_check_registers_without_jobs(monkeypatch):
    with TestClient(app) as c:
        created = c.post("/api/channels", json={"url": "https://www.youtube.com/@channel_e"})
        channel_id = created.json()["id"]

    monkeypatch.setattr(
        channel_watch,
        "_extract_flat_entries",
        lambda url: _fake_result(["vidE1", "vidE2", "vidE3"]),
    )

    result = channel_watch.check_channel(channel_id)
    assert result["checked"] is True
    assert result["first_check"] is True
    assert result["videos_seen"] == 3
    assert result["new_videos"] == 3
    assert result["jobs_created"] == 0

    with Session(engine) as session:
        videos = session.exec(
            select(TrackedChannelVideo).where(TrackedChannelVideo.channel_id == channel_id)
        ).all()
        assert len(videos) == 3
        assert all(v.job_id is None for v in videos)


# ==============================================================================
# Matrix F: Repeated check, no duplicate jobs
# ==============================================================================
def test_matrix_f_repeated_check_no_duplicate_jobs(monkeypatch):
    with TestClient(app) as c:
        created = c.post("/api/channels", json={"url": "https://www.youtube.com/@channel_f"})
        channel_id = created.json()["id"]

    # 1er check: registra vidF1
    monkeypatch.setattr(
        channel_watch,
        "_extract_flat_entries",
        lambda url: _fake_result(["vidF1"]),
    )
    channel_watch.check_channel(channel_id)

    # 2do check: aparece vidF2 -> crea 1 job
    monkeypatch.setattr(
        channel_watch,
        "_extract_flat_entries",
        lambda url: _fake_result(["vidF1", "vidF2"]),
    )
    second = channel_watch.check_channel(channel_id)
    assert second["first_check"] is False
    assert second["new_videos"] == 1
    assert second["jobs_created"] == 1

    # 3er check: misma lista -> 0 jobs
    third = channel_watch.check_channel(channel_id)
    assert third["new_videos"] == 0
    assert third["jobs_created"] == 0

    with Session(engine) as session:
        videos = session.exec(
            select(TrackedChannelVideo).where(TrackedChannelVideo.channel_id == channel_id)
        ).all()
        assert len(videos) == 2
        jobs = [v.job_id for v in videos if v.job_id is not None]
        assert len(jobs) == 1


# ==============================================================================
# Matrix G: One bad video does not break channel check
# ==============================================================================
def test_matrix_g_one_bad_video_does_not_break_check(monkeypatch):
    with TestClient(app) as c:
        created = c.post("/api/channels", json={"url": "https://www.youtube.com/@channel_g"})
        channel_id = created.json()["id"]

    # 1er check
    monkeypatch.setattr(
        channel_watch,
        "_extract_flat_entries",
        lambda url: _fake_result(["vidG1"]),
    )
    channel_watch.check_channel(channel_id)

    # 2do check con 2 videos nuevos: vidG2 falla al crear job, vidG3 succeed
    monkeypatch.setattr(
        channel_watch,
        "_extract_flat_entries",
        lambda url: _fake_result(["vidG1", "vidG2", "vidG3"]),
    )

    original_create_job = channel_watch._create_job_for_channel_video

    def _flaky_create(session, channel, entry):
        if entry["id"] == "vidG2":
            raise RuntimeError("Database error on job creation")
        return original_create_job(session, channel, entry)

    monkeypatch.setattr(channel_watch, "_create_job_for_channel_video", _flaky_create)

    result = channel_watch.check_channel(channel_id)
    assert result["checked"] is True
    assert result["success"] is True
    assert result["new_videos"] == 2
    assert result["jobs_created"] == 1  # Solo vidG3 se encoló, pero el check no explotó

    with Session(engine) as session:
        videos = session.exec(
            select(TrackedChannelVideo).where(TrackedChannelVideo.channel_id == channel_id)
        ).all()
        assert len(videos) == 3
        vid2 = next(v for v in videos if v.video_id == "vidG2")
        vid3 = next(v for v in videos if v.video_id == "vidG3")
        assert vid2.job_id is None
        assert vid3.job_id is not None


# ==============================================================================
# Matrix H: One bad channel does not break polling of others
# ==============================================================================
def test_matrix_h_one_bad_channel_does_not_break_others(monkeypatch):
    with TestClient(app) as c:
        c1 = c.post("/api/channels", json={"url": "https://www.youtube.com/@bad_ch"})
        c2 = c.post("/api/channels", json={"url": "https://www.youtube.com/@good_ch"})
        id1 = c1.json()["id"]
        id2 = c2.json()["id"]

    # Mock _extract_flat_entries: explota en id1, funciona en id2
    def _extract_mock(url: str):
        if "bad_ch" in url:
            raise RuntimeError("Fatal extraction crash")
        return _fake_result(["goodVid1"])

    monkeypatch.setattr(channel_watch, "_extract_flat_entries", _extract_mock)

    checked_count = channel_watch.poll_due_channels()
    assert checked_count >= 2

    with Session(engine) as session:
        session.get(TrackedChannel, id1)
        ch2 = session.get(TrackedChannel, id2)
        assert ch2.last_check_status == "healthy"
        assert ch2.videos_seen_last_check == 1


# ==============================================================================
# Matrix I: Persisted health survives reload / serialization
# ==============================================================================
def test_matrix_i_persisted_health_survives_reload():
    with TestClient(app) as c:
        created = c.post("/api/channels", json={"url": "https://www.youtube.com/@channel_i"})
        channel_id = created.json()["id"]

    now = datetime.now(UTC)
    with Session(engine) as session:
        ch = session.get(TrackedChannel, channel_id)
        ch.last_check_status = "healthy"
        ch.last_check_error = None
        ch.last_success_at = now
        ch.consecutive_failures = 0
        ch.videos_seen_last_check = 8
        ch.new_videos_last_check = 2
        ch.canonical_url = "https://www.youtube.com/channel/UCcanonical"
        ch.channel_title = "Canonical Title"
        session.add(ch)
        session.commit()

    # Consultar vía API GET /api/channels
    with TestClient(app) as c:
        resp = c.get("/api/channels")
        assert resp.status_code == 200
        found = next(item for item in resp.json() if item["id"] == channel_id)
        assert found["last_check_status"] == "healthy"
        assert found["videos_seen_last_check"] == 8
        assert found["new_videos_last_check"] == 2
        assert found["canonical_url"] == "https://www.youtube.com/channel/UCcanonical"
        assert found["channel_title"] == "Canonical Title"
        assert found["last_success_at"] is not None


# ==============================================================================
# Matrix J: Manual check returns diagnostic payload
# ==============================================================================
def test_matrix_j_manual_check_diagnostic_payload(monkeypatch):
    with TestClient(app) as c:
        created = c.post("/api/channels", json={"url": "https://www.youtube.com/@channel_j"})
        channel_id = created.json()["id"]

        monkeypatch.setattr(
            channel_watch,
            "_extract_flat_entries",
            lambda url: _fake_result(["vidJ1", "vidJ2"]),
        )
        r = c.post(f"/api/channels/{channel_id}/check")
        assert r.status_code == 200
        data = r.json()
        assert "check_result" in data
        diag = data["check_result"]
        assert diag["checked"] is True
        assert diag["success"] is True
        assert diag["first_check"] is True
        assert diag["videos_seen"] == 2
        assert diag["new_videos"] == 2
        assert diag["jobs_created"] == 0
        assert data["last_check_status"] == "healthy"
        assert data["videos_seen_last_check"] == 2


# ==============================================================================
# YoutubeTab Multi-Tab Extraction Unpacking Fixture
# ==============================================================================
def test_extract_flat_entries_unpacks_youtubetab_playlists(monkeypatch):
    class _FakeYoutubeTabYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=False):
            return {
                "_type": "playlist",
                "id": "UCchannelID1234567890",
                "title": "Marques Tech",
                "uploader": "Marques Tech",
                "channel_url": "https://www.youtube.com/channel/UCchannelID1234567890",
                "entries": [
                    {
                        "_type": "playlist",
                        "id": "UCchannelID1234567890",
                        "title": "Marques Tech - Videos",
                        "entries": [
                            {"id": "vidReal1", "title": "Real Video 1", "url": "https://www.youtube.com/watch?v=vidReal1"},
                            {"id": "vidReal2", "title": "Real Video 2", "webpage_url": "https://www.youtube.com/watch?v=vidReal2"},
                        ],
                    },
                    {
                        "_type": "playlist",
                        "id": "UCchannelID1234567890",
                        "title": "Marques Tech - Shorts",
                        "entries": [
                            {"id": "short1", "title": "Short 1", "url": "https://www.youtube.com/shorts/short1"},
                        ],
                    },
                ],
            }

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYoutubeTabYDL)
    res = channel_watch._extract_flat_entries("https://www.youtube.com/@marquestech")
    assert res.status == ExtractionStatus.SUCCESS
    assert res.channel_title == "Marques Tech"
    assert res.canonical_url == "https://www.youtube.com/channel/UCchannelID1234567890"
    ids = [e["id"] for e in res.entries]
    assert ids == ["vidReal1", "vidReal2", "short1"]
    # Channel ID itself should never be included as a video
    assert "UCchannelID1234567890" not in ids


def test_delete_channel_removes_it_from_listing():
    with TestClient(app) as c:
        created = c.post("/api/channels", json={"url": "https://www.youtube.com/@del_ch"})
        channel_id = created.json()["id"]
        r = c.delete(f"/api/channels/{channel_id}")
        assert r.status_code == 204
        ids = [item["id"] for item in c.get("/api/channels").json()]
        assert channel_id not in ids


def test_update_channel_toggle_enabled_and_interval():
    with TestClient(app) as c:
        created = c.post("/api/channels", json={"url": "https://www.youtube.com/@up_ch"})
        channel_id = created.json()["id"]
        r = c.patch(
            f"/api/channels/{channel_id}",
            json={"enabled": False, "poll_interval_minutes": 15},
        )
        assert r.status_code == 200
        assert r.json()["enabled"] is False
        assert r.json()["poll_interval_minutes"] == 15
