"""Tests de la API con la BD/lifespan activos."""

import json
import socket
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app import __version__
from app.api import clips as clips_api
from app.core.config import settings
from app.core.db import engine
from app.main import app
from app.models import Clip, Job, JobStatus, SourceType, TrackedChannel, TrackedChannelVideo
from app.pipeline import ingest, render


@pytest.fixture(autouse=True)
def _resolve_test_urls_as_public(monkeypatch):
    monkeypatch.setattr(
        ingest.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))],
    )


def test_health_and_endpoints():
    with TestClient(app) as c:
        h = c.get("/api/health")
        assert h.status_code == 200
        assert h.json()["status"] == "ok"
        project = tomllib.loads(
            (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        )
        assert h.json()["version"] == __version__ == project["project"]["version"]

        jobs = c.get("/api/jobs")
        assert jobs.status_code == 200
        assert isinstance(jobs.json(), list)

        s = c.get("/api/settings")
        assert s.status_code == 200
        assert "naive" in s.json()["providers"]
        assert "auto_smart" in {mode["value"] for mode in s.json()["processing_modes"]}
        assert "podcast_viral" in {mode["value"] for mode in s.json()["processing_modes"]}
        assert "gaming" in {mode["value"] for mode in s.json()["processing_modes"]}
        assert "gaming_auto" not in {mode["value"] for mode in s.json()["processing_modes"]}
        assert "gaming_hook" not in {mode["value"] for mode in s.json()["processing_modes"]}
        assert "viral_bold" in {style["value"] for style in s.json()["caption_styles"]}
        assert "ffmpeg" in s.json()["binaries"]
        assert s.json()["min_clip_seconds"] <= s.json()["max_clip_seconds"]
        assert "enable_media_signals" in s.json()

        storage = c.get("/api/storage")
        assert storage.status_code == 200
        assert "work_bytes" in storage.json()


def test_cross_origin_upload_is_rejected_but_local_frontend_is_allowed():
    files = {"file": ("sample.mp4", b"not-a-real-video", "video/mp4")}

    with TestClient(app) as c:
        blocked = c.post(
            "/api/jobs",
            headers={"Origin": "https://malicious.example"},
            files=files,
        )
        assert blocked.status_code == 403
        assert blocked.json()["detail"] == "Origen no permitido para modificar Snipra."

        allowed = c.post(
            "/api/jobs",
            headers={"Origin": "http://localhost:5173"},
            files=files,
        )
        assert allowed.status_code == 201


def test_cross_origin_protection_covers_all_mutating_methods():
    with TestClient(app) as c:
        for method in ("post", "put", "patch", "delete"):
            response = c.request(
                method,
                "/api/settings",
                headers={"Origin": "null"},
                json={},
            )
            assert response.status_code == 403

        without_origin = c.put(
            "/api/settings",
            headers={"Sec-Fetch-Site": "cross-site"},
            json={},
        )
        assert without_origin.status_code == 403


def test_extension_origin_requires_configured_token(monkeypatch):
    with TestClient(app) as c:
        monkeypatch.setattr(settings, "extension_token", "")
        no_token_configured = c.put(
            "/api/settings",
            headers={
                "Origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
                "X-Snipra-Extension-Token": "whatever",
            },
            json={},
        )
        assert no_token_configured.status_code == 403
        assert "Token de extensión" in no_token_configured.json()["detail"]

        monkeypatch.setattr(settings, "extension_token", "s3cr3t-token")

        wrong_token = c.put(
            "/api/settings",
            headers={
                "Origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
                "X-Snipra-Extension-Token": "not-the-token",
            },
            json={},
        )
        assert wrong_token.status_code == 403

        missing_header = c.put(
            "/api/settings",
            headers={"Origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop"},
            json={},
        )
        assert missing_header.status_code == 403

        correct_token = c.put(
            "/api/settings",
            headers={
                "Origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
                "X-Snipra-Extension-Token": "s3cr3t-token",
            },
            json={},
        )
        assert correct_token.status_code == 200


def test_extension_v120_job_options_contract(monkeypatch):
    """La extensión 1.2.0 envía perfiles, formatos, captions y edición nativa."""
    token = "extension-v120-token"
    headers = {
        "Origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
        "X-Snipra-Extension-Token": token,
    }
    with TestClient(app) as c:
        # El lifespan carga app_settings al entrar al TestClient; el token de la
        # extensión debe simularse después de esa carga, igual que en producción.
        monkeypatch.setattr(settings, "extension_token", token)
        clips = c.post(
            "/api/jobs/url",
            headers=headers,
            json={
                "url": "https://example.com/gameplay.mp4",
                "output_mode": "clips",
                "download_quality": "1440",
                "processing_mode": "gaming_highlights",
                "num_clips": 9,
                "formats": ["9x16", "1x1"],
                "captions_enabled": False,
                "editorial_options": {
                    "compact_editing": True,
                    "semantic_cards": True,
                    "privacy_assist": True,
                    "protected_zones": True,
                    "overlay_qa": True,
                    "transcription_quality": "precise",
                },
            },
        )
        assert clips.status_code == 201
        clips_job = c.get(f"/api/jobs/{clips.json()['id']}").json()
        assert clips_job["processing_mode"] == "gaming_highlights"
        assert clips_job["num_clips"] == 1
        assert clips_job["formats"] == ["9x16", "1x1"]
        assert clips_job["captions_enabled"] is False
        assert clips_job["download_quality"] == "1440"
        assert clips_job["editorial_options"]["compact_editing"] is True
        assert clips_job["editorial_options"]["semantic_cards"] is True
        assert clips_job["editorial_options"]["transcription_quality"] == "precise"

        audio = c.post(
            "/api/jobs/url",
            headers=headers,
            json={
                "url": "https://example.com/interview.mp4",
                "output_mode": "audio",
                "audio_format": "flac",
            },
        )
        assert audio.status_code == 201
        audio_job = c.get(f"/api/jobs/{audio.json()['id']}").json()
        assert audio_job["output_mode"] == "audio"
        assert audio_job["audio_format"] == "flac"


def test_extension_token_regenerate_endpoint():
    with TestClient(app) as c:
        first = c.post("/api/settings/extension-token/regenerate")
        assert first.status_code == 200


def test_job_media_options_contract():
    """Media foundation: media_options por job, normalizadas y backwards-compat."""
    with TestClient(app) as c:
        response = c.post(
            "/api/jobs/url",
            json={
                "url": "https://example.com/hdr.mp4",
                "output_mode": "clips",
                "media_options": {
                    "destination_profile": "tiktok",
                    "color_handling": "sdr",
                    "stabilization": "medium",
                    "video_denoise": "light",
                    "audio_cleanup": "voice",
                },
            },
        )
        assert response.status_code == 201
        job = c.get(f"/api/jobs/{response.json()['id']}").json()
        assert job["media_options"]["destination_profile"] == "tiktok"
        assert job["media_options"]["color_handling"] == "sdr"
        assert job["media_options"]["stabilization"] == "medium"
        assert job["media_options"]["video_denoise"] == "light"
        assert job["media_options"]["audio_cleanup"] == "voice"

        # Valores inválidos caen a defaults seguros, claves desconocidas fuera.
        response = c.post(
            "/api/jobs/url",
            json={
                "url": "https://example.com/bad.mp4",
                "media_options": {
                    "destination_profile": "netflix",
                    "color_handling": "hdr5000",
                    "stabilization": "turbo",
                    "evil": True,
                },
            },
        )
        assert response.status_code == 201
        job = c.get(f"/api/jobs/{response.json()['id']}").json()
        assert job["media_options"] == {
            "destination_profile": "auto",
            "color_handling": "auto",
            "stabilization": "off",
            "video_denoise": "off",
            "audio_cleanup": "off",
        }
        assert "evil" not in job["media_options"]

        # Sin media_options: el contrato sigue funcionando como siempre.
        response = c.post("/api/jobs/url", json={"url": "https://example.com/legacy.mp4"})
        assert response.status_code == 201
        job = c.get(f"/api/jobs/{response.json()['id']}").json()
        assert job["media_options"]["stabilization"] == "off"
        assert job["media_options"]["color_handling"] == "auto"


def test_settings_media_defaults_and_catalog():
    """GET /api/settings expone defaults de medios + catálogo de destinos."""
    with TestClient(app) as c:
        response = c.get("/api/settings")
        assert response.status_code == 200
        payload = response.json()
        assert set(payload["media_defaults"]) == {
            "destination_profile",
            "color_handling",
            "stabilization",
            "video_denoise",
            "audio_cleanup",
        }
        assert payload["media_defaults"]["stabilization"] == "off"
        assert payload["media_defaults"]["color_handling"] == "auto"
        profile_ids = {item["id"] for item in payload["destination_profiles"]}
        assert {"generic_vertical", "tiktok", "instagram_reels", "youtube"} <= profile_ids


def test_settings_media_defaults_persist(monkeypatch, tmp_path):
    """Los defaults de medios se guardan como el resto de ajustes runtime."""
    from app.core import runtime_settings

    path = tmp_path / "app_settings.json"
    monkeypatch.setattr(runtime_settings, "RUNTIME_SETTINGS_PATH", path)
    monkeypatch.setattr(settings, "stabilization", "off")
    monkeypatch.setattr(settings, "destination_profile", "auto")

    with TestClient(app) as c:
        response = c.put(
            "/api/settings",
            json={"stabilization": "light", "destination_profile": "tiktok"},
        )
    assert response.status_code == 200
    assert response.json()["media_defaults"]["stabilization"] == "light"
    assert response.json()["media_defaults"]["destination_profile"] == "tiktok"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["stabilization"] == "light"
    assert persisted["destination_profile"] == "tiktok"


def test_extension_token_regenerate_rotates(monkeypatch):
    """La rotación del token de extensión sigue funcionando (regresión)."""
    token = "extension-rotate-token"
    with TestClient(app) as c:
        monkeypatch.setattr(settings, "extension_token", token)
        first = c.post("/api/settings/extension-token/regenerate")
        assert first.status_code == 200
        token_a = first.json()["extension_token"]
        assert token_a

        second = c.post("/api/settings/extension-token/regenerate")
        token_b = second.json()["extension_token"]
        assert token_b and token_b != token_a


def test_get_settings_never_leaks_raw_extension_token():
    """GET /api/settings no tiene proteccion de Origin (solo mutaciones la tienen);
    una extension con host_permissions amplios puede leer su respuesta igual, asi
    que el valor crudo del token NUNCA debe viajar ahi, solo un booleano."""
    with TestClient(app) as c:
        regenerate = c.post("/api/settings/extension-token/regenerate")
        token = regenerate.json()["extension_token"]
        assert token

        settings_payload = c.get("/api/settings").json()
        assert "extension_token" not in settings_payload
        assert settings_payload["extension_token_configured"] is True
        assert token not in str(settings_payload)


def test_update_settings():
    with TestClient(app) as c:
        r = c.put(
            "/api/settings",
            json={
                "default_num_clips": 4,
                "caption_style": "clean",
                "min_clip_seconds": 12,
                "max_clip_seconds": 75,
                "enable_media_signals": True,
                "storage_retention_days": 45,
            },
        )
        assert r.status_code == 200
        assert r.json()["default_num_clips"] == 4
        assert r.json()["caption_style"] == "clean"
        assert r.json()["min_clip_seconds"] == 12
        assert r.json()["max_clip_seconds"] == 75
        assert r.json()["enable_media_signals"] is True
        assert r.json()["storage_retention_days"] == 45


def test_update_settings_persiste_proveedor_y_modelo_de_vision(monkeypatch, tmp_path):
    """El PUT no debe anunciar exito y descartar silenciosamente estos campos."""
    from app.core import runtime_settings

    path = tmp_path / "app_settings.json"
    monkeypatch.setattr(runtime_settings, "RUNTIME_SETTINGS_PATH", path)
    monkeypatch.setattr(settings, "vision_provider", "auto")
    monkeypatch.setattr(settings, "vision_model", "")

    with TestClient(app) as c:
        response = c.put(
            "/api/settings",
            json={"vision_provider": "claude_code", "vision_model": "claude-sonnet-5"},
        )

    assert response.status_code == 200
    assert response.json()["vision_provider"] == "claude_code"
    assert response.json()["vision_model"] == "claude-sonnet-5"
    assert response.json()["vision_models_by_provider"]["codex"] == [
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
    ]
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["vision_provider"] == "claude_code"
    assert persisted["vision_model"] == "claude-sonnet-5"


def test_update_settings_descarta_modelo_de_otro_proveedor(monkeypatch, tmp_path):
    from app.core import runtime_settings

    monkeypatch.setattr(runtime_settings, "RUNTIME_SETTINGS_PATH", tmp_path / "app_settings.json")
    monkeypatch.setattr(settings, "vision_provider", "auto")
    monkeypatch.setattr(settings, "vision_model", "")

    with TestClient(app) as c:
        response = c.put(
            "/api/settings",
            json={"vision_provider": "codex", "vision_model": "claude-sonnet-5"},
        )

    assert response.status_code == 200
    assert response.json()["vision_provider"] == "codex"
    assert response.json()["vision_model"] == ""
    assert response.json()["vision_effective"]["model"] == "gpt-5.6-terra"


def test_update_api_keys_persiste_sin_revelar_valores(monkeypatch, tmp_path):
    from app.core import runtime_settings

    path = tmp_path / "app_settings.json"
    monkeypatch.setattr(runtime_settings, "RUNTIME_SETTINGS_PATH", path)
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "nvidia_api_key", "")
    secrets = {
        "claude": "claude-test-value",
        "openai": "openai-test-value",
        "nvidia": "nvidia-test-value",
    }

    with TestClient(app) as c:
        response = c.put("/api/settings/api-keys", json=secrets)
        cleared = c.put("/api/settings/api-keys", json={"claude": ""})

    assert response.status_code == 200
    assert response.json()["keys_present"] == {
        "claude": True,
        "openai": True,
        "nvidia": True,
    }
    assert not any(secret in response.text for secret in secrets.values())
    assert cleared.json()["keys_present"]["claude"] is False
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["anthropic_api_key"] == ""
    assert persisted["openai_api_key"] == secrets["openai"]
    assert persisted["nvidia_api_key"] == secrets["nvidia"]


def test_create_url_job_queues():
    with TestClient(app) as c:
        r = c.post(
            "/api/jobs/url",
            json={"url": "https://example.com/video.mp4", "processing_mode": "gaming"},
        )
        assert r.status_code == 201
        assert r.json()["status"] == "queued"

        job = c.get(f"/api/jobs/{r.json()['id']}")
        assert job.status_code == 200
        assert job.json()["processing_mode"] == "gaming"

        automatic = c.post(
            "/api/jobs/url",
            json={
                "url": "https://example.com/automatic.mp4",
                "processing_mode": "gaming_auto",
                "num_clips": 7,
            },
        )
        automatic_job = c.get(f"/api/jobs/{automatic.json()['id']}")
        assert automatic.status_code == 201
        assert automatic_job.json()["processing_mode"] == "gaming_auto"
        assert automatic_job.json()["num_clips"] == 7

        bad = c.post("/api/jobs/url", json={"url": "   "})
        assert bad.status_code == 400

        bad_mode = c.post(
            "/api/jobs/url",
            json={"url": "https://example.com/video.mp4", "processing_mode": "unknown"},
        )
        assert bad_mode.status_code == 400

        openai_page = c.post(
            "/api/jobs/url",
            json={"url": "https://openai.com/chatgpt-work/?video=1208185418"},
        )
        assert openai_page.status_code == 400
        assert "ChatGPT/OpenAI" in openai_page.json()["detail"]


def test_download_job_contract_and_asset_endpoint():
    with TestClient(app) as c:
        created = c.post(
            "/api/jobs/url",
            json={
                "url": "https://example.com/video.mp4",
                "output_mode": "download",
                "download_quality": "2160",
            },
        )
        assert created.status_code == 201
        job_id = created.json()["id"]
        job = c.get(f"/api/jobs/{job_id}")
        assert job.json()["output_mode"] == "download"
        assert job.json()["download_quality"] == "2160"

        with Session(engine) as session:
            stored = session.get(Job, job_id)
            assert stored is not None
            stored.status = JobStatus.done
            session.add(stored)
            session.commit()
        source = ingest.job_dir(job_id) / "source.mp4"
        source.write_bytes(b"video")

        response = c.get(f"/api/jobs/{job_id}/download")
        assert response.status_code == 200
        assert response.content == b"video"


def test_create_clips_without_captions_persists_option():
    with TestClient(app) as c:
        response = c.post(
            "/api/jobs/url",
            json={
                "url": "https://example.com/video.mp4",
                "output_mode": "clips",
                "captions_enabled": False,
            },
        )
        assert response.status_code == 201
        job = c.get(f"/api/jobs/{response.json()['id']}")
        assert job.json()["captions_enabled"] is False


def test_update_clip_rejects_end_after_source_duration():
    with TestClient(app) as c, Session(engine) as session:
        job = Job(
            source_type=SourceType.url,
            source_name="https://example.com/video.mp4",
            status=JobStatus.done,
            duration=60,
        )
        session.add(job)
        session.flush()
        clip = Clip(job_id=job.id, index=0, start=10, end=20, title="Limite")
        session.add(clip)
        session.commit()

        response = c.patch(f"/api/clips/{clip.id}", json={"end": 61})

        assert response.status_code == 400
        assert "duracion" in response.json()["detail"]


def test_apply_to_all_enqueues_persistent_clip_operation():
    with TestClient(app) as c, Session(engine) as session:
        job = Job(
            source_type=SourceType.url,
            source_name="https://example.com/video.mp4",
            status=JobStatus.done,
        )
        session.add(job)
        session.flush()
        clip = Clip(job_id=job.id, index=0, start=10, end=20, title="Masivo")
        session.add(clip)
        session.commit()

        response = c.post(
            f"/api/clips/{clip.id}/style",
            json={"style": "clean", "apply_to_all": True},
        )

        assert response.status_code == 202
        operation = response.json()
        assert operation["kind"] == "style"
        assert operation["status"] == "queued"
        status = c.get(f"/api/clip-operations/{operation['id']}")
        assert status.status_code == 200
        assert status.json()["progress"] == 0

        duplicate = c.post(
            f"/api/clips/{clip.id}/style",
            json={"style": "neon", "apply_to_all": True},
        )
        assert duplicate.status_code == 409


def test_clip_dir_falls_back_by_index_after_title_change(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    job = Job(
        id="job-title-change",
        source_type=SourceType.url,
        source_name="https://example.com/video.mp4",
        status=JobStatus.done,
    )
    clip = Clip(job_id=job.id, index=0, start=0, end=5, title="Titulo nuevo")
    previous = settings.work_dir / job.id / "clips" / "01-titulo-anterior"
    previous.mkdir(parents=True)

    assert clips_api._clip_dir(job, clip) == previous


def test_list_jobs_reports_channel_label_for_channel_created_jobs():
    with TestClient(app) as c, Session(engine) as session:
        manual_job = Job(
            source_type=SourceType.url,
            source_name="https://example.com/manual.mp4",
            status=JobStatus.done,
        )
        channel_job = Job(
            source_type=SourceType.url,
            source_name="https://example.com/from-channel.mp4",
            status=JobStatus.done,
        )
        session.add(manual_job)
        session.add(channel_job)
        session.flush()

        channel = TrackedChannel(url="https://www.youtube.com/@somechannel", label="Mi canal")
        session.add(channel)
        session.flush()
        session.add(
            TrackedChannelVideo(channel_id=channel.id, video_id="vid1", job_id=channel_job.id)
        )
        session.commit()

        response = c.get("/api/jobs")
        assert response.status_code == 200
        by_id = {item["id"]: item for item in response.json()}
        assert by_id[manual_job.id]["channel_label"] is None
        assert by_id[channel_job.id]["channel_label"] == "Mi canal"

        single = c.get(f"/api/jobs/{channel_job.id}")
        assert single.json()["channel_label"] == "Mi canal"


def test_retry_job_and_logs():
    with TestClient(app) as c:
        r = c.post("/api/jobs/url", json={"url": "https://example.com/video.mp4"})
        assert r.status_code == 201
        job_id = r.json()["id"]
        detail = c.get(f"/api/jobs/{job_id}")
        assert detail.json()["stage_started_at"].endswith("Z")

        with Session(engine) as session:
            job = session.get(Job, job_id)
            assert job is not None
            job.status = JobStatus.failed
            job.error = "fallo simulado"
            session.add(job)
            session.commit()

        retry = c.post(f"/api/jobs/{job_id}/retry")
        assert retry.status_code == 200
        assert retry.json()["status"] == "queued"
        assert retry.json()["error"] is None
        assert retry.json()["stage_started_at"].endswith("Z")

        logs = c.get(f"/api/jobs/{job_id}/logs")
        assert logs.status_code == 200
        assert any("reencolado" in item["message"] for item in logs.json())


def test_export_manifest_and_cleanup():
    with TestClient(app) as c:
        r = c.post("/api/jobs/url", json={"url": "https://example.com/video.mp4"})
        assert r.status_code == 201
        job_id = r.json()["id"]

        manifest = c.get(f"/api/jobs/{job_id}/export-manifest")
        assert manifest.status_code == 200
        assert manifest.json()["id"] == job_id
        assert manifest.json()["clips"] == []

        cleanup = c.post("/api/storage/cleanup", json={"remove_orphans": True})
        assert cleanup.status_code == 200
        assert "removed_paths" in cleanup.json()


def test_cancel_active_job():
    # La API solo encola; el worker persistente toma el job fuera de esta prueba.
    with TestClient(app) as c:
        r = c.post("/api/jobs/url", json={"url": "https://example.com/video.mp4"})
        assert r.status_code == 201
        job_id = r.json()["id"]

        cancel = c.post(f"/api/jobs/{job_id}/cancel")
        assert cancel.status_code == 200
        assert cancel.json()["status"] == "canceled"

        logs = c.get(f"/api/jobs/{job_id}/logs")
        assert logs.status_code == 200
        assert any("cancelado" in item["message"] for item in logs.json())

        retry = c.post(f"/api/jobs/{job_id}/retry")
        assert retry.status_code == 200
        assert retry.json()["status"] == "queued"


def test_clip_timeline_endpoint_serves_artifact():
    with TestClient(app) as c, Session(engine) as session:
        job = Job(
            source_type=SourceType.url,
            source_name="https://example.com/video.mp4",
            status=JobStatus.done,
            title="Timeline test",
        )
        session.add(job)
        session.flush()
        clip = Clip(
            job_id=job.id,
            index=0,
            start=10,
            end=20,
            title="Clip timeline",
            score=0.8,
        )
        session.add(clip)
        session.commit()
        session.refresh(job)
        session.refresh(clip)

        clip_dir = ingest.job_dir(job.id) / "clips" / render._slug(clip.index, clip.title)
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "timeline-9x16-frame-1.jpg").write_bytes(b"fake-jpg")
        (clip_dir / "timeline.json").write_text(
            """
            {
              "version": 1,
              "clip": {"source_start": 10, "source_end": 20},
              "processing": {"requested_reframe": "center"},
              "formats": [
                {
                  "format": "9x16",
                  "quality": {"passed": true, "attempts": 1, "issues": [], "checks": {}},
                  "frames": [{"time": 1.0, "filename": "timeline-9x16-frame-1.jpg"}],
                  "audio_energy": [{"time": 0.0, "rms": 0.4}]
                }
              ],
              "subtitles": [],
              "words": []
            }
            """,
            encoding="utf-8",
        )

        timeline = c.get(f"/api/clips/{clip.id}/timeline")
        assert timeline.status_code == 200
        assert timeline.json()["formats"][0]["audio_energy"][0]["rms"] == 0.4
        frame_url = timeline.json()["formats"][0]["frames"][0]["url"]
        assert frame_url.endswith("/timeline-9x16-frame-1.jpg")

        frame = c.get(frame_url)
        assert frame.status_code == 200


def test_rewrite_clip_captions_endpoint(monkeypatch):
    with TestClient(app) as c, Session(engine) as session:
        job = Job(
            source_type=SourceType.url,
            source_name="https://example.com/video.mp4",
            status=JobStatus.done,
            title="Rewrite test",
            provider="naive",
        )
        session.add(job)
        session.flush()
        clip = Clip(
            job_id=job.id,
            index=0,
            start=10,
            end=20,
            title="Clip rewrite",
            score=0.8,
        )
        session.add(clip)
        session.commit()
        session.refresh(job)
        session.refresh(clip)

        clip_dir = ingest.job_dir(job.id) / "clips" / render._slug(clip.index, clip.title)
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "subtitles.json").write_text(
            '[{"start":0,"end":2,"text":"Eh este punto importa","words":[]}]',
            encoding="utf-8",
        )

        def fake_rerender(session, job, item, style_config=None, subtitles=None):
            assert subtitles is not None
            assert "este" not in subtitles[0]["text"].lower()
            return {"id": item.id, "assets": []}

        monkeypatch.setattr("app.api.clips._rerender_from_artifacts", fake_rerender)

        response = c.post(f"/api/clips/{clip.id}/captions/rewrite", json={})

        assert response.status_code == 200
        assert response.json()["id"] == clip.id


def test_translate_clip_captions_endpoint(monkeypatch):
    with TestClient(app) as c, Session(engine) as session:
        job = Job(
            source_type=SourceType.url,
            source_name="https://example.com/video.mp4",
            status=JobStatus.done,
            title="Translate test",
            provider="claude",
        )
        session.add(job)
        session.flush()
        clip = Clip(
            job_id=job.id,
            index=0,
            start=10,
            end=20,
            title="Clip translate",
            score=0.8,
        )
        session.add(clip)
        session.commit()
        session.refresh(job)
        session.refresh(clip)

        clip_dir = ingest.job_dir(job.id) / "clips" / render._slug(clip.index, clip.title)
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "subtitles.json").write_text(
            '[{"start":0,"end":2,"text":"Hola mundo","words":[]}]',
            encoding="utf-8",
        )

        def fake_translate(blocks, *, target_language, provider, model):
            assert target_language == "ingles"
            return [{**blocks[0], "text": "Hello world", "words": []}]

        def fake_rerender(session, job, item, style_config=None, subtitles=None):
            assert subtitles is not None
            assert subtitles[0]["text"] == "Hello world"
            return {"id": item.id, "assets": []}

        monkeypatch.setattr("app.api.clips.translate_subtitle_blocks", fake_translate)
        monkeypatch.setattr("app.api.clips._rerender_from_artifacts", fake_rerender)

        response = c.post(
            f"/api/clips/{clip.id}/captions/translate",
            json={"target_language": "ingles"},
        )

        assert response.status_code == 200
        assert response.json()["id"] == clip.id


def test_translate_clip_captions_naive_provider_returns_clear_error():
    with TestClient(app) as c, Session(engine) as session:
        job = Job(
            source_type=SourceType.url,
            source_name="https://example.com/video.mp4",
            status=JobStatus.done,
            title="Translate naive",
            provider="naive",
        )
        session.add(job)
        session.flush()
        clip = Clip(
            job_id=job.id,
            index=0,
            start=10,
            end=20,
            title="Clip translate naive",
            score=0.8,
        )
        session.add(clip)
        session.commit()
        session.refresh(job)
        session.refresh(clip)

        clip_dir = ingest.job_dir(job.id) / "clips" / render._slug(clip.index, clip.title)
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "subtitles.json").write_text(
            '[{"start":0,"end":2,"text":"Hola mundo","words":[]}]',
            encoding="utf-8",
        )

        response = c.post(
            f"/api/clips/{clip.id}/captions/translate",
            json={"target_language": "ingles"},
        )

        assert response.status_code == 422


def test_translate_clip_captions_apply_to_all_enqueues_operation(monkeypatch):
    with TestClient(app) as c, Session(engine) as session:
        job = Job(
            source_type=SourceType.url,
            source_name="https://example.com/video.mp4",
            status=JobStatus.done,
            title="Translate batch",
            provider="claude",
        )
        session.add(job)
        session.flush()
        clip = Clip(
            job_id=job.id,
            index=0,
            start=10,
            end=20,
            title="Clip translate batch",
            score=0.8,
        )
        session.add(clip)
        session.commit()
        session.refresh(job)
        session.refresh(clip)

        response = c.post(
            f"/api/clips/{clip.id}/captions/translate",
            json={"target_language": "portugues", "apply_to_all": True},
        )

        assert response.status_code == 202
        operation = response.json()
        assert operation["kind"] == "translate_captions"
        assert operation["status"] == "queued"
