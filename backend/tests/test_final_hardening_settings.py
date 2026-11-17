import asyncio
import json
import subprocess
import sys

from app.api.jobs import _defaults
from app.core import runtime_settings
from app.core.config import settings
from app.decision.ai import ExistingAIDecisionEngine
from app.models import Job, SourceType
from app.worker import runner


def test_active_worker_refreshes_settings_at_next_work_boundary(monkeypatch, tmp_path):
    path = tmp_path / "app_settings.json"
    monkeypatch.setattr(runtime_settings, "RUNTIME_SETTINGS_PATH", path)
    for key in runtime_settings.EDITABLE_FIELDS:
        monkeypatch.setattr(settings, key, getattr(settings, key))
    monkeypatch.setattr(settings, "openai_api_key", "")
    assert not asyncio.run(ExistingAIDecisionEngine("openai").available())
    job = Job(source_type=SourceType.upload, provider="naive", model="original")
    payload = {
        "openai_api_key": "hardening-sentinel",
        "llm_provider": "openai", "llm_model": "gpt-4o-mini",
        "whisper_model": "tiny", "enable_media_signals": False,
        "enable_visual_rerank": False, "yt_dlp_cookies_path": "fixture-cookies.txt",
    }
    # A separate process represents the API writer while the worker stays alive.
    script = (
        "import sys,json; from pathlib import Path; "
        "from app.core import runtime_settings as r; "
        "r.RUNTIME_SETTINGS_PATH=Path(sys.argv[1]); "
        "r.apply_runtime_settings(json.loads(sys.argv[2]))"
    )
    subprocess.run([sys.executable, "-c", script, str(path), json.dumps(payload)], check=True)
    assert not settings.openai_api_key
    monkeypatch.setattr(runner, "claim_next", lambda: job.id)
    processed = []

    def process(job_id, claimed):
        assert job_id == job.id and claimed
        for key, value in payload.items():
            assert getattr(settings, key) == value
        assert asyncio.run(ExistingAIDecisionEngine("openai").available())
        defaults = _defaults(None, None, None, None, None, None, None, None, None, None)
        assert defaults["provider"] == "openai"
        assert defaults["model"] == "gpt-4o-mini"
        assert (job.provider, job.model) == ("naive", "original")
        processed.append(job_id)

    assert runner._try_one_work_round(0, ["job"], process_job_fn=process)[0]
    assert processed == [job.id]
