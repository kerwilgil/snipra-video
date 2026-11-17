"""Provider bridge contracts, offline fallback and actual process cleanup."""

import asyncio
import json
import multiprocessing
import time
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.decision import ai
from app.decision.engines import RulesDecisionEngine
from app.decision.models import DecisionTask
from app.decision.registry import resolve_decision_engine
from app.decision.service import classify_batch
from tests.test_decision import config, request


@pytest.mark.parametrize("provider", ["openai", "claude", "nvidia"])
def test_auto_inherits_job(provider):
    engine = resolve_decision_engine("auto", job_provider=provider, job_model="job-model")
    assert engine.id() == provider
    assert engine.model == "job-model"
    assert resolve_decision_engine(provider).model == ai.DEFAULT_MODELS[provider]


@pytest.mark.parametrize("provider", [None, "naive", "codex", "claude_code"])
def test_safe_local_resolution(provider):
    assert isinstance(resolve_decision_engine("auto", job_provider=provider), RulesDecisionEngine)


def test_unknown_resolution():
    assert resolve_decision_engine("auto", job_provider="unknown") is None


@pytest.mark.parametrize("provider", ["openai", "claude", "nvidia"])
@pytest.mark.parametrize("fault", [
    None, "json", "missing", "probabilities", "options", "task", "provider", "model",
    "provenance", "confidence", "exception", "timeout", "abstain",
])
async def test_bridge_and_fallback(monkeypatch, provider, fault):
    calls = []
    async def complete(p, m, user, schema):
        calls.append((p, m))
        data = json.loads(user)
        assert data["requests"][0]["evidence"]["existing_score"] == 0.87
        if fault == "exception":
            raise RuntimeError("unavailable")
        if fault == "timeout":
            await asyncio.sleep(1)
        if fault == "json":
            return "```json invalid```"
        results = []
        for item in data["requests"]:
            options = item["options"]
            label = options[0]
            results.append(dict(
                schema_version="1", engine=p, model=m, task=item["task"],
                status="success", classification=label,
                probabilities={x: float(x == label) for x in options},
                mutually_exclusive=True, confidence=0.9, duration_ms=0,
                evidence_refs=item["evidence"]["evidence_refs"],
                recommended_action=label if item["task"] == "recommended_action" else None,
            ))
        first = results[0]
        if fault == "missing":
            del first["confidence"]
        if fault == "probabilities":
            first["probabilities"][first["classification"]] = float("nan")
        if fault == "options":
            first["probabilities"]["invented"] = 0
        if fault == "task":
            first["task"] = "content_type"
        if fault == "provider":
            first["engine"] = "other"
        if fault == "model":
            first["model"] = "other"
        if fault == "provenance":
            first["evidence_refs"] = ["invented"]
        if fault == "confidence":
            first["confidence"] = 0.1
        if fault == "abstain":
            for item in results:
                item.update(
                    status="unavailable", classification=None, confidence=None,
                    recommended_action=None, probabilities={},
                )
        return json.dumps({"results": results})

    monkeypatch.setattr(ai, "complete_isolated", complete)
    monkeypatch.setattr(ai.ExistingAIDecisionEngine, "available", _available)
    requests = [request(task) for task in DecisionTask]
    before = [r.model_dump() for r in requests]
    results = await classify_batch(
        requests, config(decision_engine_provider="auto", decision_engine_timeout_ms=20),
        job_provider=provider, job_model="job-model",
    )
    assert len(calls) == 1
    assert len(results) == 5
    assert [r.model_dump() for r in requests] == before
    if fault is None:
        assert all(r.engine == provider and r.model == "job-model" for r in results)
    else:
        assert results[0].engine == "rules"
        assert results[0].status == "success"
        if fault != "confidence":
            assert results[1].status == "unavailable"


async def _available(self):
    return True


async def test_unconfigured_provider(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    results = await classify_batch(
        [request()], config(decision_engine_provider="auto"), job_provider="openai",
    )
    assert results[0].engine == "rules"


def _sleep_child(*args):
    time.sleep(60)


async def test_real_cancellation_reaps_child(monkeypatch):
    monkeypatch.setattr(ai, "_complete_child", _sleep_child)
    before = {p.pid for p in multiprocessing.active_children()}
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(ai.complete_isolated("openai", "model", "{}", {}), 0.2)
    assert {p.pid for p in multiprocessing.active_children()} == before


@pytest.mark.parametrize("provider", ["openai", "nvidia", "claude"])
def test_child_reuses_adapter_and_local_settings(monkeypatch, provider):
    from app.core import runtime_settings
    from app.providers import registry

    loaded = []
    calls = []
    sent = []
    monkeypatch.setattr(runtime_settings, "load_runtime_settings", lambda: loaded.append(True))

    class Adapter:
        def complete(self, system, user, **kwargs):
            calls.append(kwargs)
            assert "NOT a probability" in system
            return SimpleNamespace(content='{"results":[]}', provider=provider, model="job-model")

    class Detector:
        def _complete(self, system, user):
            calls.append(system)
            assert "OUTPUT CONTRACT" in system
            return '{"results":[]}'

    monkeypatch.setattr(
        registry, "get_structured_completion_adapter",
        lambda p, m: None if p == "claude" else Adapter(),
    )
    monkeypatch.setattr(registry, "get_detector", lambda p, m: Detector())
    connection = SimpleNamespace(send=sent.append, close=lambda: None)
    ai._complete_child(connection, provider, "job-model", "{}", {"type": "object"})
    assert loaded == [True]
    assert len(calls) == 1
    assert sent == [(True, '{"results":[]}')]
    if provider != "claude":
        assert calls[0]["json_schema"] == {"type": "object"}
