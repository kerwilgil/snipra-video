"""No inference calls or paid providers: catalog and persistence contract tests."""
import json

import httpx
import pytest

from app.providers import nvidia_catalog as nim


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    monkeypatch.setattr(nim.settings, "storage_dir", tmp_path)
    monkeypatch.setattr(nim.settings, "nvidia_base_url", "https://nim.example/v1")
    monkeypatch.setattr(nim, "_retry_at", 0)
    monkeypatch.setattr(nim, "_retry_source", "")
    clock = [10000.0]
    monkeypatch.setattr(nim.time, "time", lambda: clock[0])
    rows = [{"id": "vendor/text-instruct", "modalities": ["text"]},
            {"id": "vendor/image-chat", "modalities": ["text", "image"]},
            {"id": "vendor/embeddings", "modalities": ["embedding"]}]
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        return httpx.Response(200, json={"data": rows}, request=httpx.Request("GET", url))

    monkeypatch.setattr(nim.httpx, "get", fetch)
    return rows, clock, calls, tmp_path


def test_refresh_removes_retired_models_and_discovers_new_ones(catalog):
    rows, clock, calls, _ = catalog
    assert nim.resolve_model("auto") == "vendor/text-instruct"
    assert nim.resolve_model("") == "vendor/text-instruct"
    assert len(calls) == 1
    rows[:] = [{"id": "new-provider/new-chat", "modalities": ["text"]}]
    clock[0] += 301
    assert nim.resolve_model("auto") == "new-provider/new-chat"
    with pytest.raises(nim.NvidiaCatalogError, match="no está disponible"):
        nim.resolve_model("vendor/text-instruct")


def test_manual_is_preserved_and_vision_is_separate(catalog):
    assert nim.resolve_model("vendor/image-chat") == "vendor/image-chat"
    assert nim.resolve_model("auto", vision=True) == "vendor/image-chat"
    with pytest.raises(nim.NvidiaCatalogError, match="no está disponible"):
        nim.resolve_model("vendor/text-instruct", vision=True)
    assert all(m["id"] != "vendor/embeddings" for m in nim.get_catalog()["models"])


def test_policy_updates_without_build_and_cannot_invent_availability(catalog):
    _, _, _, path = catalog
    nim.get_catalog()
    (path / "nvidia_policy.json").write_text(json.dumps({
        "text_priority": ["not-available/chat", "vendor/image-chat"],
    }))
    assert nim.resolve_model("auto") == "vendor/image-chat"
    (path / "nvidia_policy.json").write_text(json.dumps({
        "text_priority": ["vendor/text-instruct"],
        "models": {"vendor/image-chat": {"modalities": []}},
    }))
    assert nim.resolve_model("auto") == "vendor/text-instruct"
    with pytest.raises(nim.NvidiaCatalogError, match="no ofrece modelos"):
        nim.resolve_model("auto", vision=True)


def test_offline_uses_bounded_stale_cache_then_fails_closed(catalog, monkeypatch):
    _, clock, calls, _ = catalog
    nim.get_catalog()

    def offline(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(nim.httpx, "get", offline)
    clock[0] += 301
    data = nim.get_catalog()
    assert data["available"] and data["stale"]
    assert "desactualizado" in data["warning"]
    assert nim.resolve_model("auto") == "vendor/text-instruct"
    clock[0] += 86401
    assert nim.get_catalog()["models"] == []
    with pytest.raises(nim.NvidiaCatalogError, match="no disponible"):
        nim.resolve_model("auto")


def test_offline_without_cache_does_not_invent_models(catalog, monkeypatch):
    monkeypatch.setattr(nim.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(
        httpx.ConnectError("offline")))
    assert nim.get_catalog()["models"] == []
    with pytest.raises(nim.NvidiaCatalogError):
        nim.resolve_model("some/manual-model")


def test_malformed_provider_response_keeps_bounded_cache(catalog, monkeypatch):
    _, clock, _, _ = catalog
    nim.get_catalog()
    clock[0] += 301
    monkeypatch.setattr(nim.httpx, "get", lambda url, **kw: httpx.Response(
        200, json=[], request=httpx.Request("GET", url)))
    assert nim.get_catalog()["stale"]
    assert nim.resolve_model("auto") == "vendor/text-instruct"


def test_usage_records_actual_model_and_selection(catalog):
    with nim.capture_model_usage() as usage:
        nim.resolve_model("auto")
        nim.resolve_model("auto", vision=True)
    assert [m["model"] for m in usage] == ["vendor/text-instruct", "vendor/image-chat"]
    assert all(m["selection"] == "auto" for m in usage)


def test_empty_catalog_is_authoritative_and_does_not_resurrect_retired_ids(catalog):
    rows, clock, _, _ = catalog
    nim.get_catalog()
    rows.clear()
    clock[0] += 301
    assert nim.get_catalog()["models"] == []
    with pytest.raises(nim.NvidiaCatalogError, match="no ofrece modelos"):
        nim.resolve_model("auto")
    with pytest.raises(nim.NvidiaCatalogError, match="no está disponible"):
        nim.resolve_model("vendor/text-instruct")


def test_detection_sends_resolved_id_and_keeps_global_auto(catalog, monkeypatch):
    from types import SimpleNamespace
    from app.providers.nvidia import NvidiaDetector

    monkeypatch.setattr(nim.settings, "llm_model", "auto")
    sent = []

    def create(**kwargs):
        sent.append(kwargs["model"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    detector = NvidiaDetector("auto")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(detector, "_client", lambda: client)
    assert detector._complete("system", "user") == "ok"
    assert sent == ["vendor/text-instruct"]
    assert nim.settings.llm_model == "auto"


def test_vision_resolves_auto_at_runtime_and_keeps_opt_in(catalog, monkeypatch):
    from app.providers.vision import resolve_vision

    monkeypatch.setattr(nim.settings, "vision_provider", "nvidia")
    monkeypatch.setattr(nim.settings, "nvidia_vision_model", "auto")
    assert resolve_vision("nvidia", "auto", via=True) == ("nvidia", "vendor/image-chat")
    assert resolve_vision("nvidia", "auto") == ("", "")
    monkeypatch.setattr(nim.settings, "vision_provider", "auto")
    assert resolve_vision("nvidia", "auto", via=True) == ("", "")


def test_inferred_modalities_are_labeled_and_overridable(catalog):
    rows, _, _, path = catalog
    rows[:] = [{"id": "vendor/new-vision-chat"}, {"id": "vendor/reranker"}]
    model = nim.get_catalog()["models"][0]
    assert model["vision"] and model["compatibility_source"] == "inferred"
    (path / "nvidia_policy.json").write_text(json.dumps({
        "models": {"vendor/new-vision-chat": {"modalities": ["text"]}},
    }))
    assert not nim.get_catalog()["models"][0]["vision"]


def test_settings_persist_auto_and_legacy_manual_without_rewriting(catalog, monkeypatch):
    from app.core import runtime_settings as runtime
    from app.api.settings import SettingsUpdate, update_settings

    _, _, _, path = catalog
    monkeypatch.setattr(runtime, "RUNTIME_SETTINGS_PATH", path / "app_settings.json")
    monkeypatch.setattr(nim.settings, "llm_provider", "claude")
    monkeypatch.setattr(nim.settings, "llm_model", "previous-provider/model")
    monkeypatch.setattr(nim.settings, "nvidia_vision_model", "auto")
    # Avoid hardware/CLI discovery while retaining the real settings mutation.
    import app.api.settings as api_settings
    monkeypatch.setattr(api_settings, "get_settings", lambda: {
        "model": nim.settings.llm_model, "nvidia_vision_model": nim.settings.nvidia_vision_model,
    })
    result = update_settings(SettingsUpdate(provider="nvidia"))
    assert result["model"] == "auto"  # Do not carry a different provider's model.
    result = update_settings(SettingsUpdate(model="auto"))
    assert result["model"] == "auto"
    persisted = json.loads((path / "app_settings.json").read_text())
    assert persisted["llm_model"] == "auto"
    result = update_settings(SettingsUpdate(model="retired/manual", nvidia_vision_model="retired/vision"))
    assert result == {"model": "retired/manual", "nvidia_vision_model": "retired/vision"}
