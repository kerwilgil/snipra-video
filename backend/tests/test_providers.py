"""Tests de la capa de detección (naive + saneo de candidatos)."""

from app.providers.base import ClipCandidate, DetectParams, clamp_candidates
from app.providers.naive import NaiveDetector
from app.providers.registry import get_detector


def test_naive_detect_returns_clips(fake_segments):
    det = NaiveDetector()
    params = DetectParams(num_clips=5, min_seconds=8, max_seconds=90)
    clips = det.detect(fake_segments["segments"], params)
    assert 1 <= len(clips) <= 5
    assert all(c.end > c.start for c in clips)
    assert all(c.title for c in clips)


def test_naive_prioritizes_hook_language():
    segments = [
        {"start": 0.0, "end": 5.0, "text": "Hoy vamos a hablar de una cosa normal."},
        {"start": 5.0, "end": 10.0, "text": "Esto es contexto sin demasiada fuerza."},
        {"start": 10.0, "end": 15.0, "text": "No vas a creer este error brutal que cambia todo."},
        {"start": 15.0, "end": 20.0, "text": "La solucion es rapida y funciona en 3 pasos."},
        {"start": 20.0, "end": 25.0, "text": "Luego seguimos con otro comentario general."},
    ]
    params = DetectParams(num_clips=1, min_seconds=8, max_seconds=30, processing_mode="tutorial")

    clips = NaiveDetector().detect(segments, params)

    assert len(clips) == 1
    assert clips[0].start >= 5
    assert clips[0].score > 0.55
    assert "gancho" in clips[0].reason or "senal" in clips[0].reason


def test_naive_empty_segments():
    assert NaiveDetector().detect([], DetectParams()) == []


def test_registry_defaults_to_naive():
    assert isinstance(get_detector(None), NaiveDetector)
    assert isinstance(get_detector("desconocido"), NaiveDetector)


def test_clamp_discards_out_of_bounds():
    params = DetectParams(num_clips=10, min_seconds=8, max_seconds=30)
    cands = [
        ClipCandidate(start=0, end=20, score=0.9),  # ok
        ClipCandidate(start=10, end=5, score=0.8),  # end<=start -> fuera
        ClipCandidate(start=50, end=200, score=0.7),  # excede duración total
        ClipCandidate(start=0, end=2, score=0.6),  # demasiado corto
    ]
    out = clamp_candidates(cands, params, total_duration=60)
    assert all(0 <= c.start < c.end <= 60 for c in out)
    assert all((c.end - c.start) <= params.max_seconds for c in out)
    # el de 0-20 debe sobrevivir; el invertido y el corto deben caer
    assert any(abs(c.start - 0) < 0.01 and abs(c.end - 20) < 0.01 for c in out)


def test_claude_api_no_envia_temperature():
    """Los modelos actuales devuelven 400 si llega `temperature`.

    El 400 no rompia el job: caia al detector naive en silencio y el usuario
    recibia clips elegidos por heuristica creyendo que los eligio la IA.
    """
    import app.providers.claude as claude_mod

    captured: dict = {}

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)

            class _Block:
                type = "text"
                text = "[]"

            class _Resp:
                content = [_Block()]

            return _Resp()

    class _FakeAnthropic:
        def __init__(self, **_):
            self.messages = _FakeMessages()

    import sys
    import types

    fake_sdk = types.ModuleType("anthropic")
    fake_sdk.Anthropic = _FakeAnthropic
    sys.modules["anthropic"] = fake_sdk
    original_key = claude_mod.settings.anthropic_api_key
    claude_mod.settings.anthropic_api_key = "sk-test"
    try:
        claude_mod.ClaudeDetector(model="claude-sonnet-5")._complete("sys", "user")
    finally:
        claude_mod.settings.anthropic_api_key = original_key
        sys.modules.pop("anthropic", None)

    assert "temperature" not in captured
    assert "top_p" not in captured
    assert "top_k" not in captured
    assert captured["model"] == "claude-sonnet-5"


def test_normalize_no_degrada_suscripcion_a_api_de_pago():
    """claude_code/codex usan el CLI de la suscripcion, no la API con key.

    Como sus modelos empiezan por "claude"/"gpt-", la inferencia de proveedor
    por modelo los reescribia a los proveedores de API de pago.
    """
    from app.api.jobs import _normalize_provider_model

    assert _normalize_provider_model("claude_code", "claude-sonnet-5") == (
        "claude_code",
        "claude-sonnet-5",
    )
    assert _normalize_provider_model("codex", "gpt-5.6-terra") == ("codex", "gpt-5.6-terra")


def test_inferencia_por_modelo_solo_si_nadie_eligio_proveedor(monkeypatch):
    """La deduccion por nombre de modelo es el ULTIMO recurso.

    Si Ajustes tiene un proveedor valido, manda ese: nunca se salta a la API de
    pago por el nombre del modelo. Solo se deduce cuando no hay proveedor
    elegido en ningun lado.
    """
    from app.api.jobs import _normalize_provider_model
    from app.core.config import settings

    monkeypatch.setattr(settings, "llm_provider", "", raising=False)
    monkeypatch.setattr(settings, "llm_model", "", raising=False)
    assert _normalize_provider_model("", "claude-haiku-4-5")[0] == "claude"
    assert _normalize_provider_model("desconocido", "gpt-4o-mini")[0] == "openai"

    # Con un proveedor elegido en Ajustes, la deduccion NO se aplica.
    monkeypatch.setattr(settings, "llm_provider", "naive", raising=False)
    assert _normalize_provider_model("", "claude-haiku-4-5")[0] == "naive"


def test_ajustes_con_suscripcion_nunca_caen_en_la_api_de_pago(monkeypatch):
    """La UI normalmente NO manda proveedor: usa el de Ajustes.

    Proteger solo el caso explicito no alcanzaba. Con Ajustes en claude_code y
    la UI sin mandar proveedor, la deduccion por nombre de modelo reescribia a
    'claude' y el job se facturaba a la API key del usuario en vez de consumir
    su suscripcion. La API solo debe usarse si el usuario la eligio.
    """
    from app.api.jobs import _normalize_provider_model
    from app.core.config import settings

    monkeypatch.setattr(settings, "llm_provider", "claude_code", raising=False)
    monkeypatch.setattr(settings, "llm_model", "claude-sonnet-5", raising=False)

    assert _normalize_provider_model(None, None)[0] == "claude_code"
    assert _normalize_provider_model("", "")[0] == "claude_code"
    assert _normalize_provider_model(None, "claude-sonnet-5")[0] == "claude_code"

    monkeypatch.setattr(settings, "llm_provider", "codex", raising=False)
    monkeypatch.setattr(settings, "llm_model", "gpt-5.6-terra", raising=False)
    assert _normalize_provider_model(None, None)[0] == "codex"


def test_modelo_de_ajustes_no_se_filtra_a_un_proveedor_distinto(monkeypatch):
    """Bug real: Ajustes en NVIDIA/deepseek, usuario elige Claude Code al subir.

    job.model terminaba en "deepseek-ai/deepseek-v4-pro" con provider=
    "claude_code" -- ese nombre se pasaba tal cual como --model al CLI de
    Claude Code, que no lo reconoce. settings.llm_model solo debe heredarse
    si el proveedor elegido es el MISMO que el guardado en Ajustes; si es
    otro, usa el default de ESE proveedor.
    """
    from app.api.jobs import _normalize_provider_model
    from app.core.config import settings

    monkeypatch.setattr(settings, "llm_provider", "nvidia", raising=False)
    monkeypatch.setattr(settings, "llm_model", "deepseek-ai/deepseek-v4-pro", raising=False)

    assert _normalize_provider_model("claude_code", None) == ("claude_code", "claude-sonnet-5")

    # El mismo proveedor que Ajustes SI hereda su modelo guardado.
    assert _normalize_provider_model("nvidia", None) == (
        "nvidia",
        "deepseek-ai/deepseek-v4-pro",
    )


def test_la_api_de_pago_solo_sale_si_se_elige(monkeypatch):
    """Contraparte: si el usuario SI eligio la API, se respeta."""
    from app.api.jobs import _normalize_provider_model
    from app.core.config import settings

    monkeypatch.setattr(settings, "llm_provider", "claude", raising=False)
    monkeypatch.setattr(settings, "llm_model", "claude-haiku-4-5", raising=False)
    assert _normalize_provider_model(None, None)[0] == "claude"
    # y elegirla explicitamente en el request tambien
    monkeypatch.setattr(settings, "llm_provider", "claude_code", raising=False)
    assert _normalize_provider_model("claude", "claude-haiku-4-5")[0] == "claude"
