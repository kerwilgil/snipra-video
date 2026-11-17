"""Tests del re-rank visual (sin tocar FFmpeg ni APIs reales)."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.providers import vision
from app.providers.base import ClipCandidate


def _cand(start, score, title):
    return ClipCandidate(start=start, end=start + 30, score=score, title=title)


def test_rerank_noop_for_provider_without_vision():
    cands = [_cand(0, 0.9, "a"), _cand(60, 0.5, "b")]
    out = vision.rerank_by_vision(cands, "fake.mp4", provider="naive", model="x")
    assert out == cands  # naive no tiene visión → sin cambios


def test_rerank_noop_para_nvidia_porque_alucina():
    """NVIDIA ya no se ofrece para visión (ver test_nvidia_no_se_ofrece_para_vision).

    Antes este test verificaba que el re-rank con NVIDIA forzara el modelo
    multimodal de NIM. Se retiró esa ruta: el modelo alucinaba sobre frames
    reales, y una puntuación inventada desordena el ranking en vez de mejorarlo.
    """
    cands = [_cand(0, 0.6, "a"), _cand(60, 0.4, "b")]
    out = vision.rerank_by_vision(cands, "fake.mp4", provider="nvidia", model="llama-3.3-70b")
    assert out == cands


def test_score_nvidia_ignora_el_modelo_de_texto_del_job(monkeypatch):
    """El scorer sigue implementado por si se reactiva: no debe usar el modelo de texto.

    Los modelos de texto de NIM (nemotron, llama-3.3, deepseek) no ven imágenes.
    """
    used = {}

    def _fake_compatible(client, model, frames, title):
        used["model"] = model
        return 0.9, "acción"

    monkeypatch.setattr(vision, "_score_openai_compatible", _fake_compatible)
    monkeypatch.setattr("openai.OpenAI", lambda **kw: object())
    monkeypatch.setattr("app.providers.nvidia_catalog.resolve_model",
                        lambda value, **kw: "catalog/current-vision")
    vision._score_nvidia("llama-3.3-70b", [b"x"], "a")
    assert used["model"] == "catalog/current-vision"


def test_rerank_noop_when_no_candidates():
    assert vision.rerank_by_vision([], "fake.mp4", provider="claude", model="x") == []


def test_rerank_reorders_by_combined_score(monkeypatch):
    # 'b' tiene score textual bajo pero visualmente es lo mejor → debe subir.
    cands = [_cand(0, 0.80, "a"), _cand(60, 0.50, "b")]
    monkeypatch.setattr(vision, "_extract_frames", lambda *a, **k: [b"jpgbytes"])
    visual_scores = {"a": (0.10, "estático"), "b": (1.0, "acción brutal")}
    monkeypatch.setattr(vision, "_score_claude", lambda model, frames, title: visual_scores[title])

    out = vision.rerank_by_vision(cands, "fake.mp4", provider="claude", model="claude-x")

    # a: 0.55*0.80 + 0.45*0.10 = 0.485 ; b: 0.55*0.50 + 0.45*1.0 = 0.725 → b primero
    assert [c.title for c in out] == ["b", "a"]
    assert "visual: acción brutal" in out[0].reason


def test_structured_shadow_off_does_no_work(monkeypatch):
    monkeypatch.setattr(vision.settings, "enable_structured_visual_shadow", False)
    calls = []
    monkeypatch.setattr(vision, "_score_structured_shadow", lambda *args: calls.append(args))
    candidate = _cand(0, 0.5, "a")
    assert vision._append_structured_shadow(candidate, [b"x"], "codex", "terra") == candidate
    assert calls == []


def test_structured_shadow_on_is_telemetry_only(monkeypatch):
    monkeypatch.setattr(vision.settings, "enable_structured_visual_shadow", True)
    response = (
        '{"event_present":true,"event_type":"combat","event_strength":"high",'
        '"payoff_present":false,"outcome":"uncertain","reaction_present":false,'
        '"visual_novelty":"medium","menu_or_idle":false,"peak_frame":3,'
        '"evidence_frames":[3],"evidence_quality":"partial","observed_evidence":"golpe"}'
    )
    monkeypatch.setattr(vision, "_score_structured_shadow", lambda *args: response)
    candidate = _cand(0, 0.5, "a")
    result = vision._append_structured_shadow(candidate, [b"x"], "codex", "terra")
    entry = result.score_breakdown[-1]
    assert result.score == candidate.score and result.reason == candidate.reason
    assert entry.signal == "structured_visual_shadow"
    assert entry.contribution == 0 and entry.score_before == entry.score_after == candidate.score
    assert entry.metadata["structured_event_features"]["outcome"] == "uncertain"


def test_structured_shadow_off_and_on_preserve_legacy_ordering(monkeypatch):
    candidates = [_cand(0, 0.8, "a"), _cand(60, 0.5, "b")]
    monkeypatch.setattr(vision, "_extract_frames", lambda *args, **kwargs: [b"x"])
    monkeypatch.setattr(
        vision,
        "_score_claude",
        lambda model, frames, title: (0.1 if title == "a" else 1.0, "legacy"),
    )
    monkeypatch.setattr(vision.settings, "enable_structured_visual_shadow", False)
    off = vision.rerank_by_vision(candidates, "fake.mp4", provider="claude", model="x")
    response = (
        '{"event_present":true,"event_type":"combat","event_strength":"low",'
        '"payoff_present":false,"outcome":"uncertain","reaction_present":false,'
        '"visual_novelty":"low","menu_or_idle":false,"peak_frame":1,'
        '"evidence_frames":[1],"evidence_quality":"partial","observed_evidence":"golpe"}'
    )
    monkeypatch.setattr(vision.settings, "enable_structured_visual_shadow", True)
    monkeypatch.setattr(vision, "_score_structured_shadow", lambda *args: response)
    monkeypatch.setattr(vision.StructuredVisualCache, "get", lambda *args: None)
    monkeypatch.setattr(vision.StructuredVisualCache, "put", lambda *args: None)
    on = vision.rerank_by_vision(candidates, "fake.mp4", provider="claude", model="x")
    assert [item.title for item in off] == [item.title for item in on]
    assert [(item.start, item.end, item.score, item.reason) for item in off] == [
        (item.start, item.end, item.score, item.reason) for item in on
    ]
    entries = [item.score_breakdown[-1] for item in on]
    assert all(
        entry.contribution == 0 and entry.score_before == entry.score_after for entry in entries
    )


def test_rerank_keeps_order_when_frames_fail(monkeypatch):
    # Sin frames extraíbles (FFmpeg falla) → no_scored → lista original intacta.
    cands = [_cand(0, 0.9, "a"), _cand(60, 0.5, "b")]
    monkeypatch.setattr(vision, "_extract_frames", lambda *a, **k: [])
    out = vision.rerank_by_vision(cands, "fake.mp4", provider="claude", model="x")
    assert out == cands


def test_rerank_survives_api_error(monkeypatch):
    cands = [_cand(0, 0.9, "a")]
    monkeypatch.setattr(vision, "_extract_frames", lambda *a, **k: [b"x"])

    def _boom(*a, **k):
        raise RuntimeError("API caída")

    monkeypatch.setattr(vision, "_score_claude", _boom)
    out = vision.rerank_by_vision(cands, "fake.mp4", provider="claude", model="x")
    assert out == cands  # error de API → candidato sin cambios, no rompe


def test_rerank_reports_each_candidate_even_when_frames_fail(monkeypatch):
    cands = [_cand(0, 0.9, "a"), _cand(60, 0.5, "b")]
    monkeypatch.setattr(vision, "_extract_frames", lambda *a, **k: [])
    updates: list[tuple[float, str]] = []

    vision.rerank_by_vision(
        cands,
        "fake.mp4",
        provider="claude",
        model="x",
        progress_callback=lambda progress, message: updates.append((progress, message)),
    )

    assert [progress for progress, _ in updates] == [0.0, 0.5, 0.5, 1.0]
    assert updates[0][1] == "Visión IA: procesando candidato 1/2"
    assert updates[-1][1] == "Visión IA: candidato 2/2 completado"


def test_codex_vision_fails_fast_and_falls_back_to_claude_code(monkeypatch):
    cands = [_cand(0, 0.5, "a"), _cand(60, 0.5, "b")]
    monkeypatch.setattr(vision, "_extract_frames", lambda *a, **k: [b"x"])
    codex_calls = 0
    claude_calls = 0

    def broken_codex(*args):
        nonlocal codex_calls
        codex_calls += 1
        raise RuntimeError("Codex -i falló en Windows")

    def working_claude(*args):
        nonlocal claude_calls
        claude_calls += 1
        return 0.9, "acción verificada"

    monkeypatch.setattr(vision, "_score_codex", broken_codex)
    monkeypatch.setattr(vision, "_score_claude_code", working_claude)
    monkeypatch.setattr(
        "app.providers.cli_subscription.resolve_provider_cli",
        lambda provider: "claude.exe" if provider == "claude_code" else None,
    )
    events: list[tuple[str, str]] = []

    out = vision.rerank_by_vision(
        cands,
        "fake.mp4",
        provider="codex",
        model="gpt-5.6-terra",
        event_callback=lambda level, message: events.append((level, message)),
    )

    assert codex_calls == 1
    assert claude_calls == 2
    assert all(item.score > 0.5 for item in out)
    assert any("fallback visual" in message for _, message in events)


def test_parse_score_extracts_json():
    assert vision._parse_score('{"score": 0.83, "reason": "buena"}') == (0.83, "buena")
    assert vision._parse_score("basura sin json") == (0.5, "")
    # clamp fuera de rango
    assert vision._parse_score('{"score": 5}')[0] == 1.0


def test_respuesta_visual_invalida_es_fallo_no_score_neutral():
    """Salida vacia/cuota agotada no debe modificar el ranking con un 0.5 falso."""
    with pytest.raises(ValueError):
        vision._parse_visual_response("")
    with pytest.raises(ValueError):
        vision._parse_visual_response('{"is_error":true,"result":"quota exhausted"}')


def test_vision_caps_own_death_and_preserves_peak_frame():
    score, reason, _ = vision._parse_visual_response(
        '{"score":0.91,"reason":"hay medallas","peak_frame":2,"outcome":"death"}'
    )

    assert score == 0.25
    assert "resultado visual: death" in reason
    assert "pico visual frame 2/4" in reason


def test_claude_code_pide_copy_cierra_stdin_y_valida_envelope(monkeypatch):
    from app.providers import cli_subscription

    captured = {}

    def _run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=(
                '{"is_error":false,"result":"'
                '{\\"score\\":0.9,\\"reason\\":\\"accion\\",'
                '\\"title\\":\\"Triple baja\\",\\"hook\\":\\"Mira esto\\",'
                '\\"hashtags\\":[\\"#gaming\\"]}"}'
            ),
        )

    monkeypatch.setattr(cli_subscription, "resolve_provider_cli", lambda _: "claude")
    monkeypatch.setattr(vision.subprocess, "run", _run)

    score, reason, copy = vision._score_claude_code("claude-sonnet-5", [b"jpg"], "")

    assert score == 0.9
    assert reason == "accion"
    assert copy["title"] == "Triple baja"
    assert "hashtags" in captured["cmd"][2]
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL


def test_claude_code_propaga_error_de_cuota(monkeypatch):
    from app.providers import cli_subscription

    monkeypatch.setattr(cli_subscription, "resolve_provider_cli", lambda _: "claude")
    monkeypatch.setattr(
        vision.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout='{"is_error":true,"result":"quota exhausted"}',
        ),
    )

    with pytest.raises(cli_subscription._CliError, match="quota exhausted"):
        vision._score_claude_code("claude-sonnet-5", [b"jpg"], "titulo")


def test_codex_usa_imagen_absoluta_y_archivo_de_salida(monkeypatch):
    from app.providers import cli_subscription

    captured = {}

    def _run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        out_path = Path(cmd[cmd.index("-o") + 1])
        out_path.write_text('{"score":0.8,"reason":"imagen visible"}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(cli_subscription, "resolve_provider_cli", lambda _: "codex.exe")
    monkeypatch.setattr(vision.subprocess, "run", _run)

    score, reason, _ = vision._score_codex("gpt-5.6-luna", [b"jpg"], "titulo")

    image_arg = captured["cmd"][captured["cmd"].index("-i") + 1]
    assert Path(image_arg).is_absolute()
    assert "-o" in captured["cmd"]
    assert score == 0.8
    assert reason == "imagen visible"
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL


def test_nvidia_no_puede_recibir_frames_de_512px():
    """Regresion: con FRAME_WIDTH=512 el re-rank con NVIDIA devolvia 400 siempre.

    NVIDIA NIM tiene 32k de contexto y la imagen viaja en base64 dentro del
    prompt (~0.95 tokens/byte). Un frame de 512px son ~39k tokens: ni uno entra.
    Como este modulo nunca rompe el job, fallaba en silencio.
    """
    from app.providers.vision import frame_limits

    n, width = frame_limits("nvidia")
    assert width <= 336, "un frame mas ancho no entra en el contexto de NVIDIA"
    assert n == 1, "NVIDIA no admite mas de un frame por llamada"


def test_proveedores_de_suscripcion_tienen_vision():
    """claude_code/codex usan la suscripcion; sin esto la vision exigia API key."""
    from app.providers.vision import VISION_PROVIDERS

    assert "claude_code" in VISION_PROVIDERS
    assert "codex" in VISION_PROVIDERS


def test_todo_proveedor_con_vision_tiene_scorer():
    """Un proveedor listado sin scorer revienta con KeyError en pleno job."""
    import inspect

    from app.providers import vision

    src = inspect.getsource(vision.rerank_by_vision)
    for provider in vision.VISION_PROVIDERS:
        assert f'"{provider}"' in src, f"{provider} esta en VISION_PROVIDERS pero no tiene scorer"


def test_nvidia_no_se_ofrece_para_vision():
    """NVIDIA alucina en visión: puntuar con razones inventadas es peor que no puntuar.

    Medido sobre frames reales: un frame de Call of Duty -> "un hombre lanzando
    un balon"; uno de podcast (un hombre ante un microfono) -> "una multitud en
    un parque". Sigue siendo valido como proveedor de TEXTO.
    """
    from app.providers.registry import DEFAULT_MODELS
    from app.providers.vision import VISION_PROVIDERS

    assert "nvidia" not in VISION_PROVIDERS
    assert "nvidia" in DEFAULT_MODELS, "debe seguir disponible como proveedor de texto"


def test_avisa_si_se_pide_vision_con_proveedor_sin_vision(monkeypatch):
    """Sin aviso, el job termina bien y el usuario cree que la vision corrio."""
    from app.core.config import settings
    from app.pipeline import detect as detect_mod

    monkeypatch.setattr(settings, "enable_visual_rerank", True, raising=False)
    eventos: list[tuple[str, str]] = []

    segments = [{"start": i * 5.0, "end": i * 5.0 + 5.0, "text": "hola " * 30} for i in range(40)]
    detect_mod.detect_clips(
        {"segments": segments, "duration": 200.0, "language": "es"},
        provider="nvidia",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        num_clips=2,
        formats=["9x16"],
        processing_mode="podcast_viral",
        video_path="C:/no/existe.mp4",
        event_callback=lambda level, msg: eventos.append((level, msg)),
    )

    avisos = [m for lvl, m in eventos if lvl == "warning" and "analisis visual" in m.lower()]
    assert avisos, f"deberia avisar que la vision se omitio; eventos={eventos}"


def test_vision_redacta_copy_para_clips_mudos(monkeypatch):
    """Los candidatos por señal salen sin titulo ni hashtags reales.

    Nacen de picos de audio, no del transcript, asi que el redactor (que lee
    texto) los dejaba como "(sin titulo)" con hashtags genericos. La vision ya
    mira esos fotogramas: pedirle el copy no cuesta una llamada extra.
    """
    mudo = ClipCandidate(start=0, end=30, score=0.5, title="", reason="señal: picos de audio")
    monkeypatch.setattr(vision, "_extract_frames", lambda *a, **k: [b"x"])
    monkeypatch.setattr(
        vision,
        "_score_claude",
        lambda m, f, t: (
            0.8,
            "accion clara",
            {"title": "Triple baja seguida", "hook": "No lo vio venir", "hashtags": ["#warzone"]},
        ),
    )

    out = vision.rerank_by_vision([mudo], "fake.mp4", provider="claude", model="x")

    assert out[0].title == "Triple baja seguida"
    assert out[0].hook == "No lo vio venir"
    assert out[0].hashtags == ["#warzone"]


def test_vision_no_pisa_el_copy_del_detector_de_texto(monkeypatch):
    """Si el texto ya redacto el titulo, sabe mas: leyo lo que se dijo."""
    con_titulo = ClipCandidate(start=0, end=30, score=0.5, title="Titulo del LLM")
    monkeypatch.setattr(vision, "_extract_frames", lambda *a, **k: [b"x"])
    monkeypatch.setattr(
        vision, "_score_claude", lambda m, f, t: (0.8, "ok", {"title": "Titulo de la vision"})
    )

    out = vision.rerank_by_vision([con_titulo], "fake.mp4", provider="claude", model="x")
    assert out[0].title == "Titulo del LLM"


def test_scorer_de_dos_valores_sigue_funcionando(monkeypatch):
    """Retrocompatibilidad: (score, reason) sin copy no debe romper."""
    c = ClipCandidate(start=0, end=30, score=0.5, title="x")
    monkeypatch.setattr(vision, "_extract_frames", lambda *a, **k: [b"x"])
    monkeypatch.setattr(vision, "_score_claude", lambda m, f, t: (0.9, "acción"))
    out = vision.rerank_by_vision([c], "fake.mp4", provider="claude", model="x")
    assert "visual: acción" in out[0].reason


def test_prompt_pide_copy_solo_si_falta_el_titulo():
    assert "hashtags" in vision._prompt_for("")
    assert "hashtags" not in vision._prompt_for("Ya tengo titulo")


def test_parse_copy_normaliza_hashtags():
    out = vision._parse_copy('{"title":"T","hook":"H","hashtags":["warzone","#cod"]}')
    assert out["hashtags"] == ["#warzone", "#cod"]
    assert vision._parse_copy("sin json") == {}
    assert vision._parse_copy('{"score":0.5}') == {}


def test_vision_provider_auto_sigue_al_de_texto(monkeypatch):
    """Elegir una suscripcion no debe hacer que la vision salte a una API de pago."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "vision_provider", "auto", raising=False)
    monkeypatch.setattr(settings, "vision_model", "", raising=False)
    assert vision.resolve_vision("claude_code", "claude-sonnet-5") == (
        "claude_code",
        "claude-sonnet-5",
    )
    assert vision.resolve_vision("codex", "gpt-5.6-terra") == ("codex", "gpt-5.6-terra")
    # proveedor de texto sin vision -> no hay vision
    assert vision.resolve_vision("naive", "") == ("", "")


def test_vision_provider_se_puede_elegir_distinto_al_de_texto(monkeypatch):
    """Caso pedido: texto con Claude Code, vision con Codex."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "vision_provider", "codex", raising=False)
    monkeypatch.setattr(settings, "vision_model", "", raising=False)
    provider, model = vision.resolve_vision("claude_code", "claude-sonnet-5")
    assert provider == "codex"
    assert model == vision.VISION_DEFAULT_MODELS["codex"]

    # y el modelo se puede fijar a mano (para que se vea cual es en Ajustes)
    monkeypatch.setattr(settings, "vision_model", "gpt-5.6-luna", raising=False)
    assert vision.resolve_vision("claude_code", "claude-sonnet-5") == ("codex", "gpt-5.6-luna")


def test_solo_se_ofrecen_proveedores_de_vision_por_suscripcion():
    """Las APIs de pago no deben poder elegirse por descuido: son gastos con tarjeta.

    Siguen soportadas si se fijan a proposito en Ajustes/.env, pero no se listan.
    """
    # N.1: nvidia se ofrece para VIA (free-tier, sin tarjeta); sigue siendo imposible
    # elegir una API pagada por descuido. Las APIs de pago siguen sin listarse.
    assert vision.VISION_PROVIDER_CHOICES == ("auto", "claude_code", "codex", "nvidia")
    assert "claude" not in vision.VISION_PROVIDER_CHOICES
    assert "openai" not in vision.VISION_PROVIDER_CHOICES
    assert "claude" not in vision.VISION_PROVIDER_CHOICES
    assert "openai" not in vision.VISION_PROVIDER_CHOICES


def test_modelos_de_vision_filtrados_por_proveedor():
    """La union de todos mostraba claude-sonnet-5 dos veces y gpt-4o-mini con codex."""
    codex = vision.vision_models_for("codex")
    assert codex == ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
    assert not [m for m in codex if m.startswith("claude")]
    assert not [m for m in codex if m.startswith("gpt-4o")]

    cc = vision.vision_models_for("claude_code")
    assert all(m.startswith("claude") for m in cc)
    assert len(cc) == len(set(cc)), "sin duplicados"

    assert vision.vision_models_for("desconocido") == []


def test_el_modelo_por_defecto_esta_en_la_lista_del_proveedor():
    """Si el default no estuviera en la lista, el selector arrancaria en blanco."""
    for provider, default in vision.VISION_DEFAULT_MODELS.items():
        assert default in vision.vision_models_for(provider), provider
