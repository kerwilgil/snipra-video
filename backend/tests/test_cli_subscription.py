"""Tests del detector por suscripción (CLI de Claude Code / Codex).

Mockean el subprocess: no dependen del CLI real ni de una sesión iniciada.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from app.providers import cli_subscription
from app.providers.base import DetectParams
from app.providers.cli_subscription import (
    ClaudeCodeDetector,
    CodexDetector,
    _CliError,
    _extract_claude_result,
    _extract_codex_result,
    subscription_cli_available,
)
from app.providers.registry import DEFAULT_MODELS, SUBSCRIPTION_PROVIDERS, get_detector


def test_registry_exposes_subscription_providers():
    assert "claude_code" in DEFAULT_MODELS
    assert "codex" in DEFAULT_MODELS
    assert set(SUBSCRIPTION_PROVIDERS) == {"claude_code", "codex"}
    assert type(get_detector("claude_code")).__name__ == "ClaudeCodeDetector"
    assert type(get_detector("codex")).__name__ == "CodexDetector"


def test_extract_claude_result_reads_result_field():
    envelope = '{"type":"result","is_error":false,"result":"[{\\"start\\":1,\\"end\\":30}]"}'
    assert _extract_claude_result(envelope) == '[{"start":1,"end":30}]'


def test_extract_claude_result_raises_on_cli_error():
    envelope = '{"type":"result","is_error":true,"result":"OAuth session expired"}'
    with pytest.raises(_CliError, match="OAuth session expired"):
        _extract_claude_result(envelope)


def test_extract_claude_result_passthrough_when_not_json():
    # si por lo que sea no es el envelope JSON, devuelve el texto crudo
    assert _extract_claude_result('[{"start":1}]') == '[{"start":1}]'


def test_subscription_unavailable_when_cli_missing(monkeypatch):
    monkeypatch.setattr(cli_subscription, "_resolve_executable", lambda name: None)
    assert subscription_cli_available("claude_code") is False
    assert subscription_cli_available("codex") is False


def test_claude_code_detector_raises_clear_error_when_cli_missing(monkeypatch):
    monkeypatch.setattr(cli_subscription, "_resolve_executable", lambda name: None)
    with pytest.raises(_CliError, match="no está instalado"):
        ClaudeCodeDetector()._complete("system", "user")


def test_claude_code_detector_parses_cli_output(monkeypatch):
    monkeypatch.setattr(cli_subscription, "_resolve_executable", lambda name: "claude")

    def fake_run(cmd, **kwargs):
        assert "--model" in cmd and "sonnet" in cmd
        assert "--output-format" in cmd and "json" in cmd
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"is_error":false,"result":"[{\\"start\\":1,\\"end\\":40,\\"title\\":\\"x\\"}]"}',
            stderr="",
        )

    monkeypatch.setattr(cli_subscription.subprocess, "run", fake_run)
    segments = [{"start": 0, "end": 50, "text": "hola"}]
    params = DetectParams(num_clips=1, min_seconds=30, max_seconds=90, formats=["9x16"])
    clips = ClaudeCodeDetector(model="sonnet").detect(segments, params)
    assert len(clips) == 1
    assert clips[0].start == 1
    assert clips[0].end == 40


def test_codex_detector_reads_output_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_subscription, "resolve_provider_cli", lambda provider: "codex.exe")

    def fake_run(cmd, **kwargs):
        # codex escribe la respuesta en el archivo pasado con -o
        out_index = cmd.index("-o") + 1
        from pathlib import Path

        Path(cmd[out_index]).write_text(
            '{"clips":[{"start":2,"end":45,"title":"y","hook":"h",'
            '"score":0.8,"reason":"r","hashtags":[],"description":"d"}]}',
            encoding="utf-8",
        )
        assert cmd[-1] == "-"
        assert kwargs["input"]
        assert "--output-schema" in cmd
        assert "--ephemeral" in cmd
        assert "--ignore-rules" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cli_subscription.subprocess, "run", fake_run)
    segments = [{"start": 0, "end": 60, "text": "hola mundo"}]
    params = DetectParams(num_clips=1, min_seconds=30, max_seconds=90, formats=["9x16"])
    clips = CodexDetector(model="gpt-5-codex").detect(segments, params)
    assert len(clips) == 1
    assert clips[0].start == 2


def test_extract_codex_result_rejects_empty_or_malformed_output():
    with pytest.raises(_CliError, match="vacia"):
        _extract_codex_result("")
    with pytest.raises(_CliError, match="JSON invalido"):
        _extract_codex_result("no-json")
    with pytest.raises(_CliError, match="clips"):
        _extract_codex_result('{"resultado":[]}')


def test_extract_codex_result_unwraps_structured_clips():
    raw = (
        '{"clips":[{"start":1,"end":50,"title":"Ñ","hook":"","score":0.7,'
        '"reason":"","hashtags":[],"description":""}]}'
    )
    clips = _extract_codex_result(raw)
    assert '"start": 1' in clips
    assert "Ñ" in clips


def test_codex_detector_raises_clear_error_on_empty_output(monkeypatch):
    monkeypatch.setattr(cli_subscription, "resolve_provider_cli", lambda provider: "codex.exe")

    def fake_run(cmd, **kwargs):
        out_index = cmd.index("-o") + 1
        from pathlib import Path

        Path(cmd[out_index]).write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cli_subscription.subprocess, "run", fake_run)
    with pytest.raises(_CliError, match="respuesta vacia"):
        CodexDetector()._complete("system", "user")


def test_claude_code_detector_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(cli_subscription, "_resolve_executable", lambda name: "claude")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(cli_subscription.subprocess, "run", fake_run)
    with pytest.raises(_CliError, match="boom"):
        ClaudeCodeDetector()._complete("system", "user")


def test_cli_decodifica_utf8_no_el_locale_del_sistema():
    """Sin `encoding` explicito, Python usa el locale (cp1252 en Windows es-*).

    La salida UTF-8 del CLI se decodificaba mal y el mojibake terminaba quemado
    en el titulo del clip: "Punalada" llegaba como "PuA+-alada".
    """
    import inspect
    import subprocess

    import app.providers.cli_subscription as mod

    calls: list[dict] = []
    original = subprocess.run

    def _spy(cmd, **kwargs):
        calls.append(kwargs)

        class _P:
            returncode = 0
            stdout = '{"result": "[]"}'
            stderr = ""

        return _P()

    subprocess.run = _spy
    mod._resolve_executable = lambda _: "claude"
    try:
        mod.ClaudeCodeDetector()._complete("sys", "user")
    finally:
        subprocess.run = original

    assert calls, "no se invoco el CLI"
    assert calls[0].get("encoding") == "utf-8"

    # Ambos CLIs (claude y codex) deben fijarlo, no solo uno.
    src = inspect.getsource(mod)
    assert src.count('encoding="utf-8"') >= 2


def test_resolver_mapea_proveedor_a_binario():
    """El binario de claude_code se llama "claude", no "claude_code".

    Pasar el nombre del proveedor a shutil.which devuelve None y el CLI parece
    "no instalado" aunque este ahi. Por ese error la vision con claude_code
    fallaba SIEMPRE, y en silencio: rerank_by_vision captura la excepcion y
    devuelve los candidatos sin tocar.
    """
    import app.providers.cli_subscription as mod

    assert mod.PROVIDER_BINARIES["claude_code"] == "claude"
    assert mod.PROVIDER_BINARIES["codex"] == "codex"
    assert mod.resolve_provider_cli("desconocido") is None

    llamadas = []
    original = mod._resolve_executable
    mod._resolve_executable = lambda n: (llamadas.append(n), "/fake/" + n)[1]
    try:
        assert mod.resolve_provider_cli("claude_code") == "/fake/claude"
    finally:
        mod._resolve_executable = original
    assert llamadas == ["claude"], "debe resolver el BINARIO, no el proveedor"


def test_los_cli_cierran_stdin():
    """Ambos CLIs reciben el prompt por stdin con `input=`: el pipe se escribe y
    se cierra. Sin eso, Codex imprime "Reading additional input from stdin..." y
    espera; en el worker (proceso oculto) eso daba un archivo de salida vacio y el
    usuario veia un JSONDecodeError sin relacion con la causa.
    """
    import inspect

    import app.providers.cli_subscription as mod

    for cls in (mod.ClaudeCodeDetector, mod.CodexDetector):
        src = inspect.getsource(cls)
        assert "input=prompt" in src, f"{cls.__name__} debe escribir el prompt y cerrar el pipe"
        assert "stdin=subprocess.DEVNULL" not in src


def test_claude_code_prompt_va_por_stdin_no_como_argumento(monkeypatch):
    """Un system prompt de Director puede empezar con "---" (frontmatter). Como
    argumento posicional el CLI lo interpreta como opcion ("unknown option");
    por stdin llega intacto, "--" incluido."""

    captured = {"cmd": None, "input": None}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(
            cmd, 0, stdout='{"is_error":false,"result":"[]"}', stderr=""
        )

    monkeypatch.setattr(cli_subscription, "_resolve_executable", lambda name: "claude")
    monkeypatch.setattr(cli_subscription.subprocess, "run", fake_run)

    raw = ClaudeCodeDetector()._complete("---\nfrontmatter: si\n---", "usuario")

    assert raw == "[]"
    cmd = captured["cmd"]
    assert cmd[1] == "-p", "transporte headless -p se mantiene"
    # El prompt no viaja como argumento: nada en cmd empieza con '---'
    assert all(not str(a).startswith("---") for a in cmd[1:])
    assert "--model" in cmd and "--output-format" in cmd
    assert "json" in cmd
    prompt = captured["input"]
    assert prompt.startswith("---"), "stdin transporta el prompt sin escapar ni alterar"
    assert "usuario" in prompt


def test_claude_code_prompt_unicode_y_error(monkeypatch):
    """UTF-8 español intacto por stdin; exit != 0 eleva _CliError legible."""
    captured = {"input": None}

    def fake_run(cmd, **kwargs):
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(
            cmd, 0, stdout='{"is_error":false,"result":"Puñalada íntima: ñandú"}', stderr=""
        )

    monkeypatch.setattr(cli_subscription, "_resolve_executable", lambda name: "claude")
    monkeypatch.setattr(cli_subscription.subprocess, "run", fake_run)

    raw = ClaudeCodeDetector()._complete("Título con ñ", "¿Qué celebración?")
    assert "Puñalada íntima" in raw
    assert "Título con ñ" in captured["input"]
    assert "¿Qué celebración?" in captured["input"]

    def failing_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom claude")

    monkeypatch.setattr(cli_subscription.subprocess, "run", failing_run)
    with pytest.raises(_CliError, match="boom claude"):
        ClaudeCodeDetector()._complete("---", "user")


def test_codex_generic_complete_no_fuerza_clips(monkeypatch, tmp_path):
    """_complete genérico: devuelve el JSON del caller (p.ej. Director) tal cual,
    SIN --output-schema de detection y SIN requisito {"clips": [...]} en el prompt."""
    captured = {"cmd": None, "input": None}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        from pathlib import Path

        # Respuesta tipo Director (editorial_plan), válida para su parser propio.
        Path(cmd[cmd.index("-o") + 1]).write_text(
            '{"type":"editorial_plan","editorial_plan":{}}',
            encoding="utf-8",
        )
        assert cmd[-1] == "-"
        assert "--ephemeral" in cmd
        assert "--ignore-rules" in cmd
        assert "--skip-git-repo-check" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cli_subscription, "resolve_provider_cli", lambda provider: "codex.exe")
    monkeypatch.setattr(cli_subscription.subprocess, "run", fake_run)

    raw = CodexDetector()._complete("system del Director", "user con evidencia")

    assert raw == '{"type":"editorial_plan","editorial_plan":{}}'
    assert "--output-schema" not in captured["cmd"], "completion genérica no usa el schema de clips"
    prompt = captured["input"]
    assert "system del Director" in prompt and "user con evidencia" in prompt
    assert '"clips"' not in prompt, "el prompt genérico no fuerza el contrato de detection"


def test_codex_detection_conserva_schema_de_clips(monkeypatch):
    """_complete_detection sí usa _CODEX_DETECTION_SCHEMA y exige {"clips": [...]}."""
    captured = {"cmd": None, "input": None}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        from pathlib import Path

        Path(cmd[cmd.index("-o") + 1]).write_text(
            '{"clips":[{"start":2,"end":45,"title":"y","hook":"h",'
            '"score":0.8,"reason":"r","hashtags":[],"description":"d"}]}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cli_subscription, "resolve_provider_cli", lambda provider: "codex.exe")
    monkeypatch.setattr(cli_subscription.subprocess, "run", fake_run)

    out = CodexDetector()._complete_detection("system", "user")

    assert "--output-schema" in captured["cmd"]
    assert '"clips"' in captured["input"], "detection sigue pidiendo el objeto clips"
    clips = json.loads(out)
    assert isinstance(clips, list) and clips[0]["start"] == 2


def test_codex_generic_complete_alimenta_metadata(monkeypatch):
    """Metadata usa la completion genérica: su schema {items:[...]} no pasa por clips."""
    from pathlib import Path

    from app.editor.metadata_generator import generate_metadata_for_shorts

    captured = {"cmd": None}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[cmd.index("-o") + 1]).write_text(
            json.dumps(
                {
                    "items": [
                        {
                            "content_id": "s1",
                            "content_type": "short",
                            "source_event_ids": ["ev_0"],
                            "title": "Título generado",
                            "description": "Descripción",
                            "hashtags": ["#test"],
                            "keywords": ["test"],
                            "language": "es",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cli_subscription, "resolve_provider_cli", lambda provider: "codex.exe")
    monkeypatch.setattr(cli_subscription.subprocess, "run", fake_run)

    plan = {
        "project_id": "p1",
        "shorts": [
            {
                "short_id": "s1",
                "event_id": "ev_0",
                "title": "T",
                "hook": "h",
                "reason": "r",
                "director_reason": "d",
            }
        ],
    }
    batch = generate_metadata_for_shorts(plan, CodexDetector()._complete)

    assert "--output-schema" not in captured["cmd"]
    assert batch.items[0].title == "Título generado"
    assert batch.items[0].content_id == "s1"


@pytest.mark.skipif(
    os.name != "nt",
    # monkeypatch de os.name tambien engana a pathlib: fuera de Windows el test
    # construye un WindowsPath y Python 3.14 lo rechaza (UnsupportedOperation).
    reason="las rutas nativas de Codex solo existen en Windows",
)
def test_codex_windows_prefiere_el_binario_nativo(monkeypatch, tmp_path):
    """El wrapper npm .cmd altera argumentos de imagen; el paquete trae un .exe."""
    import app.providers.cli_subscription as mod

    wrapper = tmp_path / "codex.cmd"
    wrapper.write_text("@echo off", encoding="utf-8")
    native = (
        tmp_path
        / "node_modules"
        / "@openai"
        / "codex"
        / "node_modules"
        / "@openai"
        / "codex-win32-x64"
        / "vendor"
        / "x86_64-pc-windows-msvc"
        / "bin"
        / "codex.exe"
    )
    native.parent.mkdir(parents=True)
    native.write_bytes(b"exe")
    monkeypatch.setattr(mod.os, "name", "nt")

    assert mod._prefer_native_codex(str(wrapper)) == str(native)
