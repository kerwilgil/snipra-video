"""Detección vía los CLI de Claude Code / Codex, usando la SUSCRIPCIÓN del usuario.

En vez de pegarle a la API con una key (y gastar créditos), spawnea el CLI que ya
está instalado y logueado en la máquina que corre el worker. Así una detección de
clips consume la cuota Pro/Max (Claude) o Plus/Pro (ChatGPT) en vez de créditos de API.

Trade-offs asumidos a propósito (ver context-snipra.md):
- Más lento que la API directa: cada llamada de detección spawnea un proceso.
- Depende del CLI instalado y con sesión válida en esta máquina. Si no lo está,
  el detector falla con un mensaje claro y el pipeline degrada al fallback naive.
- No es una API oficial: si el CLI cambia su salida headless, hay que ajustar aquí.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from app.core.sanitize import sanitize_error_text
from app.providers._llm import run_detection
from app.providers.base import ClipCandidate, ClipDetector, DetectParams

# Timeout generoso por llamada: los CLI arrancan más lento que una request HTTP y
# la detección por chunks hace varias llamadas. No es interactivo, puede tardar.
_CLI_TIMEOUT_SECONDS = 180

_CODEX_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "title": {"type": "string"},
                    "hook": {"type": "string"},
                    "score": {"type": "number"},
                    "reason": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                    "editorial_cards": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {
                                    "type": "string",
                                    "enum": ["contrast", "consequence", "derived"],
                                },
                                "at": {"type": "number"},
                                "duration": {"type": "number"},
                                "kicker": {"type": "string"},
                                "text": {"type": "string"},
                                "secondary": {"type": "string"},
                                "evidence": {"type": "string"},
                            },
                            "required": [
                                "kind",
                                "at",
                                "duration",
                                "kicker",
                                "text",
                                "secondary",
                                "evidence",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "start",
                    "end",
                    "title",
                    "hook",
                    "score",
                    "reason",
                    "hashtags",
                    "description",
                    "editorial_cards",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}


def _resolve_executable(name: str) -> str | None:
    """Ruta al ejecutable del CLI, o None si no está instalado/en PATH.

    OJO: recibe el nombre del BINARIO ("claude", "codex"), no el del proveedor.
    Para resolver desde el nombre del proveedor usar `resolve_provider_cli`.
    """
    return shutil.which(name)


# Nombre del proveedor -> nombre del binario. El binario de claude_code se llama
# "claude", no "claude_code": pasar el nombre del proveedor a shutil.which
# devuelve None y el CLI parece "no instalado" aunque este perfectamente ahi.
PROVIDER_BINARIES = {"claude_code": "claude", "codex": "codex"}


def _prefer_native_codex(executable: str) -> str:
    """Evita el wrapper npm .cmd de Codex en Windows si existe el .exe nativo.

    El wrapper funciona para texto, pero el reenvio de argumentos de imagen (`-i`)
    es fragil en Windows. El paquete oficial incluye el binario nativo junto al
    wrapper; ejecutarlo directamente conserva rutas y prompts exactamente.
    """
    path = Path(executable)
    if os.name != "nt" or path.suffix.lower() == ".exe":
        return executable

    package_root = path.parent / "node_modules" / "@openai" / "codex" / "node_modules" / "@openai"
    for package in ("codex-win32-x64", "codex-win32-arm64"):
        vendor = package_root / package / "vendor"
        if not vendor.is_dir():
            continue
        matches = sorted(vendor.glob("*/bin/codex.exe"))
        if matches:
            return str(matches[0])
    return executable


def resolve_provider_cli(provider: str) -> str | None:
    """Ruta al ejecutable del CLI de un proveedor de suscripcion."""
    binary = PROVIDER_BINARIES.get((provider or "").lower())
    executable = _resolve_executable(binary) if binary else None
    if executable and binary == "codex":
        return _prefer_native_codex(executable)
    return executable


class _CliError(RuntimeError):
    """Error legible de la invocación del CLI (para mostrar en el job).

    Todo mensaje con salida del CLI pasa por sanitize_error_text: el stderr de
    un SDK/CLI puede incluir la propia API key y nunca debe llegar a la UI.
    """


class ClaudeCodeDetector(ClipDetector):
    """Detección con Claude Code CLI (`claude -p`), usando la suscripción Anthropic."""

    name = "claude_code"

    def __init__(self, model: str = "claude-sonnet-5"):
        # el CLI acepta alias (haiku/sonnet/opus) o nombre completo; usamos el
        # nombre completo por defecto para que quede sin ambiguedad de version
        self.model = model or "claude-sonnet-5"

    def _complete(self, system: str, user: str) -> str:
        exe = _resolve_executable("claude")
        if exe is None:
            raise _CliError(
                "El CLI de Claude Code no está instalado o no está en el PATH de la "
                "máquina que corre Snipra. Instálalo y ejecútalo una vez para iniciar sesión."
            )
        # El CLI headless no tiene parámetro 'system' separado: se antepone al prompt.
        # El prompt va por stdin (no como argumento): un system prompt de Director
        # puede empezar con "---" (frontmatter) y el parser de argumentos del CLI
        # lo rechazaba con "unknown option". Con stdin el contenido llega intacto.
        prompt = f"{system}\n\n{user}"
        cmd = [exe, "-p", "--model", self.model, "--output-format", "json"]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                # encoding explicito: con text=True a secas, Python decodifica con
                # el locale del sistema (cp1252 en Windows es-*) y la salida UTF-8
                # del CLI sale como mojibake: "Punalada" llegaba como "PuÃ±alada"
                # y quedaba quemado en el titulo del clip.
                encoding="utf-8",
                errors="replace",
                # `input` crea un pipe, escribe el prompt y cierra stdin: ni espera
                # de entrada adicional (el worker es un proceso oculto sin consola)
                # ni reinterpretación del prompt como opciones de línea de comandos.
                input=prompt,
                timeout=_CLI_TIMEOUT_SECONDS,
                # sin heredar el cwd del repo: evita que el CLI cargue CLAUDE.md ajeno
                cwd=tempfile.gettempdir(),
                env={**os.environ},
            )
        except subprocess.TimeoutExpired as exc:
            raise _CliError("Claude Code CLI no respondió a tiempo (timeout).") from exc
        except OSError as exc:
            raise _CliError(f"No se pudo ejecutar Claude Code CLI: {exc}") from exc

        if proc.returncode != 0:
            raise _CliError(
                f"Claude Code CLI terminó con error ({proc.returncode}): "
                f"{sanitize_error_text((proc.stderr or proc.stdout or '').strip()[:300])}"
            )
        return _extract_claude_result(proc.stdout)

    def detect(self, segments: list[dict], params: DetectParams) -> list[ClipCandidate]:
        return run_detection(self._complete, segments, params)


class CodexDetector(ClipDetector):
    """Detección con Codex CLI (`codex exec`), usando la suscripción de ChatGPT."""

    name = "codex"

    def __init__(self, model: str = "gpt-5.6-terra"):
        # verificado contra el catalogo real de Codex (models_cache.json), no
        # un nombre inventado: gpt-5.6-{luna,terra,sol} son variantes reales
        self.model = model or "gpt-5.6-terra"

    def _run_cli(self, prompt: str, output_schema: dict | None = None) -> str:
        """Ejecuta `codex exec` con `prompt` por stdin y devuelve su respuesta.

        Costo compartido de _complete/_complete_detection: mismo flags, mismo
        manejo de errores. Si se pasa `output_schema` se añade --output-schema
        (solo detección). Devuelve el último mensaje del modelo como texto.
        """
        # Usa el ejecutable nativo en Windows. codex.CMD pasa por cmd.exe, cuya
        # linea de comandos no admite una transcripcion real de varios KB.
        exe = resolve_provider_cli("codex")
        if exe is None:
            raise _CliError(
                "El CLI de Codex no está instalado o no está en el PATH de la máquina "
                "que corre Snipra. Instálalo y ejecútalo una vez para iniciar sesión."
            )
        # -o escribe SOLO el último mensaje del agente (la respuesta) a un archivo,
        # sin el ruido de los eventos intermedios. --skip-git-repo-check porque el
        # cwd temporal no es un repo git.
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "codex-out.txt"
            cmd = [
                exe,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-rules",
                "--model",
                self.model,
            ]
            if output_schema is not None:
                schema_path = Path(tmp) / "codex-schema.json"
                schema_path.write_text(
                    json.dumps(output_schema, ensure_ascii=False),
                    encoding="utf-8",
                )
                cmd.extend(["--output-schema", str(schema_path)])
            cmd.extend(["-o", str(out_path), "-"])
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    # ver nota de encoding en ClaudeCodeDetector._complete
                    encoding="utf-8",
                    errors="replace",
                    # `input` crea un pipe, escribe el prompt y lo cierra. Evita
                    # tanto la espera de entrada adicional como el limite de
                    # longitud de cmd.exe para transcripciones reales.
                    input=prompt,
                    timeout=_CLI_TIMEOUT_SECONDS,
                    cwd=tmp,
                    env={**os.environ},
                )
            except subprocess.TimeoutExpired as exc:
                raise _CliError("Codex CLI no respondió a tiempo (timeout).") from exc
            except OSError as exc:
                raise _CliError(f"No se pudo ejecutar Codex CLI: {exc}") from exc

            if proc.returncode != 0:
                raise _CliError(
                    f"Codex CLI terminó con error ({proc.returncode}): "
                    f"{sanitize_error_text((proc.stderr or '').strip()[:300])}"
                )
            if not out_path.exists():
                raise _CliError("Codex CLI no escribió ninguna respuesta.")
            text = out_path.read_text(encoding="utf-8").strip()
            if not text:
                raise _CliError("Codex CLI devolvio una respuesta vacia.")
            return text

    def _complete(self, system: str, user: str) -> str:
        """Completion genérica: devuelve el último mensaje del modelo tal cual.

        NO impone el schema de clips: Director, metadata y captions traen su
        propio contrato JSON y lo parsean ellos mismos.
        """
        return self._run_cli(f"{system}\n\n{user}")

    def _complete_detection(self, system: str, user: str) -> str:
        """Completion estructurada de detección: exige {"clips": [...]}."""
        prompt = (
            f"{system}\n\n{user}\n\n"
            "REQUISITO DE SALIDA DE CODEX: devuelve un objeto JSON con la forma "
            '{"clips": [...]} y coloca dentro de "clips" el array solicitado.'
        )
        return _extract_codex_result(self._run_cli(prompt, output_schema=_CODEX_DETECTION_SCHEMA))

    def detect(self, segments: list[dict], params: DetectParams) -> list[ClipCandidate]:
        # Un fallo del CLI (plugins, auth, argumentos) es global. Repetir la
        # misma llamada cara no lo corrige y retrasa el fallback del orquestador.
        return run_detection(self._complete_detection, segments, params, retries=0)


def _extract_claude_result(stdout: str) -> str:
    """Saca el texto del modelo del JSON de `claude -p --output-format json`.

    El envelope trae la respuesta en `.result`. Si por lo que sea no es JSON válido,
    devuelve el stdout crudo (parse_candidates igual sabe rescatar el array JSON).
    """
    text = (stdout or "").strip()
    if not text:
        raise _CliError("Claude Code CLI no devolvió salida.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(data, dict):
        if data.get("is_error"):
            raise _CliError(
                f"Claude Code CLI reportó error: "
                f"{sanitize_error_text(str(data.get('result', ''))[:300])}"
            )
        result = data.get("result")
        if isinstance(result, str):
            return result
    return text


def _extract_codex_result(stdout: str) -> str:
    """Valida la salida estructurada de Codex y devuelve el array comun."""
    text = (stdout or "").strip()
    if not text:
        raise _CliError("Codex CLI devolvio una respuesta vacia.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _CliError("Codex CLI devolvio JSON invalido.") from exc
    clips = data.get("clips") if isinstance(data, dict) else None
    if not isinstance(clips, list):
        raise _CliError('Codex CLI no devolvio el array requerido en "clips".')
    return json.dumps(clips, ensure_ascii=False)


def subscription_cli_available(provider: str) -> bool:
    """True si el CLI del proveedor de suscripción está instalado (no verifica login)."""
    return bool(resolve_provider_cli(provider))
