#!/usr/bin/env python3
"""Verificación estática del bundle Snipra.app (sin ejecutar la UI).

Comprueba: Info.plist, estructura, arquitectura arm64, dependencias dinámicas,
rutas de desarrollo, archivos prohibidos (secretos, bases, multimedia, Windows),
capacidades reales de FFmpeg empaquetado e importación de dependencias Python.
Escribe un informe JSON y termina con código != 0 si algún check FAIL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SYSTEM_PREFIXES = ("/System/", "/usr/lib/")
RELOCATABLE_PREFIXES = ("@rpath/", "@loader_path/", "@executable_path/")
# Install names de los wheels de PyAV/CTranslate2: se resuelven porque ya están cargadas
# las dylibs con el mismo nombre (verificado por importación en el runtime empaquetado).
WHEEL_PLACEHOLDER_PREFIXES = ("/DLC/",)
REQUIRED_FILTERS = [
    "subtitles", "drawtext", "zscale", "tonemap", "vidstabdetect", "vidstabtransform",
    "deshake", "hqdn3d", "nlmeans", "afftdn", "anlmdn", "loudnorm", "highpass",
    "lowpass", "acompressor", "alimiter",
]
FORBIDDEN_NAMES = re.compile(
    r"(^\.env$)|(^\.env\.(?!example$))|(^cookies\.(txt|sqlite)$)|(\.cookies\.)|(^Cookies(-journal)?$)"
    r"|(\.(db|sqlite3?|db-wal|db-shm)$)|(\.(mp4|mov|mkv|webm|mp3|wav|m4a|flac|opus)$)"
    r"|(\.(exe|bat|ps1|lnk)$)|(^\.git$)|(node_modules)|(\.venv)|(^storage$)|(^\.DS_Store$)"
    r"|(^(app_settings|mediashorts)\.)",
    re.IGNORECASE,
)
# Árbol del código propio. El Python integrado se revisa aparte (contiene su stdlib y wheels).
OWN_TREE_DIRS = ["Contents/Resources/app", "Contents/Resources/models", "Contents/Resources/fonts",
                 "Contents/Helpers", "Contents/Frameworks", "Contents/MacOS"]
SECRET_PATTERNS = re.compile(
    r"(sk-ant-[A-Za-z0-9_\-]{20,})|(sk-[A-Za-z0-9]{32,})|(nvapi-[A-Za-z0-9_\-]{20,})"
    r"|(AKIA[0-9A-Z]{16})|(-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)
TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".js", ".html", ".css", ".env", ".example", ".conf", ".toml"}


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append({"name": name, "status": status, "detail": detail})

    @property
    def failed(self) -> bool:
        return any(c["status"] == "FAIL" for c in self.checks)


def run(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def is_macho(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) in {b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe"}
    except OSError:
        return False


def macho_files(app: Path) -> list[Path]:
    found = []
    for dirpath, _dirs, files in os.walk(app):
        for name in files:
            path = Path(dirpath) / name
            if not path.is_symlink() and is_macho(path):
                found.append(path)
    return found


def deps_of(path: Path) -> list[str]:
    """Dependencias reales; excluye el install name propio que `otool -L` lista en las dylibs."""
    own_lines = run(["otool", "-D", str(path)]).stdout.splitlines()
    own = own_lines[1].strip() if len(own_lines) > 1 else None
    out = run(["otool", "-L", str(path)]).stdout.splitlines()[1:]
    return [m.group(1) for line in out if (m := re.match(r"^\s+(\S+) \(", line)) and m.group(1) != own]


def rpaths_of(path: Path) -> list[str]:
    lines = run(["otool", "-l", str(path)]).stdout.splitlines()
    out = []
    for index, line in enumerate(lines):
        if line.strip() == "cmd LC_RPATH":
            for follow in lines[index + 1 : index + 4]:
                m = re.match(r"^\s+path (.+) \(offset \d+\)$", follow)
                if m:
                    out.append(m.group(1))
                    break
    return out


def minos_of(path: Path) -> str | None:
    out = run(["vtool", "-show-build", str(path)]).stdout
    match = re.search(r"minos (\d+(?:\.\d+)*)", out)
    return match.group(1) if match else None


def version_key(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def check_min_os(app: Path, report: Report) -> dict:
    """Versión mínima técnica: máximo `minos` de todos los Mach-O del bundle."""
    histogram: dict[str, int] = {}
    worst: tuple[tuple[int, ...], str] = ((0,), "")
    for path in macho_files(app):
        value = minos_of(path)
        if value is None:
            continue
        histogram[value] = histogram.get(value, 0) + 1
        if version_key(value) > worst[0]:
            worst = (version_key(value), str(path.relative_to(app)))
    technical = ".".join(str(part) for part in worst[0]) if worst[1] else "desconocido"
    plist = app / "Contents/Info.plist"
    declared = run(["/usr/libexec/PlistBuddy", "-c", "Print :LSMinimumSystemVersion", str(plist)]).stdout.strip()
    ok = bool(declared) and worst[1] != "" and version_key(declared) >= worst[0]
    report.add("Versión mínima declarada >= mínimo técnico de los binarios", "PASS" if ok else "FAIL",
               f"declarada {declared} · técnica {technical} (fija: {worst[1]})")
    return {"technical_minimum": technical, "technical_minimum_source": worst[1],
            "minos_histogram": histogram, "declared_minimum": declared}


def check_plist(app: Path, expected_version: str, report: Report) -> None:
    plist = app / "Contents/Info.plist"
    lint = run(["plutil", "-lint", str(plist)])
    report.add("Info.plist válido", "PASS" if lint.returncode == 0 else "FAIL", lint.stdout.strip())

    def read(key: str) -> str:
        return run(["/usr/libexec/PlistBuddy", "-c", f"Print :{key}", str(plist)]).stdout.strip()

    version = read("CFBundleShortVersionString")
    report.add("CFBundleShortVersionString", "PASS" if version == expected_version else "FAIL", version)
    build = read("CFBundleVersion")
    report.add("CFBundleVersion", "PASS" if build == "1" else "FAIL", build)
    report.add("CFBundleIdentifier", "PASS" if read("CFBundleIdentifier") == "com.snipra.launcher" else "FAIL",
               read("CFBundleIdentifier"))
    minimum = read("LSMinimumSystemVersion")
    report.add("LSMinimumSystemVersion", "PASS" if minimum else "FAIL", minimum)
    executable = read("CFBundleExecutable")
    report.add("CFBundleExecutable", "PASS" if (app / "Contents/MacOS" / executable).is_file() else "FAIL", executable)


def check_layout(app: Path, report: Report) -> None:
    required = {
        "Lanzador": "Contents/MacOS/Snipra",
        "ffmpeg": "Contents/Helpers/ffmpeg",
        "ffprobe": "Contents/Helpers/ffprobe",
        "Python integrado": "Contents/Resources/python/bin/python3.11",
        "Backend": "Contents/Resources/app/backend/app/main.py",
        "Interfaz compilada": "Contents/Resources/app/frontend/dist/index.html",
        "Skill editorial": "Contents/Resources/app/.agents/skills/snipra-video-director/SKILL.md",
        "Modelo whisper": "Contents/Resources/models/hub/models--Systran--faster-whisper-base/snapshots",
        "Icono Brand V2": "Contents/Resources/snipra.icns",
        "Fuentes fontconfig": "Contents/Resources/fonts/fonts.conf",
        "Lock de dependencias": "Contents/Resources/requirements-lock.txt",
    }
    for label, rel in required.items():
        exists = (app / rel).exists()
        report.add(f"Estructura: {label}", "PASS" if exists else "FAIL", rel)
    for rel in ("Contents/MacOS/Snipra", "Contents/Helpers/ffmpeg", "Contents/Helpers/ffprobe",
                "Contents/Resources/python/bin/python3.11"):
        path = app / rel
        ok = path.is_file() and os.access(path, os.X_OK)
        report.add(f"Permiso ejecutable: {rel}", "PASS" if ok else "FAIL")
    dylibs = list((app / "Contents/Frameworks").glob("*.dylib"))
    report.add("Dylibs de FFmpeg relocalizadas", "PASS" if len(dylibs) >= 90 else "FAIL", f"{len(dylibs)} dylibs")


def check_architectures(app: Path, report: Report) -> None:
    arch_counts: dict[str, int] = {}
    fat, x86_only = [], []
    for path in macho_files(app):
        archs = run(["lipo", "-archs", str(path)]).stdout.split()
        key = " ".join(archs) or "?"
        arch_counts[key] = arch_counts.get(key, 0) + 1
        if "arm64" not in archs:
            x86_only.append(str(path.relative_to(app)))
        if len(archs) > 1:
            fat.append(str(path.relative_to(app)))
    report.add("Binarios arm64 (sin x86_64 exclusivo)", "PASS" if not x86_only else "FAIL",
               f"exclusivos x86_64: {x86_only[:5]}" if x86_only else str(arch_counts))
    report.add("Binarios universales (arm64 + x86_64)", "WARN" if fat else "PASS",
               ", ".join(fat) if fat else "ninguno")


def check_links(app: Path, report: Report) -> None:
    frameworks = app / "Contents/Frameworks"
    bundle_only = [app / "Contents/Helpers/ffmpeg", app / "Contents/Helpers/ffprobe", *frameworks.glob("*.dylib")]
    problems = []
    for path in bundle_only:
        for ref in deps_of(path):
            if ref.startswith(SYSTEM_PREFIXES):
                continue
            if ref.startswith("@rpath/") and (frameworks / os.path.basename(ref)).exists():
                continue
            problems.append(f"{path.name} -> {ref}")
        for rp in rpaths_of(path):
            if rp.startswith("/"):
                problems.append(f"{path.name} LC_RPATH absoluto {rp}")
    report.add("FFmpeg y dylibs: sin rutas de Homebrew/desarrollo", "PASS" if not problems else "FAIL",
               "; ".join(problems[:6]) if problems else f"{len(bundle_only)} binarios revisados")

    wheel_refs, unresolved, third_party = 0, [], []
    for path in macho_files(app):
        if path in bundle_only:
            continue
        for ref in deps_of(path):
            if ref.startswith(SYSTEM_PREFIXES) or ref.startswith(RELOCATABLE_PREFIXES):
                continue
            if ref.startswith(WHEEL_PLACEHOLDER_PREFIXES):
                wheel_refs += 1
                continue
            if ref.startswith("/") and ref.startswith(str(app)):
                unresolved.append(f"{path.relative_to(app)} -> {ref}")
                continue
            # Referencias relativas o de build de terceros: se registran sin bloquear.
            third_party.append(f"{path.relative_to(app)} -> {ref}")
    report.add("Dependencias absolutas del propio bundle", "PASS" if not unresolved else "FAIL",
               "; ".join(unresolved[:6]) if unresolved else "ninguna")
    report.add("Dependencias de wheels de terceros (placeholders y rutas de build)",
               "WARN" if third_party else "PASS",
               (f"{len(third_party)} referencias no relocalizables: " + "; ".join(third_party[:3]))
               if third_party else f"placeholders /DLC: {wheel_refs}")


def check_forbidden(app: Path, report: Report) -> None:
    bad = []
    for rel in OWN_TREE_DIRS:
        for dirpath, dirs, files in os.walk(app / rel):
            for name in dirs + files:
                if FORBIDDEN_NAMES.search(name) and name != ".env.example":
                    bad.append(str((Path(dirpath) / name).relative_to(app)))
    report.add("Sin archivos prohibidos en el código propio (.env, cookies, BD, multimedia, Windows, .git)",
               "PASS" if not bad else "FAIL", ", ".join(bad[:8]) if bad else "limpio")
    site = app / "Contents/Resources/python/lib/python3.11/site-packages"
    gpl_in_process = sorted(
        str(p.relative_to(app)) for pattern in ("libx264*", "libx265*", "libvidstab*", "libxvid*")
        for p in site.rglob(pattern)
    ) + ([str((site / "av/.dylibs").relative_to(app))] if (site / "av/.dylibs").exists() else [])
    report.add("Sin bibliotecas GPL (PyAV/x264/x265) dentro del proceso Python del backend",
               "PASS" if not gpl_in_process else "FAIL", ", ".join(gpl_in_process[:6]) if gpl_in_process else "limpio")
    windows_helpers = [str(p.relative_to(app)) for p in (app / "Contents/Resources/python").rglob("*.exe")]
    report.add("Sin ejecutables de Windows en el Python integrado", "PASS" if not windows_helpers else "FAIL",
               ", ".join(windows_helpers[:4]) if windows_helpers else "limpio")


def check_secrets(app: Path, report: Report) -> None:
    hits = []
    for dirpath, _dirs, files in os.walk(app / "Contents/Resources/app"):
        for name in files:
            path = Path(dirpath) / name
            if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 2_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if SECRET_PATTERNS.search(text):
                hits.append(str(path.relative_to(app)))
    report.add("Sin secretos con formato de clave en el código empaquetado", "PASS" if not hits else "FAIL",
               ", ".join(hits[:5]) if hits else "ninguno")
    env_example = app / "Contents/Resources/app/backend/.env.example"
    if env_example.exists():
        values = re.findall(r"^(ANTHROPIC|OPENAI|NVIDIA)_API_KEY=(.+)$", env_example.read_text(encoding="utf-8"), re.M)
        report.add("backend/.env.example sin claves", "PASS" if not any(v.strip() for _k, v in values) else "FAIL")


def check_ffmpeg(app: Path, report: Report) -> dict:
    env = {"PATH": "/usr/bin:/bin", "FONTCONFIG_FILE": str(app / "Contents/Resources/fonts/fonts.conf")}
    ffmpeg = app / "Contents/Helpers/ffmpeg"
    version = run([str(ffmpeg), "-hide_banner", "-version"], env=env)
    first = version.stdout.splitlines()[0] if version.stdout else ""
    report.add("ffmpeg empaquetado ejecuta sin Homebrew", "PASS" if version.returncode == 0 else "FAIL", first)
    filters = run([str(ffmpeg), "-hide_banner", "-filters"], env=env).stdout
    available = {line.split()[1] for line in filters.splitlines() if len(line.split()) >= 3 and line.split()[0][:1] in {".", "T", "S", "A", "V", "C"} and line.split()[1].isidentifier()}
    missing = [f for f in REQUIRED_FILTERS if f not in available]
    report.add("FFmpeg empaquetado: filtros requeridos", "PASS" if not missing else "FAIL",
               f"faltan: {missing}" if missing else f"{len(REQUIRED_FILTERS)}/{len(REQUIRED_FILTERS)} presentes")
    encoders = run([str(ffmpeg), "-hide_banner", "-encoders"], env=env).stdout
    for enc in ("libx264", "libx265", "aac", "libmp3lame", "libopus", "prores_ks"):
        ok = re.search(rf"\s{re.escape(enc)}\s", encoders) is not None
        report.add(f"Encoder empaquetado: {enc}", "PASS" if ok else "WARN")
    return {"version": first}


def check_python(app: Path, report: Report) -> None:
    py = app / "Contents/Resources/python/bin/python3.11"
    # STORAGE_DIR temporal: importar el backend crea directorios de datos y no deben quedar en el bundle.
    scratch = tempfile.mkdtemp(prefix="snipra-check-")
    env = {"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": str(app / "Contents/Resources/app/backend"), "STORAGE_DIR": scratch}
    code = ("import fastapi, uvicorn, sqlmodel, pydantic_settings, faster_whisper, ctranslate2, "
            "yt_dlp, curl_cffi, httpx; import app.main, app.worker.runner; print('ok')")
    result = run([str(py), "-c", code], env=env)
    report.add("Python empaquetado importa dependencias y backend", "PASS" if result.stdout.strip() == "ok" else "FAIL",
               result.stderr.strip()[-300:] if result.returncode else "ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("app", type=Path)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    app = args.app.resolve()
    report = Report()
    check_plist(app, args.expected_version, report)
    check_layout(app, report)
    check_architectures(app, report)
    check_links(app, report)
    check_forbidden(app, report)
    check_secrets(app, report)
    extra = {"ffmpeg": check_ffmpeg(app, report), "min_os": check_min_os(app, report)}
    check_python(app, report)
    payload = {"app": app.name, "expected_version": args.expected_version, "checks": report.checks, "extra": extra,
               "failed": report.failed}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for check in report.checks:
        print(f"[{check['status']}] {check['name']}" + (f" — {check['detail']}" if check["detail"] else ""))
    print("RESULTADO:", "FAIL" if report.failed else "PASS")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
