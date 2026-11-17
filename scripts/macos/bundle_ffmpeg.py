#!/usr/bin/env python3
"""Empaqueta ffmpeg/ffprobe y sus dylibs dentro de Snipra.app (arm64).

- Copia los binarios a Contents/Helpers y las dylibs no-sistema a Contents/Frameworks,
  con el mismo nombre que referencian los binarios (p. ej. libavcodec.63.dylib).
- Reescribe install names a @rpath, elimina LC_RPATH absolutos (p. ej. /opt/homebrew/lib)
  y deja @executable_path/../Frameworks (binarios) y @loader_path (dylibs).
- Verifica que ninguna referencia escape al bundle salvo /System y /usr/lib.

No firma: la firma ad hoc se aplica después, porque install_name_tool invalida la firma.
Uso: bundle_ffmpeg.py --ffmpeg-prefix DIR --helpers DIR --frameworks DIR --manifest FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SYSTEM_PREFIXES = ("/System/", "/usr/lib/")
LOADER_PREFIXES = ("@rpath/", "@loader_path/", "@executable_path/")
FORBIDDEN_FRAGMENTS = ("/opt/homebrew", "/usr/local", "/Cellar/", "/Users/")


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Falló {' '.join(cmd)}: {result.stderr.strip()}")
    return result.stdout


def own_install_name(path: Path) -> str | None:
    lines = run(["otool", "-D", str(path)]).splitlines()
    return lines[1].strip() if len(lines) > 1 else None


def load_deps(path: Path) -> list[str]:
    """Dependencias reales (LC_LOAD_DYLIB). `otool -L` también lista el install name propio."""
    own = own_install_name(path)
    lines = run(["otool", "-L", str(path)]).splitlines()[1:]
    deps = []
    for line in lines:
        match = re.match(r"^\s+(\S+) \(", line)
        if match and match.group(1) != own:
            deps.append(match.group(1))
    return deps


def load_rpaths(path: Path) -> list[str]:
    rpaths: list[str] = []
    lines = run(["otool", "-l", str(path)]).splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "cmd LC_RPATH":
            for follow in lines[index + 1 : index + 4]:
                match = re.match(r"^\s+path (.+) \(offset \d+\)$", follow)
                if match:
                    rpaths.append(match.group(1))
                    break
    return rpaths


def is_system(ref: str) -> bool:
    return ref.startswith(SYSTEM_PREFIXES)


def is_loader_relative(ref: str) -> bool:
    return ref.startswith(LOADER_PREFIXES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


HOMEBREW_ROOT: Path | None = None  # prefijo de bottles (placeholders @@HOMEBREW_*@@)


def expand(path: str) -> str:
    """Sustituye los placeholders de los bottles de Homebrew por el prefijo descargado."""
    if HOMEBREW_ROOT is None:
        return path
    return (path.replace("@@HOMEBREW_CELLAR@@", str(HOMEBREW_ROOT / "Cellar"))
                .replace("@@HOMEBREW_PREFIX@@", str(HOMEBREW_ROOT)))


def resolve_ref(owner: Path, ref: str) -> Path:
    """Resuelve una referencia de dependencia a un archivo real del build host.

    @rpath se busca en los LC_RPATH del propio binario que la referencia (Homebrew usa
    rutas absolutas o relativas a @loader_path). @loader_path/@executable_path se resuelven
    respecto al directorio de ese binario.
    """
    if ref.startswith("@rpath/"):
        leaf = ref[len("@rpath/"):]
        for rpath in load_rpaths(owner):
            base = expand(rpath).replace("@loader_path", str(owner.parent)).replace(
                "@executable_path", str(owner.parent)
            )
            candidate = Path(base) / leaf
            if candidate.exists():
                return candidate.resolve()
        raise SystemExit(f"@rpath sin resolver en {owner}: {ref} (rpaths: {load_rpaths(owner)})")
    if ref.startswith(("@loader_path/", "@executable_path/")):
        leaf = ref.split("/", 1)[1]
        return (owner.parent / leaf).resolve()
    return Path(expand(ref)).resolve()


def collect_closure(binaries: list[Path]) -> dict[str, Path]:
    """Mapa nombre-referenciado -> archivo real (realpath) de todas las dylibs no-sistema."""
    by_name: dict[str, Path] = {}
    queue = list(binaries)
    visited: set[Path] = set()
    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        for ref in load_deps(current):
            if is_system(ref):
                continue
            name = os.path.basename(ref)
            source = resolve_ref(current, ref)
            if not source.exists():
                raise SystemExit(f"Dependencia inexistente en el build host: {ref} (de {current})")
            if name in by_name and by_name[name] != source:
                raise SystemExit(
                    f"Conflicto de nombre '{name}': {by_name[name]} vs {source}"
                )
            if name not in by_name:
                by_name[name] = source
                queue.append(source)
    return by_name


def rewrite(path: Path, id_name: str | None, rpath: str) -> None:
    """Aplica install_name_tool en una sola pasada: id, dependencias, rpaths."""
    args = ["install_name_tool"]
    if id_name is not None:
        args += ["-id", f"@rpath/{id_name}"]
    for ref in load_deps(path):
        if is_system(ref):
            continue
        args += ["-change", ref, f"@rpath/{os.path.basename(ref)}"]
    for old in load_rpaths(path):
        args += ["-delete_rpath", old]
    args += ["-add_rpath", rpath, str(path)]
    run(args)


def verify(path: Path, frameworks: Path) -> list[str]:
    problems = []
    for ref in load_deps(path):
        if is_system(ref):
            continue
        if not ref.startswith("@rpath/"):
            problems.append(f"{path.name}: referencia no relocada {ref}")
            continue
        if not (frameworks / os.path.basename(ref)).exists():
            problems.append(f"{path.name}: falta {ref} dentro de Frameworks")
    for rpath in load_rpaths(path):
        if rpath.startswith("/") or any(f in rpath for f in FORBIDDEN_FRAGMENTS):
            problems.append(f"{path.name}: LC_RPATH absoluto {rpath}")
    if any(f in " ".join(load_deps(path)) for f in FORBIDDEN_FRAGMENTS):
        problems.append(f"{path.name}: referencia a ruta de desarrollo")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ffmpeg-prefix", required=True, type=Path)
    parser.add_argument("--helpers", required=True, type=Path)
    parser.add_argument("--frameworks", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--homebrew-root", type=Path,
                        help="Prefijo de bottles descargados (resuelve @@HOMEBREW_PREFIX@@/@@HOMEBREW_CELLAR@@)")
    args = parser.parse_args()
    global HOMEBREW_ROOT
    HOMEBREW_ROOT = args.homebrew_root.resolve() if args.homebrew_root else None

    ffmpeg_bin = (args.ffmpeg_prefix / "bin" / "ffmpeg").resolve()
    ffprobe_bin = (args.ffmpeg_prefix / "bin" / "ffprobe").resolve()
    for binary in (ffmpeg_bin, ffprobe_bin):
        if not binary.exists():
            raise SystemExit(f"No existe {binary}")

    args.helpers.mkdir(parents=True, exist_ok=True)
    args.frameworks.mkdir(parents=True, exist_ok=True)

    closure = collect_closure([ffmpeg_bin, ffprobe_bin])
    print(f"dylibs no-sistema: {len(closure)}")

    copied_libs = []
    for name, source in sorted(closure.items()):
        target = args.frameworks / name
        shutil.copy2(source, target)
        target.chmod(0o755)
        copied_libs.append((name, target))

    binaries = []
    for name, source in (("ffmpeg", ffmpeg_bin), ("ffprobe", ffprobe_bin)):
        target = args.helpers / name
        shutil.copy2(source, target)
        target.chmod(0o755)
        binaries.append(target)

    for name, target in copied_libs:
        rewrite(target, id_name=name, rpath="@loader_path")
    for target in binaries:
        rewrite(target, id_name=None, rpath="@executable_path/../Frameworks")

    problems: list[str] = []
    for target in binaries:
        problems += verify(target, args.frameworks)
    for _, target in copied_libs:
        problems += verify(target, args.frameworks)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1

    manifest = {
        "source_prefix": str(args.ffmpeg_prefix.resolve()),
        "binaries": {
            target.name: {"sha256": sha256(target), "source": str(source)}
            for target, source in zip(binaries, (ffmpeg_bin, ffprobe_bin))
        },
        "frameworks": {
            name: {"source": str(closure[name]), "sha256": sha256(target)}
            for name, target in copied_libs
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"OK: {len(binaries)} binarios, {len(copied_libs)} dylibs relocadas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
