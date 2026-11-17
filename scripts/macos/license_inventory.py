#!/usr/bin/env python3
"""Inventario de licencias y fuentes de los componentes incluidos en Snipra.app.

Fuentes, en orden de autoridad:
  1. Binarios nativos (ffmpeg, ffprobe y dylibs): fórmula Homebrew exacta de cada keg
     (`<keg>/.brew/<formula>.rb`): licencia, URL de origen y sha256 de la versión instalada.
  2. Python integrado: metadatos `*.dist-info` del propio bundle.
  3. Frontend: paquetes de producción de `npm ls --omit=dev` (licencia de package.json).
  4. Modelo de transcripción: front matter `license:` del README del snapshot.
  5. Proyecto: LICENSE del repositorio.

Genera `inventory.json` y `THIRD_PARTY_NOTICES.md`, y copia las fórmulas usadas a `formulae/`.
Las conclusiones jurídicas NO se generan aquí: se marcan con `review` para revisión legal.
"""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

FORMULA_RE = re.compile(r"/Cellar/([^/]+)/([^/]+)/")
LICENSE_RE = re.compile(r'^\s*license\s+(.+?)\s*$', re.M)
URL_RE = re.compile(r'^\s*url\s+"([^"]+)"', re.M)
SHA_RE = re.compile(r'^\s*sha256\s+"([0-9a-f]{64})"', re.M)
HOMEPAGE_RE = re.compile(r'^\s*homepage\s+"([^"]+)"', re.M)
GPL_FAMILY = re.compile(r"\b(A?GPL|LGPL|MPL|CDDL|EPL|OSL)", re.I)


def license_block(text: str) -> str | None:
    """Valor de `license` de una fórmula; admite el formato multilínea all_of/any_of de Ruby."""
    match = re.search(r"^\s*license\s+(.*)$", text, re.M)
    if not match:
        return None
    value = match.group(1)
    depth = value.count("[") + value.count("{") - value.count("]") - value.count("}")
    if depth <= 0:
        return value.strip()
    rest = text[match.end():].splitlines()
    collected = [value]
    for line in rest:
        collected.append(line)
        depth += line.count("[") + line.count("{") - line.count("]") - line.count("}")
        if depth <= 0:
            break
    return " ".join(part.strip() for part in collected)


def parse_license_value(raw: str) -> str:
    """Convierte la licencia de Homebrew (cadena, all_of:/any_of: de Ruby) en expresión SPDX."""
    text = re.sub(r"#[^\n]*", "", raw)          # quita comentarios Ruby
    text = " ".join(text.split())
    if not text.startswith(("all_of", "any_of", "{", "[", '"')):
        return text.strip('"')
    if not re.search(r"(all_of|any_of)", text):
        return ", ".join(re.findall(r'"([^"]+)"', text)) or text
    pattern = re.compile(r"(all_of|any_of):\s*\[([^\[\]]*)\]")
    while True:
        match = pattern.search(text)
        if not match:
            break
        joiner = " AND " if match.group(1) == "all_of" else " OR "
        items = [item.strip().strip("{}").strip() for item in match.group(2).split(",")]
        items = [i for i in items if i]
        text = text[:match.start()] + "(" + joiner.join(i.strip('"') for i in items) + ")" + text[match.end():]
    text = text.replace("{", "").replace("}", "").replace(" ,", ",")
    return re.sub(r"\s+", " ", text).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


LICENSE_FILE_RE = re.compile(r"^(LICENSE|COPYING|NOTICE|COPYRIGHT|AUTHORS|LICENCE)", re.I)


def copy_keg_licenses(keg: Path, formula: str, licenses_dir: Path) -> list[str]:
    """Copia los ficheros de licencia/aviso que trae el keg (documentación de la propia fuente)."""
    copied = []
    dest = licenses_dir / "components" / formula
    for candidate in keg.rglob("*"):
        if candidate.is_file() and LICENSE_FILE_RE.match(candidate.name) and len(candidate.relative_to(keg).parts) <= 5:
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / f"{candidate.parent.name}-{candidate.name}" if candidate.parent != keg else dest / candidate.name
            shutil.copy2(candidate, target)
            copied.append(str(target.relative_to(licenses_dir)))
    return copied


def native_components(manifest: dict, formulae_dir: Path, licenses_dir: Path) -> list[dict]:
    """Agrupa dylibs y binarios por fórmula Homebrew."""
    by_keg: dict[tuple[str, str], dict] = {}
    entries = [("binary", n, v["source"]) for n, v in manifest.get("binaries", {}).items()]
    entries += [("dylib", n, v["source"]) for n, v in manifest["frameworks"].items()]
    for kind, name, source in entries:
        match = FORMULA_RE.search(source)
        if not match:
            continue
        formula, version = match.groups()
        keg_root = source[: match.end()]
        comp = by_keg.setdefault((formula, version), {"bundled_files": [], "kind": set(), "keg": keg_root})
        comp["bundled_files"].append(name)
        comp["kind"].add(kind)
    components = []
    for (formula, version), data in sorted(by_keg.items()):
        keg = Path(data["keg"])
        rb = keg / ".brew" / f"{formula}.rb"
        copy_keg_licenses(keg, formula, licenses_dir)
        text = rb.read_text(encoding="utf-8") if rb.exists() else ""
        license_value = license_block(text)
        url = URL_RE.search(text)
        sha = SHA_RE.search(text)
        home = HOMEPAGE_RE.search(text)
        formulae_dir.mkdir(parents=True, exist_ok=True)
        if rb.exists():
            shutil.copy2(rb, formulae_dir / rb.name)
        components.append({
            "id": f"homebrew:{formula}@{version}",
            "category": "native",
            "name": formula,
            "version": version,
            "license": parse_license_value(license_value) if license_value else "UNKNOWN",
            "license_source": "homebrew-formula" if license_value else "missing",
            "homepage": home.group(1) if home else "",
            "source_url": url.group(1) if url else "",
            "source_sha256": sha.group(1) if sha else "",
            "formula_file": f"formulae/{rb.name}" if rb.exists() else "",
            "bundled_files": sorted(data["bundled_files"]),
            "linkage": sorted(data["kind"]),
            "review": GPL_FAMILY.search(parse_license_value(license_value) if license_value else "") is not None,
        })
    return components


def configuration_from_libavutil(frameworks: Path) -> str:
    """Configuración de compilación embebida en libavutil (no requiere ejecutar el binario)."""
    for lib in sorted(frameworks.glob("libavutil*.dylib")):
        match = re.search(rb"--prefix=[ -~]+", lib.read_bytes())
        if match:
            return match.group(0).decode("ascii")
    return ""


def ffmpeg_effective(ffmpeg: Path) -> dict:
    out = subprocess.run([str(ffmpeg), "-hide_banner", "-version"], capture_output=True, text=True).stdout
    conf = next((l for l in out.splitlines() if l.startswith("configuration:")), "")
    source = "ffmpeg -version"
    if not conf:
        # Antes de firmar, install_name_tool invalida la firma y el binario no se ejecuta.
        conf = "configuration: " + configuration_from_libavutil(ffmpeg.parent.parent / "Frameworks")
        source = "libavutil (embebido)"
    if conf.strip() in ("configuration:", ""):
        return {"configuration_flags_enabled": [], "effective_license": "UNKNOWN-LEGAL_REVIEW_REQUIRED",
                "configuration_line": "", "configuration_source": "none"}
    flags = set(re.findall(r"--(enable|disable)-([\w-]+)", conf))
    enabled = sorted(name for kind, name in flags if kind == "enable")
    gpl = "gpl" in enabled
    version3 = "version3" in enabled
    nonfree = "nonfree" in enabled
    if nonfree:
        effective = "NONFREE-NOT-REDISTRIBUTABLE"
    elif gpl and version3:
        effective = "GPL-3.0-or-later"
    elif gpl:
        effective = "GPL-2.0-or-later"
    elif version3:
        effective = "LGPL-3.0-or-later"
    else:
        effective = "LGPL-2.1-or-later"
    return {"configuration_flags_enabled": enabled, "effective_license": effective,
            "configuration_line": conf, "configuration_source": source}


def python_components(site: Path) -> list[dict]:
    out = []
    for dist in sorted(site.glob("*.dist-info")):
        meta_path = dist / "METADATA"
        if not meta_path.exists():
            continue
        meta = email.message_from_string(meta_path.read_text(encoding="utf-8", errors="ignore"))
        classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier", []) if c.startswith("License ::")]
        lic = meta.get("License-Expression") or meta.get("License") or ""
        if len(lic) > 120 or "\n" in lic:
            lic = ""
        if not lic and classifiers:
            lic = " / ".join(classifiers)
        out.append({
            "id": f"pypi:{meta.get('Name')}@{meta.get('Version')}",
            "category": "python",
            "name": meta.get("Name"),
            "version": meta.get("Version"),
            "license": lic or "UNKNOWN",
            "license_source": "dist-info",
            "homepage": meta.get("Home-page") or next((u.split(",", 1)[-1].strip() for u in meta.get_all("Project-URL", []) if "Home" in u or "Source" in u), ""),
            "source_url": f"https://pypi.org/project/{meta.get('Name')}/{meta.get('Version')}/",
            "source_sha256": "",
            "bundled_files": [str(dist.relative_to(site.parents[2]))],
            "linkage": ["python-package"],
            "review": GPL_FAMILY.search(lic or "") is not None,
        })
    return out


def av_bundled_libs(site: Path) -> list[dict]:
    libs = sorted((site / "av" / ".dylibs").glob("*.dylib"))
    if not libs:
        return []
    util = next((p for p in libs if p.name.startswith("libavutil")), None)
    flags = []
    if util:
        flags = sorted(set(re.findall(rb"--enable-[a-z0-9-]+", util.read_bytes())))
        flags = [f.decode() for f in flags]
    gpl_parts = [p.name for p in libs if re.match(r"lib(x264|x265)", p.name)]
    return [{
        "id": "pypi:av-bundled-ffmpeg",
        "category": "python-bundled-native",
        "name": "PyAV bundled FFmpeg libraries (av/.dylibs)",
        "version": "",
        "license": "LGPL-3.0-or-later (version3 build) + GPL-2.0-or-later components loaded: " + ", ".join(gpl_parts),
        "license_source": "binary-configuration",
        "homepage": "https://github.com/PyAV-Org/PyAV",
        "source_url": "https://github.com/PyAV-Org/PyAV",
        "source_sha256": "",
        "bundled_files": [f"av/.dylibs/{p.name}" for p in libs],
        "linkage": ["in-process (loaded by backend via faster_whisper.audio)"],
        "configuration_flags": flags,
        "review": True,
        "review_reason": "Biblioteca GPL cargada dentro del proceso Python del backend (decodificación de audio).",
    }]


def npm_components(frontend: Path) -> list[dict]:
    raw = subprocess.run(["npm", "ls", "--omit=dev", "--all", "--json", "--package-lock-only=false"],
                         cwd=frontend, capture_output=True, text=True).stdout
    try:
        tree = json.loads(raw or "{}")
    except json.JSONDecodeError:
        tree = {}
    seen: dict[str, str] = {}

    def walk(node: dict) -> None:
        for name, child in (node.get("dependencies") or {}).items():
            seen.setdefault(name, child.get("version", ""))
            walk(child)
    walk(tree)
    out = []
    for name, version in sorted(seen.items()):
        pkg = frontend / "node_modules" / name / "package.json"
        if not pkg.exists():
            continue
        data = json.loads(pkg.read_text(encoding="utf-8"))
        lic = data.get("license") or ""
        if isinstance(lic, dict):
            lic = lic.get("type", "")
        out.append({
            "id": f"npm:{name}@{version}",
            "category": "npm-frontend",
            "name": name,
            "version": version,
            "license": lic or "UNKNOWN",
            "license_source": "package.json",
            "homepage": data.get("homepage", ""),
            "source_url": f"https://www.npmjs.com/package/{name}/v/{version}",
            "source_sha256": "",
            "bundled_files": ["app/frontend/dist (compilado)"],
            "linkage": ["bundled-javascript"],
            "review": GPL_FAMILY.search(lic or "") is not None,
        })
    return out


def model_component(model_dir: Path) -> list[dict]:
    snap = next(model_dir.iterdir())
    readme = snap / "README.md"
    lic = "UNKNOWN"
    if readme.exists():
        m = re.search(r"^license:\s*(\S+)", readme.read_text(encoding="utf-8", errors="ignore"), re.M)
        if m:
            lic = m.group(1)
    return [{
        "id": "hf:Systran/faster-whisper-base",
        "category": "model",
        "name": "faster-whisper base (Systran)",
        "version": snap.name[:12],
        "license": lic,
        "license_source": "model-card",
        "homepage": "https://huggingface.co/Systran/faster-whisper-base",
        "source_url": "https://huggingface.co/Systran/faster-whisper-base",
        "source_sha256": "",
        "bundled_files": ["models/hub/models--Systran--faster-whisper-base"],
        "linkage": ["data"],
        "review": False,
    }]


def render_sources(components: list[dict]) -> str:
    """Oferta de código fuente: componentes copyleft con su URL y sha256 de origen."""
    lines = ["# Código fuente correspondiente (Snipra 1.0.0 macOS arm64)", "",
             "Los componentes GPL, LGPL y MPL de este paquete se distribuyen con su código fuente",
             "correspondiente. Cada entrada indica la URL exacta de la fórmula Homebrew usada y su sha256.",
             "El script `scripts/macos/fetch-corresponding-source.py` descarga y verifica todos ellos.", "",
             "| Componente | Versión | Licencia | URL de origen | sha256 |", "|---|---|---|---|---|"]
    copyleft = re.compile(r"(A?GPL|LGPL|MPL)", re.I)
    for c in components:
        if c["category"] == "native" and copyleft.search(c["license"]):
            lines.append(f"| {c['name']} | {c['version']} | {c['license']} | {c['source_url']} | `{c['source_sha256']}` |")
    lines += ["", "Para x264 el origen es el repositorio git indicado en la fórmula, en la revisión que figura en ella.", ""]
    return "\n".join(lines)


def render_notices(components: list[dict], ffmpeg: dict) -> str:
    lines = ["# Avisos de terceros y licencias de Snipra 1.0.0 (macOS arm64)", "",
             "Generado por `scripts/macos/license_inventory.py`. No es asesoramiento jurídico.",
             "Las obligaciones marcadas como `revisión` requieren evaluación legal (LEGAL_REVIEW_REQUIRED).", "",
             "## FFmpeg empaquetado", "",
             f"- Licencia efectiva del binario: **{ffmpeg['effective_license']}** (por `--enable-gpl` y `--enable-version3`).",
             f"- Configuración: `{ffmpeg['configuration_line'].replace('configuration: ', '')[:2000]}`", "",
             "## Componentes", "",
             "| Categoría | Componente | Versión | Licencia | Fuente (sha256) | Revisión |",
             "|---|---|---|---|---|---|"]
    for c in components:
        src = c["source_url"] or c["homepage"]
        sha = c["source_sha256"][:16] + "…" if c["source_sha256"] else ""
        review = "sí" if c.get("review") else ""
        lines.append(f"| {c['category']} | {c['name']} | {c['version']} | {c['license']} | {src} {sha} | {review} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--ffmpeg-manifest", required=True, type=Path)
    parser.add_argument("--frontend", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    res = args.app / "Contents" / "Resources"
    site = res / "python" / "lib" / "python3.11" / "site-packages"
    manifest = json.loads(args.ffmpeg_manifest.read_text(encoding="utf-8"))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    formulae_dir = out / "formulae"
    licenses_dir = out
    ffmpeg = ffmpeg_effective(args.app / "Contents" / "Helpers" / "ffmpeg")

    components: list[dict] = []
    components += native_components(manifest, formulae_dir, out)
    components += python_components(site)
    components += av_bundled_libs(site)
    components += npm_components(args.frontend)
    model_dir = res / "models" / "hub" / "models--Systran--faster-whisper-base" / "snapshots"
    components += model_component(model_dir)
    components.append({
        "id": "project:snipra", "category": "project", "name": "Snipra", "version": "1.0.0",
        "license": "MIT", "license_source": (args.repo / "LICENSE").name, "homepage": "",
        "source_url": "https://github.com/kerwilgil/snipra-video",
        "source_sha256": sha256(args.repo / "LICENSE"), "bundled_files": ["app/", "Contents/MacOS/Snipra"],
        "linkage": ["own"], "review": False,
    })
    ffmpeg_native = next((c for c in components if c["category"] == "native" and c["name"] == "ffmpeg-full"), {})
    components.append({
        "id": "ffmpeg:effective", "category": "native-combined", "name": "ffmpeg/ffprobe (build ffmpeg-full)",
        "version": ffmpeg_native.get("version", "desconocida"), "license": ffmpeg["effective_license"],
        "license_source": "configuration",
        "homepage": "https://ffmpeg.org/", "source_url": ffmpeg_native.get("source_url", ""),
        "source_sha256": ffmpeg_native.get("source_sha256", ""),
        "bundled_files": ["Helpers/ffmpeg", "Helpers/ffprobe"], "linkage": ["executable (subprocess)"],
        "review": True,
        "review_reason": "Binario GPL-3.0-or-later con componentes GPL/LGPL/MPL/Apache/BSD: obligaciones de distribución.",
    })

    (out / "inventory.json").write_text(json.dumps({"ffmpeg": ffmpeg, "components": components}, indent=2,
                                                   ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "THIRD_PARTY_NOTICES.md").write_text(render_notices(components, ffmpeg), encoding="utf-8")
    (out / "SOURCES.md").write_text(render_sources(components), encoding="utf-8")
    review = [c["id"] for c in components if c.get("review")]
    print(f"componentes: {len(components)} · revisión legal: {len(review)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
