#!/usr/bin/env python3
"""Genera MANIFEST.json del paquete macOS: versión, componentes, hashes y estado de firma."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    return result.stdout.strip()


def licenses_summary(licenses_dir: Path) -> dict:
    inventory = licenses_dir / "inventory.json"
    if not inventory.exists():
        return {"status": "missing"}
    data = json.loads(inventory.read_text(encoding="utf-8"))
    components = data["components"]
    review = [c["id"] for c in components if c.get("review")]
    return {
        "inventory": "Contents/Resources/licenses/inventory.json",
        "notices": "LICENSES/THIRD_PARTY_NOTICES.md",
        "source_offer": "LICENSES/SOURCES.md",
        "components": len(components),
        "ffmpeg_effective_license": data["ffmpeg"]["effective_license"],
        "legal_review_required": review,
        "legal_review_status": "LEGAL_REVIEW_REQUIRED",
    }


def ffmpeg_source(bottles: Path | None) -> dict:
    if bottles is None or not bottles.exists():
        return {"kind": "local-homebrew-keg"}
    data = json.loads(bottles.read_text(encoding="utf-8"))
    ff = next((b for b in data["bottles"] if b["formula"] == "ffmpeg-full"), {})
    return {
        "kind": "homebrew-bottle",
        "tag": data.get("tag"),
        "ffmpeg_full_version": ff.get("version"),
        "ffmpeg_full_sha256": ff.get("sha256"),
        "bottles_count": len(data["bottles"]),
        "bottles_manifest": "Contents/Resources/homebrew-bottles.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--ffmpeg-manifest", required=True, type=Path)
    parser.add_argument("--check", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--signing", default="ad-hoc")
    parser.add_argument("--notarization", default="NOTARIZATION_PENDING")
    parser.add_argument("--build-mode", choices=("private", "public-adhoc", "distribution"), default="private")
    parser.add_argument("--bottles", type=Path, help="bottles.json de fetch-homebrew-bottles.py")
    args = parser.parse_args()

    app = args.app.resolve()
    res = app / "Contents/Resources"
    ffmpeg_manifest = json.loads(args.ffmpeg_manifest.read_text(encoding="utf-8"))
    check = json.loads(args.check.read_text(encoding="utf-8"))
    commit = run(["git", "rev-parse", "HEAD"], cwd=args.root)
    dirty = bool(run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=args.root))
    model_dir = res / "models/hub/models--Systran--faster-whisper-base/snapshots"
    model_snapshot = next(model_dir.iterdir())
    model_bin = model_snapshot / "model.bin"
    ffmpeg_version = run([str(app / "Contents/Helpers/ffmpeg"), "-hide_banner", "-version"],
                         env={"PATH": "/usr/bin:/bin", "FONTCONFIG_FILE": str(res / "fonts/fonts.conf")})
    py_version = run([str(res / "python/bin/python3.11"), "-V"], env={"PATH": "/usr/bin:/bin"})
    requirements = [line for line in (res / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()
                    if line and not line.startswith((" ", "#"))]

    manifest = {
        "product": "Snipra",
        "version": args.version,
        "build": {
            "bundle_version": "1",
            "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": commit,
            "git_worktree_dirty": dirty,
            "host": {
                "macos": run(["sw_vers", "-productVersion"]),
                "build": run(["sw_vers", "-buildVersion"]),
                "arch": run(["uname", "-m"]),
            },
        },
        "platform": {
            "architecture": "arm64",
            "universal": False,
            "declared_minimum_macos": run(["/usr/libexec/PlistBuddy", "-c", "Print :LSMinimumSystemVersion",
                                           str(app / "Contents/Info.plist")]),
            "technical_minimum_macos": check.get("extra", {}).get("min_os", {}).get("technical_minimum", ""),
            "technical_minimum_source": check.get("extra", {}).get("min_os", {}).get("technical_minimum_source", ""),
            "runtime_verified_on": ["27.0.1"],
            "note": ("El mínimo declarado solo se sube cuando hay ejecución real en esa versión. "
                     "El mínimo técnico es el máximo LC_BUILD_VERSION minos de los binarios."),
        },
        "components": {
            "launcher": {"path": "Contents/MacOS/Snipra", "sha256": sha256(app / "Contents/MacOS/Snipra")},
            "python": {"version": py_version, "path": "Contents/Resources/python", "source": "python-build-standalone (uv)"},
            "python_requirements": {"count": len(requirements), "lock": "Contents/Resources/requirements-lock.txt",
                                    "sha256": sha256(res / "requirements-lock.txt")},
            "ffmpeg": {
                "version_line": ffmpeg_version.splitlines()[0] if ffmpeg_version else "",
                "configuration": next((l.strip() for l in ffmpeg_version.splitlines() if l.strip().startswith("configuration:")), ""),
                "binaries": ffmpeg_manifest["binaries"],
                "frameworks_count": len(ffmpeg_manifest["frameworks"]),
                "source_prefix": ffmpeg_manifest["source_prefix"],
            },
            "whisper_model": {
                "repo": "Systran/faster-whisper-base",
                "revision": model_snapshot.name,
                "model_bin_sha256": sha256(model_bin),
                "model_bin_bytes": model_bin.stat().st_size,
            },
            "icon": {"path": "Contents/Resources/snipra.icns", "sha256": sha256(res / "snipra.icns"),
                     "source": "frontend/image/snipra-brand-v2/app-icons (Brand V2)"},
        },
        "signing": {
            "mode": args.signing,
            "build_mode": args.build_mode,
            "distribution_checks_passed": (
                args.build_mode == "distribution" and args.signing == "developer-id"
                and args.notarization == "ACCEPTED"
            ),
            "developer_id": args.signing == "developer-id",
            "hardened_runtime": args.signing in ("developer-id", "apple-development-local-only"),
            "notarization": args.notarization,
            "notarized": args.notarization == "ACCEPTED",
            "distribution_channel": (
                "public-adhoc-not-notarized" if args.build_mode == "public-adhoc"
                else "official-notarized" if args.build_mode == "distribution" else "private-qa"
            ),
            "gatekeeper_note": (
                "Distribución pública NO notarizada (firma ad hoc): macOS mostrará una advertencia y el usuario "
                "debe autorizar la app manualmente en Ajustes del Sistema > Privacidad y seguridad."
                if args.build_mode == "public-adhoc" else
                "Firma ad hoc: Gatekeeper la rechaza en otros Macs. Distribuir exige Developer ID y notarización."
                if args.signing == "ad-hoc" else
                "Firma de desarrollo: solo para pruebas locales de Hardened Runtime; no distribuible."
                if args.signing == "apple-development-local-only" else
                "Firma Developer ID: notarización y engrapado según el campo 'notarization'."
            ),
        },
        "licenses": licenses_summary(res / "licenses"),
        "ffmpeg_source": ffmpeg_source(args.bottles),
        "verification": {
            "failed": check["failed"],
            "checks": check["checks"],
        },
        "release_status": "NOT_PUBLISHED" if args.build_mode != "public-adhoc" else "PUBLIC_ADHOC_CANDIDATE",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifiesto escrito: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
