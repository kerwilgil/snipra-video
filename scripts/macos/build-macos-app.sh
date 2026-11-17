#!/usr/bin/env bash
# Construye Snipra.app para macOS arm64, sus avisos de licencia y el ZIP de distribución.
#
# Lo que incluye el bundle (no depende del checkout, de Homebrew instalado ni de Vite en uso):
#   Contents/MacOS/Snipra            lanzador nativo (AppKit, deployment target 13.0)
#   Contents/Helpers/ffmpeg|ffprobe  + Contents/Frameworks/*.dylib (relocalizados con @rpath)
#   Contents/Resources/python        CPython 3.11.16 standalone + dependencias de backend/uv.lock (hashes)
#   Contents/Resources/app           backend/app, frontend/dist, skill editorial, extensión
#   Contents/Resources/models        faster-whisper base fijado a una revisión
#   Contents/Resources/licenses      inventario, avisos, textos de licencia y fórmulas usadas
#
# FFmpeg: por defecto bottles de Homebrew para macOS 15 (arm64_sequoia), descargados y verificados por
# sha256 en un prefijo propio (scripts/macos/fetch-homebrew-bottles.py). Con SNIPRA_FFMPEG_SOURCE=local
# se usa el ffmpeg-full instalado (compilado para macOS 27 en este equipo).
#
# Firma (SNIPRA_CODESIGN_IDENTITY, por defecto ad hoc "-"):
#   "Developer ID Application: ..."  firma de distribución con Hardened Runtime y sello de tiempo.
#   "Apple Development: ..."         solo pruebas locales de Hardened Runtime; no distribuible.
# Notarización: si hay Developer ID y SNIPRA_NOTARY_PROFILE, se ejecuta scripts/macos/notarize.sh.
# Modalidad: SNIPRA_MACOS_BUILD_MODE=private (por defecto, sin Developer ID) o
#            SNIPRA_MACOS_BUILD_MODE=distribution (requiere Developer ID, notarización y Gatekeeper).
# Ninguna modalidad crea un tag ni publica la release 1.0.0.
#
# Uso: scripts/macos/build-macos-app.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/backend/pyproject.toml" | head -1)"
PY_VERSION="3.11.16"
HF_MODEL="Systran/faster-whisper-base"
HF_MODEL_REVISION="ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
FFMPEG_SOURCE="${SNIPRA_FFMPEG_SOURCE:-sequoia}"
LOCAL_FFMPEG_PREFIX="${SNIPRA_FFMPEG_PREFIX:-/opt/homebrew/opt/ffmpeg-full}"
SIGN_IDENTITY="${SNIPRA_CODESIGN_IDENTITY:--}"
BUILD_MODE="${SNIPRA_MACOS_BUILD_MODE:-private}"
OUT="${SNIPRA_MACOS_OUT:-$ROOT/dist-release}"
CACHE="$OUT/.build-cache"
WORK="$OUT/macos-work"
HB_ROOT="$CACHE/homebrew-sequoia"
PKG_NAME="snipra-macos-${VERSION}-arm64"
PKG_DIR="$WORK/$PKG_NAME"
APP="$PKG_DIR/Snipra.app"
CONTENTS="$APP/Contents"
RES="$CONTENTS/Resources"
ZIP="$OUT/$PKG_NAME.zip"

log() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
log "Preflight"
[[ "$BUILD_MODE" == "private" || "$BUILD_MODE" == "distribution" ]] || die "SNIPRA_MACOS_BUILD_MODE debe ser 'private' o 'distribution'."
if [[ "$BUILD_MODE" == "distribution" ]]; then
  [[ "$SIGN_IDENTITY" == "Developer ID Application:"* ]] || die "Distribución bloqueada: requiere SNIPRA_CODESIGN_IDENTITY con Developer ID Application."
  [[ -n "${SNIPRA_NOTARY_PROFILE:-}" ]] || die "Distribución bloqueada: falta SNIPRA_NOTARY_PROFILE en el Keychain."
fi
[[ "$(uname -s)" == "Darwin" ]] || die "Este build solo se ejecuta en macOS."
[[ "$(uname -m)" == "arm64" ]] || die "Requiere Mac Apple Silicon (arm64)."
for tool in uv npm node swiftc codesign ditto install_name_tool otool vtool plutil lipo shasum tar; do
  command -v "$tool" >/dev/null 2>&1 || die "Falta la herramienta '$tool'."
done
[[ -n "$VERSION" ]] || die "No se pudo leer la versión de backend/pyproject.toml."
PLIST_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$ROOT/scripts/macos/Info.plist")"
[[ "$PLIST_VERSION" == "$VERSION" ]] || die "Info.plist ($PLIST_VERSION) no coincide con pyproject ($VERSION)."
if [[ "$FFMPEG_SOURCE" == "local" ]]; then
  FFMPEG_PREFIX="$LOCAL_FFMPEG_PREFIX"
  [[ -x "$FFMPEG_PREFIX/bin/ffmpeg" && -x "$FFMPEG_PREFIX/bin/ffprobe" ]] || die "No hay ffmpeg/ffprobe en $FFMPEG_PREFIX/bin."
  REQUIRED_FILTERS="subtitles drawtext zscale tonemap vidstabdetect vidstabtransform deshake hqdn3d nlmeans afftdn anlmdn loudnorm highpass lowpass acompressor alimiter"
  AVAILABLE_FILTERS="$("$FFMPEG_PREFIX/bin/ffmpeg" -hide_banner -filters 2>/dev/null | awk 'NF>=3 {print $2}')"
  for filter in $REQUIRED_FILTERS; do
    grep -qx "$filter" <<<"$AVAILABLE_FILTERS" || die "FFmpeg sin el filtro requerido '$filter' ($FFMPEG_PREFIX)."
  done
elif [[ "$FFMPEG_SOURCE" != "sequoia" ]]; then
  die "SNIPRA_FFMPEG_SOURCE debe ser 'sequoia' o 'local'."
fi
if [[ "$SIGN_IDENTITY" != "-" ]]; then
  security find-identity -v -p codesigning | grep -qF "$SIGN_IDENTITY" || die "Identidad de firma no encontrada en el Keychain: $SIGN_IDENTITY"
fi
echo "Versión: $VERSION · FFmpeg: $FFMPEG_SOURCE · firma: $([[ "$SIGN_IDENTITY" == "-" ]] && echo ad-hoc || echo "identidad configurada")"

mkdir -p "$CACHE/uv-python" "$CACHE/hf-hub" "$CACHE/uv-cache"
rm -rf "$WORK"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Helpers" "$CONTENTS/Frameworks" "$RES/app" "$RES/models/hub" "$RES/fonts" "$RES/licenses"

# ---------------------------------------------------------------- frontend
log "Frontend de producción (npm ci + build)"
(cd "$ROOT/frontend" && npm ci --no-audit --no-fund && npm run build)
[[ -f "$ROOT/frontend/dist/index.html" ]] || die "El build del frontend no generó dist/index.html."

# ---------------------------------------------------------------- Python
log "CPython $PY_VERSION standalone (python-build-standalone vía uv)"
UV_PYTHON_INSTALL_DIR="$CACHE/uv-python" uv python install "$PY_VERSION" >/dev/null
PBS_DIR="$(ls -d "$CACHE"/uv-python/cpython-"$PY_VERSION"-macos-aarch64-* | head -1)"
[[ -x "$PBS_DIR/bin/python3.11" ]] || die "No se instaló CPython $PY_VERSION."
cp -RP "$PBS_DIR" "$RES/python"
PY="$RES/python/bin/python3.11"
BUILD_PY="$PBS_DIR/bin/python3.11"
rm -rf "$RES/python/lib"/tcl9* "$RES/python/lib"/tk9* "$RES/python/lib"/itcl* "$RES/python/lib"/thread*
rm -rf "$RES/python/lib/python3.11/tkinter" "$RES/python/lib/python3.11/idlelib" \
  "$RES/python/lib/python3.11/turtledemo" "$RES/python/lib/python3.11/test"
rm -f "$RES/python/bin/idle3"* "$RES/python/bin/pydoc3"* "$RES/python/bin/2to3"*
rm -rf "$RES/python/lib/python3.11/site-packages"/pip "$RES/python/lib/python3.11/site-packages"/pip-*
# El install name de libpython apuntaba a la caché de build: se hace relocalizable.
install_name_tool -id @loader_path/libpython3.11.dylib "$RES/python/lib/libpython3.11.dylib"

# ---------------------------------------------------------------- dependencias Python
log "Dependencias de backend (uv.lock, con hashes)"
(cd "$ROOT/backend" && uv export --frozen --no-dev --no-emit-project --format requirements-txt \
  -o "$RES/requirements-lock.txt" >/dev/null)
UV_CACHE_DIR="$CACHE/uv-cache" uv pip install --python "$PY" --break-system-packages --no-config \
  --require-hashes --quiet -r "$RES/requirements-lock.txt"
find "$RES/python/lib/python3.11/site-packages" -name '*.exe' -delete
"$PY" -c "import fastapi, uvicorn, faster_whisper, yt_dlp, curl_cffi, av, ctranslate2, jsonschema" \
  || die "Las dependencias no se importan desde el runtime empaquetado."

# ---------------------------------------------------------------- FFmpeg (bottles macOS 15)
if [[ "$FFMPEG_SOURCE" == "sequoia" ]]; then
  log "Bottles de Homebrew arm64_sequoia (ffmpeg-full y cierre runtime, verificados por sha256)"
  if [[ ! -f "$HB_ROOT/bottles.json" ]]; then
    "$BUILD_PY" "$ROOT/scripts/macos/fetch-homebrew-bottles.py" --root ffmpeg-full --tag arm64_sequoia --out "$HB_ROOT"
  fi
  FFMPEG_PREFIX="$HB_ROOT/opt/ffmpeg-full"
  [[ -x "$FFMPEG_PREFIX/bin/ffmpeg" ]] || die "Falta ffmpeg en el bottle descargado."
fi

# ---------------------------------------------------------------- modelo Whisper fijado
log "Modelo $HF_MODEL @ ${HF_MODEL_REVISION:0:8}"
env -i HOME="$HOME" PATH="/usr/bin:/bin" PYTHONNOUSERSITE=1 "$PY" - "$CACHE/hf-hub" "$HF_MODEL" "$HF_MODEL_REVISION" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
cache_dir, repo, revision = sys.argv[1], sys.argv[2], sys.argv[3]
path = snapshot_download(repo_id=repo, revision=revision, cache_dir=cache_dir)
print("snapshot:", path)
PYEOF
SNAP="$CACHE/hf-hub/models--${HF_MODEL//\//--}/snapshots/$HF_MODEL_REVISION"
[[ -s "$SNAP/model.bin" ]] || die "Falta model.bin del modelo $HF_MODEL."
MODEL_DST="$RES/models/hub/models--${HF_MODEL//\//--}"
mkdir -p "$MODEL_DST/refs" "$MODEL_DST/snapshots/$HF_MODEL_REVISION"
printf '%s' "$HF_MODEL_REVISION" > "$MODEL_DST/refs/main"
cp -RL "$SNAP/." "$MODEL_DST/snapshots/$HF_MODEL_REVISION/"

# ---------------------------------------------------------------- código de la app
log "Código de la aplicación"
(cd "$ROOT" && COPYFILE_DISABLE=1 tar --exclude='__pycache__' --exclude='*.pyc' -cf - backend/app) | (cd "$RES/app" && tar -xf -)
mkdir -p "$RES/app/.agents/skills"
(cd "$ROOT/.agents/skills" && COPYFILE_DISABLE=1 tar --exclude='evals' -cf - snipra-video-director) | (cd "$RES/app/.agents/skills" && tar -xf -)
mkdir -p "$RES/app/frontend"
cp -R "$ROOT/frontend/dist" "$RES/app/frontend/dist"
(cd "$ROOT" && COPYFILE_DISABLE=1 tar --exclude='node_modules' --exclude='tests' -cf - extension) | (cd "$RES/app" && tar -xf -)
cp "$ROOT/LICENSE" "$ROOT/README.md" "$ROOT/CHANGELOG.md" "$RES/app/"
cp "$ROOT/docs/INSTALL_MACOS.md" "$RES/app/INSTALL_MACOS.md"
find "$RES/app" -name '.DS_Store' -delete
find "$RES/python/lib/python3.11/site-packages" -name '*.exe' -delete
"$PY" -m compileall -q "$RES/app/backend/app" "$RES/python/lib/python3.11/site-packages" >/dev/null 2>&1 || true

# ---------------------------------------------------------------- icono y fuentes
log "Icono Brand V2 y configuración de fuentes"
cp "$ROOT/snipra.icns" "$RES/snipra.icns"
"$PY" - "$ROOT" <<'PYEOF'
import pathlib, struct, sys
root = pathlib.Path(sys.argv[1])
data = (root / "snipra.icns").read_bytes()
assert data[:4] == b"icns", "snipra.icns inválido"
chunks, pos = {}, 8
while pos < len(data):
    kind = data[pos:pos + 4].decode("ascii")
    length = struct.unpack(">I", data[pos + 4:pos + 8])[0]
    chunks[kind] = data[pos + 8:pos + length]
    pos += length
expected = {"icp4": 16, "icp5": 32, "icp6": 64, "ic07": 128, "ic08": 256, "ic09": 512, "ic10": 1024}
for kind, size in expected.items():
    canonical = (root / f"frontend/image/snipra-brand-v2/app-icons/snipra-app-icon-{size}.png").read_bytes()
    assert chunks.get(kind) == canonical, f"{kind} no coincide con Brand V2 {size}px"
print("snipra.icns == Brand V2 oficial (7 tamaños)")
PYEOF
cp "$ROOT/scripts/macos/fonts.conf" "$RES/fonts/fonts.conf"

# ---------------------------------------------------------------- lanzador
log "Lanzador nativo (swiftc, arm64, deployment target 13.0)"
swiftc -O -swift-version 5 -target arm64-apple-macos13.0 -framework AppKit -framework Foundation \
  "$ROOT/scripts/macos/launcher/SnipraLauncher.swift" -o "$CONTENTS/MacOS/Snipra"
cp "$ROOT/scripts/macos/Info.plist" "$CONTENTS/Info.plist"
plutil -lint "$CONTENTS/Info.plist" >/dev/null
printf 'APPL????' > "$CONTENTS/PkgInfo"

# ---------------------------------------------------------------- FFmpeg relocalizado
log "FFmpeg/FFprobe y dylibs (relocalización @rpath)"
BUNDLE_ARGS=(--ffmpeg-prefix "$FFMPEG_PREFIX" --helpers "$CONTENTS/Helpers" --frameworks "$CONTENTS/Frameworks"
  --manifest "$WORK/ffmpeg-bundle-manifest.json")
[[ "$FFMPEG_SOURCE" == "sequoia" ]] && BUNDLE_ARGS+=(--homebrew-root "$HB_ROOT")
"$BUILD_PY" "$ROOT/scripts/macos/bundle_ffmpeg.py" "${BUNDLE_ARGS[@]}"

# ---------------------------------------------------------------- avisos de licencia
log "Inventario de licencias y avisos de terceros"
"$BUILD_PY" "$ROOT/scripts/macos/license_inventory.py" --app "$APP" \
  --ffmpeg-manifest "$WORK/ffmpeg-bundle-manifest.json" --frontend "$ROOT/frontend" --repo "$ROOT" \
  --out "$RES/licenses"
cp -R "$ROOT/scripts/macos/licenses/texts" "$RES/licenses/texts"
if [[ "$FFMPEG_SOURCE" == "sequoia" ]]; then
  cp "$HB_ROOT/bottles.json" "$RES/homebrew-bottles.json"
  cp "$HB_ROOT/bottles.json" "$RES/licenses/homebrew-bottles.json"
fi
cp "$ROOT/LICENSE" "$RES/licenses/LICENSE-Snipra"
cp "$ROOT/scripts/macos/licenses/README.md" "$RES/licenses/README.md"

# ---------------------------------------------------------------- limpieza antes de firmar
# Metadatos AppleDouble (._*) y el directorio de datos no deben viajar en el bundle firmado.
find "$APP" -name '._*' -type f -delete
find "$APP" -name '.DS_Store' -type f -delete
rm -rf "$RES/app/storage"

# ---------------------------------------------------------------- firma
ENTITLEMENTS="$ROOT/scripts/macos/entitlements/snipra-runtime.entitlements"
SIGN_ARGS=(--identity "$SIGN_IDENTITY")
if [[ "$SIGN_IDENTITY" != "-" ]]; then
  SIGN_ARGS+=(--runtime --timestamp --entitlements "$ENTITLEMENTS")
fi
log "Firma ($([[ "$SIGN_IDENTITY" == "-" ]] && echo ad-hoc || echo "Hardened Runtime"))"
"$BUILD_PY" "$ROOT/scripts/macos/sign_bundle.py" "$APP" "${SIGN_ARGS[@]}"
codesign --verify --deep --strict --verbose=2 "$APP" 2>&1 | tail -1

# ---------------------------------------------------------------- verificación del bundle
log "Verificación del bundle"
"$BUILD_PY" "$ROOT/scripts/macos/check_bundle.py" "$APP" --expected-version "$VERSION" \
  --report "$WORK/bundle-check.json"

# ---------------------------------------------------------------- notarización
NOTARY_STATUS="NOTARIZATION_PENDING"
SIGNING_STATUS="ad-hoc"
if [[ "$SIGN_IDENTITY" == Developer\ ID\ Application* ]]; then
  SIGNING_STATUS="developer-id"
  if [[ -n "${SNIPRA_NOTARY_PROFILE:-}" ]]; then
    log "Notarización (notarytool), engrapado y Gatekeeper"
    "$ROOT/scripts/macos/notarize.sh" "$APP" || die "Notarización/Gatekeeper falló; no se generará ZIP de esta compilación."
    NOTARY_STATUS="ACCEPTED"
  fi
elif [[ "$SIGN_IDENTITY" != "-" ]]; then
  SIGNING_STATUS="apple-development-local-only"
fi
if [[ "$BUILD_MODE" == "distribution" && "$NOTARY_STATUS" != "ACCEPTED" ]]; then
  die "Distribución bloqueada: el bundle no tiene notarización y Gatekeeper verificados."
fi
echo "modo: $BUILD_MODE · firma: $SIGNING_STATUS · notarización: $NOTARY_STATUS"

# ---------------------------------------------------------------- paquete
log "Paquete ZIP (ditto)"
cp "$ROOT/docs/INSTALL_MACOS.md" "$PKG_DIR/INSTALL_MACOS.md"
cp "$ROOT/LICENSE" "$PKG_DIR/LICENSE"
mkdir -p "$PKG_DIR/LICENSES"
cp "$RES/licenses/THIRD_PARTY_NOTICES.md" "$RES/licenses/SOURCES.md" "$PKG_DIR/LICENSES/"
cp -R "$RES/licenses/texts" "$PKG_DIR/LICENSES/texts"
"$BUILD_PY" "$ROOT/scripts/macos/write_manifest.py" --app "$APP" --root "$ROOT" --version "$VERSION" \
  --ffmpeg-manifest "$WORK/ffmpeg-bundle-manifest.json" --check "$WORK/bundle-check.json" \
  --signing "$SIGNING_STATUS" --notarization "$NOTARY_STATUS" --build-mode "$BUILD_MODE" \
  --bottles "$HB_ROOT/bottles.json" --out "$PKG_DIR/MANIFEST.json"
rm -f "$ZIP"
# Sin atributos extendidos ni recursos: evita entradas ._ (AppleDouble) en el ZIP.
ditto -c -k --noextattr --norsrc --keepParent "$PKG_DIR" "$ZIP"
(cd "$OUT" && shasum -a 256 "$(basename "$ZIP")" > SHA256SUMS)
rm -rf "$WORK/unpacked" && mkdir -p "$WORK/unpacked"
ditto -x -k "$ZIP" "$WORK/unpacked"

log "Listo"
echo "ZIP:     $ZIP"
echo "Tamaño:  $(du -h "$ZIP" | cut -f1)"
echo "SHA256:  $(cut -d' ' -f1 "$OUT/SHA256SUMS")"
echo "Bundle:  $APP"
