#!/usr/bin/env bash
# Notariza y engrapa Snipra.app (Apple notarytool + stapler) y valida con spctl.
#
# Requisitos (no se crean ni se muestran credenciales en este script):
#   - App firmada con "Developer ID Application" y Hardened Runtime (ver docs/MACOS_SIGNING.md).
#   - Perfil de notarización en el Keychain, creado una sola vez por el titular de la cuenta:
#       xcrun notarytool store-credentials "<perfil>" --apple-id ... --team-id ... (pide la contraseña específica de app)
#   - SNIPRA_NOTARY_PROFILE=<perfil>
#
# Códigos de salida: 0 aceptada y validada · 2 NOTARIZATION_PENDING (sin perfil) ·
#                    3 SIGNING_BLOCKED_CREDENTIALS (sin Developer ID) · 4 sin Hardened Runtime ·
#                    5 envío/rechazo de notarización · 6 rechazo de Gatekeeper.
set -euo pipefail

APP="${1:?uso: notarize.sh Snipra.app}"
PROFILE="${SNIPRA_NOTARY_PROFILE:-}"
if [[ -z "$PROFILE" ]]; then
  echo "NOTARIZATION_PENDING: falta SNIPRA_NOTARY_PROFILE (perfil de notarytool en el Keychain)."
  exit 2
fi
DETAILS="$(codesign -dv --verbose=4 "$APP" 2>&1 || true)"
if ! grep -q "Authority=Developer ID Application" <<<"$DETAILS"; then
  echo "SIGNING_BLOCKED_CREDENTIALS: el bundle no está firmado con Developer ID Application."
  exit 3
fi
if ! grep -q "flags=0x10000(runtime)" <<<"$DETAILS"; then
  echo "HARDENED_RUNTIME_MISSING: el bundle debe firmarse con --options runtime."
  exit 4
fi

WORK="$(mktemp -d)"
ZIP="$WORK/Snipra-notarize.zip"
RESULT="$WORK/notary-result.json"
ditto -c -k --keepParent "$APP" "$ZIP"
echo "Enviando a notarización (notarytool --wait)…"
if ! xcrun notarytool submit "$ZIP" --keychain-profile "$PROFILE" --wait --output-format json > "$RESULT"; then
  echo "NOTARIZATION_SUBMIT_FAILED: notarytool no completó el envío; no se certifica el bundle." >&2
  exit 5
fi
STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$RESULT")"
SUBMISSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("id",""))' "$RESULT")"
echo "estado: ${STATUS:-desconocido} · id: ${SUBMISSION:-n/a}"
if [[ "$STATUS" != "Accepted" ]]; then
  [[ -n "$SUBMISSION" ]] && xcrun notarytool log "$SUBMISSION" --keychain-profile "$PROFILE" "$WORK/notary-log.json" || true
  echo "NOTARIZATION_REJECTED: revisar $WORK/notary-log.json (sin credenciales en el registro)."
  exit 5
fi

xcrun stapler staple "$APP"
xcrun stapler validate "$APP"
if ! spctl --assess --type execute -vvv "$APP"; then
  echo "GATEKEEPER_REJECTED: spctl rechazó el bundle; prohibido certificarlo para distribución." >&2
  exit 6
fi
echo "NOTARIZATION_ACCEPTED: engrapado y Gatekeeper verificados. Listo para empaquetar."
