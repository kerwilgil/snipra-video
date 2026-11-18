# Cumplimiento de licencias: Snipra 1.0.0 para macOS (arm64)

Estado: **mitigaciones técnicas aplicadas; LEGAL PASS NO declarado**. Este
documento es un análisis técnico, no asesoramiento legal. Ninguna revisión
jurídica se ha realizado. La sección 7 indica qué obligaciones de licencia se
cumplen de forma verificable en la distribución pública ad hoc y qué queda
como riesgo aceptado o pendiente.

## 1. Alcance y método

El inventario se genera en cada build con `scripts/macos/license_inventory.py`
a partir del propio bundle, no de una lista escrita a mano:

| Fuente | Qué aporta |
|---|---|
| Fórmula Homebrew de cada keg (`.brew/<formula>.rb`) | Licencia, URL de origen y sha256 de la versión que se empaqueta |
| `dist-info` de los paquetes Python del bundle | Licencia y versión de cada dependencia Python |
| `package.json` de `frontend/node_modules` (`npm ls --omit=dev`) | Licencia de las dependencias del frontend compilado |
| Configuración de compilación de `libavutil` | Licencia efectiva de FFmpeg (`--enable-gpl`, `--enable-version3`) |
| README del modelo `Systran/faster-whisper-base` | Licencia del modelo de transcripción |

Resultado del último build: **165 componentes**, 33 marcados para revisión
legal. Inventario completo en `Contents/Resources/licenses/inventory.json` y
`LICENSES/THIRD_PARTY_NOTICES.md` dentro del paquete.

## 2. Licencia efectiva de FFmpeg

`ffmpeg` y `ffprobe` provienen de los bottles `ffmpeg-full` 9.0.2 (Homebrew,
`arm64_sequoia`). Su configuración incluye `--enable-gpl` y `--enable-version3`,
por lo que el binario combinado es **GPL-3.0-or-later**. Son 50 flags de
habilitación; los mismos que el build local 9.0.1 salvo el prefijo.

Componentes incluidos con licencia copyleft (todos con código fuente en la
fórmula de origen, ver sección 5):

| Componente | Licencia | Enlace con |
|---|---|---|
| x264 (r3222) | GPL-2.0-or-later | ffmpeg (encoder H.264) |
| x265 (4.3) | GPL-2.0-or-later | ffmpeg (encoder HEVC) |
| libvidstab (1.1.2) | GPL-2.0-or-later | ffmpeg (estabilización) |
| rubberband (4.0.0) | GPL-2.0-or-later | ffmpeg (time-stretch) |
| xvid (1.3.7) | GPL-2.0-or-later | ffmpeg |
| gnutls (3.8.13) | LGPL-2.1-or-later AND GPL-3.0-only | ffmpeg (TLS) |
| gettext (libintl) | GPL-3.0-or-later AND LGPL-2.1-or-later | dependencia |
| libbluray, libplacebo, libsoxr, libssh, libtasn1, libudfread, qrencode, lame, glib, aribb24, mpg123 | LGPL-2.x / LGPL-3.0 | bibliotecas dinámicas separadas |
| srt, zeromq | MPL-2.0 | bibliotecas dinámicas separadas |
| gmp, nettle, libidn2, libunistring, xz, zstd, fribidi, graphite2 | Doble licencia (elegir una compatible) | bibliotecas |

Comprobaciones de compatibilidad:

- No hay componente exclusivamente **GPL-2.0-only** sin alternativa. Las
  expresiones `any_of` permiten elegir una licencia compatible con GPLv3
  (`GPL-2.0-or-later` y `LGPL-3.0-or-later` son compatibles con GPL-3.0-or-later).
- `gnutls` incluye código **GPL-3.0-only**: es compatible con un binario
  GPL-3.0-or-later.
- Ningún componente permisivo (MIT, BSD, Apache-2.0, ISC) impone condiciones
  incompatibles con GPLv3.

## 3. Hallazgos que requieren revisión legal

Cada punto está marcado `LEGAL_REVIEW_REQUIRED` en `inventory.json`.

**H1. Bibliotecas GPL cargadas dentro del proceso del backend (PyAV).**
El paquete `av` (PyAV) trae sus propias bibliotecas FFmpeg (`av/.dylibs`).
Su configuración es `--enable-version3 --enable-libx264 --enable-libx265` y
trae `libx264.165.dylib` y `libx265.216.dylib` (GPL-2.0-or-later) como dylibs
que `libavcodec` carga. faster-whisper las usa para **decodificar audio**
(`faster_whisper.audio.decode_audio`), dentro del mismo proceso Python que el
backend de Snipra. Snipra no usa sus codificadores; la codificación se hace con
el binario `ffmpeg` separado.
- Riesgo: un proceso que carga código GPL en su espacio de direcciones puede
  considerarse una obra combinada. Eso afectaría a cómo se licencia el backend.
- Opciones a evaluar (sin decidir aquí): (a) sustituir la decodificación por
  `ffmpeg` como proceso separado y no cargar `av`, lo que exige cambiar código de
  faster-whisper o aislarlo; (b) recompilar PyAV sin x264/x265; (c) aceptar el
  riesgo con asesoramiento. No se eliminan capacidades de FFmpeg para evitar la revisión.

**H2. Sustitución de bibliotecas LGPL en un bundle firmado.**
Las LGPL van como dylibs separadas en `Contents/Frameworks`, lo que permite la
relinkización. Con Hardened Runtime y firma, un usuario no puede sustituir una
dylib sin romper la firma del bundle. Hay que decidir si eso cumple las
condiciones de LGPL sobre reemplazo y modificación para el uso privado y para la
distribución pública.

**H3. Patentes de códecs.** H.264 (x264), HEVC (x265), AAC (encoder nativo de
FFmpeg), y AV1 (aom, SVT-AV1, rav1e). Las licencias de código (GPL, BSD) no
conceden licencias de patentes de estos formatos. Requiere análisis de las
condiciones de los pools de patentes para distribución en la UE y EE. UU.

**H4. GPLv3 sección 6 (información de instalación).** Aplica a «productos de
usuario». Hay que determinar si Snipra para macOS entra en esa categoría.

**H5. Obra combinada del propio Snipra.** El backend y el launcher son código
propio (MIT). Se comunican con `ffmpeg` por línea de comandos y ficheros, lo que
en general se considera programas separados. La sección H1 es la excepción.

**H6. Repositorio privado.** `snipra-private` es privado. Si la distribución
pública exige ofrecer el código de Snipra, eso no está resuelto.

## 4. Obligaciones de distribución (resumen)

| Obligación | Dónde se cumple |
|---|---|
| Texto de la licencia de cada componente | `LICENSES/texts/` (dentro del ZIP) y `Contents/Resources/licenses/texts/` |
| Avisos de copyright y atribución | `LICENSES/THIRD_PARTY_NOTICES.md` y `Contents/Resources/licenses/components/` |
| Código fuente correspondiente (GPL/LGPL/MPL) | `LICENSES/SOURCES.md` con URL y sha256; paquete separado de código fuente (sección 5) |
| Licencia de Snipra | `LICENSE` (MIT) en la raíz del ZIP y en `Contents/Resources/licenses/LICENSE-Snipra` |
| Fórmulas de compilación (opciones y parches) | `Contents/Resources/licenses/formulae/` |
| Modelo de transcripción (MIT) | Atribución en `THIRD_PARTY_NOTICES.md` y en `inventory.json` |
| Fuentes Inter (SIL OFL 1.1) | Texto OFL en `texts/OFL-1.1.txt`; la fuente no se modifica |
| Python (PSF License) | `Contents/Resources/python/lib/python3.11/LICENSE.txt` |

## 5. Código fuente correspondiente

Cada entrada de `SOURCES.md` apunta a la URL exacta de la fórmula Homebrew
usada y a su sha256. Para obtener el código fuente completo:

```sh
scripts/macos/fetch-corresponding-source.py \
  --inventory "Snipra.app/Contents/Resources/licenses/inventory.json" \
  --formulae "Snipra.app/Contents/Resources/licenses/formulae" \
  --out ./corresponding-source
```

El script descarga cada componente GPL/LGPL/MPL, verifica su sha256 y lo guarda
con los parches de la fórmula. Los componentes obtenidos de git (x264) se
clonan en la revisión indicada por la fórmula. El resultado del último build
está en el paquete `snipra-macos-1.0.0-corresponding-source.zip` (ver
`RELEASE_1_0_0` en el informe).

Advertencia: la oferta escrita del código fuente debe mantenerse válida durante
el plazo que exija la licencia. La revisión jurídica debe confirmar la forma
exacta de esa oferta.

## 6. Pendiente

- Revisión jurídica formal (no realizada): ver sección 7.

## 7. Resolución para la distribución pública macOS ad hoc (modo `public-adhoc`)

| Hallazgo | Resolución | Verificación |
|---|---|---|
| **H1** PyAV con x264/x265 (GPL) en el proceso del backend | **Resuelto técnicamente.** PyAV no se distribuye: el bundle lleva un stub inerte de `av` (faster-whisper solo hace `import av`). La transcripción recibe el audio ya decodificado por el `ffmpeg` externo (`load_wav_16k_mono`). Ninguna biblioteca GPL se carga en el proceso Python de Snipra. | `check_bundle.py`: «Sin bibliotecas GPL (PyAV/x264/x265) dentro del proceso Python del backend»; `backend/tests/test_transcribe_audio_input.py`; transcripción de un video real en el build. |
| **H2** LGPL en bundle firmado | **Resuelto técnicamente.** La distribución pública usa firma ad hoc **sin Hardened Runtime ni library validation**: las dylibs LGPL (`Contents/Frameworks`) pueden sustituirse y la app volver a firmarse con `codesign --force --deep -s - Snipra.app`. Instrucciones en `INSTALL_MACOS.md`. | Comando documentado y probado sobre una copia del bundle. |
| **H4** GPLv3 §6 (información de instalación) | Snipra es una aplicación de software, no un «producto de usuario» (dispositivo de consumo). Aun así, el punto anterior da el mecanismo de reinstalación. Sin firma restrictiva. | Igual que H2. |
| **H5** Obra combinada | Con H1 resuelto, el backend MIT y `ffmpeg`/`ffprobe` (GPL-3.0-or-later) son programas separados que se comunican por línea de comandos y ficheros (agregación). | `check_bundle.py` (sin GPL en proceso). |
| **H6** Código fuente | Para los componentes GPL/LGPL/MPL el código fuente correspondiente se publica **en la misma release** (`snipra-macos-1.0.0-corresponding-source.zip` + SHA-256), con URL y sha256 por componente en `SOURCES.md`. El código de Snipra es MIT y va legible dentro de la app; no se publica código privado adicional. | Verificación de hashes de la release. |
| **Oferta escrita** | `LICENSES/SOURCES.md` y la release de GitHub. El enlace debe mantenerse disponible mientras se distribuya el binario y al menos tres años (GPLv3 §6(b)); no retirar los assets de fuente. | Enlaces verificados tras publicar. |
| **H3** Patentes de códecs (H.264, HEVC, AAC) | **No es una obligación de licencia de copyright y no puede resolverse técnicamente.** Se mantiene como **riesgo declarado y no evaluado jurídicamente**: la licencia de código no concede licencias de patente. Se informa en la release; no se afirma LEGAL PASS. | Acción mínima: asesoramiento jurídico si se distribuye en jurisdicciones que exijan licencias de patente (UE/EE. UU.). |

El build ad hoc **no está notarizado** y no se presenta como tal. El modo oficial
`distribution` (Developer ID + notarización + Gatekeeper) permanece intacto.
