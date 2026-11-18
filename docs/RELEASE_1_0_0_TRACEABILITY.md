# Trazabilidad y cierre técnico de Snipra 1.0.0

## Identidades de la distribución existente

Verificación realizada el 18 de noviembre de 2026 (UTC; 17 de noviembre en Panamá).
La [release publicada](https://github.com/kerwilgil/snipra-video/releases/tag/v1.0.0)
se conserva: este cambio no reconstruye, sustituye ni publica artefactos, ni mueve el tag.

| Evidencia | Valor |
| --- | --- |
| Tag público `v1.0.0` (referencia directa a commit) | `25911cd0730b37ac617d8062613ed50311522e72` |
| Base pública del PR | `814009ae12dbe670b91cd08f24e21bc45c9c6c2c` |
| Commit de compilación declarado en `MANIFEST.json` | `85931560bbdd8145bb765b046ca7746d837e01d3` |
| Árbol sucio declarado al compilar | `false` |
| Fecha de compilación declarada | `2026-11-18T00:46:45Z` |
| Publicación declarada por GitHub | `2026-11-18T01:33:27Z` |
| Canal / firma / notarización | `public-adhoc` / `ad-hoc` / `NOT_NOTARIZED` |
| Estado interno del manifiesto | `PUBLIC_ADHOC_CANDIDATE` (estado al compilar, anterior a la publicación) |

El commit de compilación y el tag público **son distintos**. El tag identifica la
publicación de documentación y no es el commit desde el cual se compiló la app.
No se declara una relación de ancestros entre ambos historiales ni una compilación
reproducible bit a bit desde ese tag. El SHA de compilación ya está divulgado en el
manifiesto distribuido; este PR no incorpora su historial.

## Integridad de los artefactos

Se descargó `SHA256SUMS` de la release. Los hashes calculados sobre ambos ZIP
locales coinciden tanto con ese fichero como con los `digest` de los assets en
la API de GitHub. Los identificadores permiten volver a consultar los mismos assets.

| Archivo | Asset ID | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `snipra-macos-1.0.0-arm64.zip` | `761281457` | 293412817 | `0ba181acd528eca6d91814cd396d4ac2e282de00f87149f64b12e0c97caa1aca` |
| `snipra-macos-1.0.0-corresponding-source.zip` | `761281458` | 110600942 | `c762a09710699768ea5ff5fb176773dd6cee6056360f5874a046b6e42223e7aa` |
| `SHA256SUMS` | `761281459` | 205 | `b101b3be77c6fc72b5a404a282956ce6d9adc56494c4a5bc30c63554b8349022` |

El manifiesto incluido declara además el hash del lanzador
`09512f68c6bf224d786556729be2e845f3c5812b4749e8d6dd2f011ef077e55e`
y del fichero de requisitos utilizado
`eb4a6552df835182883c518130dc8c94c7f9d61ce03d1ce0e3fa4f19aad03034`.
Estos datos son evidencia de procedencia declarada, no una atestación firmada por CI.

Para repetir la comprobación en una carpeta de descarga nueva:

```sh
gh release download v1.0.0 --repo kerwilgil/snipra-video --dir release-verification
cd release-verification
shasum -a 256 -c SHA256SUMS
unzip -p snipra-macos-1.0.0-arm64.zip snipra-macos-1.0.0-arm64/MANIFEST.json
gh api repos/kerwilgil/snipra-video/git/ref/tags/v1.0.0
```

## Correspondencia del código revisado

Se compararon los archivos públicos con el checkout de compilación. Se trasladan
únicamente el valor por defecto de retención (0), la espera antes del primer
mantenimiento, la entrada PCM de transcripción y el empaquetado `public-adhoc`
con exclusión de PyAV y comprobación de bibliotecas GPL en el runtime Python.
La UI usa también 0 si el backend no entrega retención. El modo `distribution`
conserva los requisitos de Developer ID y notarización.

Los tres módulos de backend modificados coinciden byte a byte con los del ZIP.
Se compararon 137 archivos de backend y extensión incluidos en el ZIP: la única
diferencia con el árbol propuesto es `extension/README.md`, documentación pública
que se conserva. El código fuente de frontend (excluidos tests) coincide con el
commit declarado de compilación después del ajuste de Settings. Esta comparación
no sustituye la reproducción del bundle de frontend.

No se trasladan documentación interna, rutas personales, referencias al repositorio
privado, datos locales, credenciales ni historial de commits. Se conservan la
presentación pública, el inventario con URL pública y la documentación de licencias
ya publicada. Los scripts mantienen los permisos de archivo del repositorio público.

Los locks públicos ya eran posteriores a los del checkout de compilación. Se
conservan: no se introducen downgrades de dependencias para aparentar identidad de
build. Para las dependencias efectivamente empaquetadas, la evidencia es
`Contents/Resources/requirements-lock.txt`, el inventario de licencias y sus hashes
incluidos en el ZIP. El PR alinea el comportamiento del código; **no convierte el
tag histórico en una receta exacta del binario**. El ZIP de fuentes correspondiente
contiene los componentes de terceros GPL/LGPL/MPL, no un snapshot Git de Snipra.

## Fallo de ChannelsPage y pruebas afectadas

El fallo reproducido en la rama de referencia fue `TypeError: The URL must be of
scheme file`. Vite transforma el patrón estático `new URL(..., import.meta.url)`
en una URL de recurso para el navegador; `node:fs` necesita una ruta local.
La versión pública evitaba ese patrón pero dependía de `process.cwd()`.
El test resuelve ahora la ruta desde `dirname(fileURLToPath(import.meta.url))`.
Se mantienen todas las expectativas de contenedores, selectores y ancho de 1200 px.
No se modifica el CSS ni se desactiva ningún test.

Comandos limitados a las áreas afectadas:

```sh
cd frontend
npm ci --no-audit --no-fund
npm test -- src/pages/ChannelsPage.test.tsx src/pages/SettingsPage.test.tsx
cd ../backend
uv sync --frozen --python 3.11
uv run --frozen pytest tests/test_worker.py tests/test_transcribe_audio_input.py -q
cd ..
uv run --project backend --frozen python -m unittest discover -s scripts/macos/tests -v
bash -n scripts/macos/build-macos-app.sh
```

Las pruebas de transcripción usan WAV sintéticos y un modelo simulado, sin descarga
de modelos: verifican PCM, normalización, timestamps, persistencia y limpieza.
Las de retención verifican archivos conservados por defecto y primer mantenimiento
tras un intervalo completo. Las de packaging simulan herramientas Apple: no son
pruebas físicas de firma o Gatekeeper. El workflow `Release source parity` ejecuta
solo estos tests de PR, sin construir, firmar ni publicar una release. No existían
workflows ni checks en la base pública; el resultado remoto se registra en el PR.

## Límites y aprobación pendiente

- Firma **ad hoc**, sin Developer ID; **sin notarización** y sin `LEGAL PASS`.
- Safari real y apertura en otro Mac con Gatekeeper: **PENDING_PHYSICAL**.
- Revisión jurídica de licencias y patentes H.264/HEVC/AAC: pendiente; este cierre
  no emite conclusiones jurídicas ni vuelve a auditar esos temas.
- La discrepancia histórica tag/build y la falta de reproducibilidad exacta se
  documentan; no se corrigen alterando los artefactos ya publicados.
- Revisión humana del diff y del CI antes del merge. No se habilita auto-merge.

Consultar [instalación](INSTALL_MACOS.md) y [licencias](LICENSE_COMPLIANCE.md)
para las limitaciones ya publicadas.
