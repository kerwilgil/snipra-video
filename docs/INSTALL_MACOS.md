# Instalación de Snipra en macOS

Snipra 1.0.0 para macOS se distribuye como `Snipra.app` autocontenida, para
Apple Silicon (arm64). La app incluye su propio Python, FFmpeg/FFprobe, las
dependencias del backend, la interfaz compilada y el modelo de transcripción
base. No necesita Homebrew, Node.js, Python ni FFmpeg instalados en el Mac.

## Estado de verificación

| Área | Estado |
| --- | --- |
| Build local arm64 en macOS 27.0.1 | Verificado en este Mac |
| Ejecución desde una carpeta extraída con espacios y Unicode | Verificado |
| API, worker, interfaz, render, export, reedit y persistencia | Verificado (ver `MANIFEST.json`) |
| Canal | **Distribución pública ad hoc, NO notarizada** (`public-adhoc`). Solo Apple Silicon (arm64) |
| Firma | **Ad hoc** (`codesign -s -`). Sin Developer ID |
| Notarización Apple | **No realizada** |
| Interfaz en Brave (claro, oscuro, 390 px, zoom 125 % y 150 %) | Verificado (emulación controlada, sin desbordes ni errores) |
| Interfaz en Safari | PENDING_PHYSICAL (la auditoría WebKit no sustituye la prueba en Safari) |
| Abrir en otro Mac con Gatekeeper | PENDING_PHYSICAL |
| Revisión jurídica de licencias | **No realizada**; mitigaciones técnicas en `docs/LICENSE_COMPLIANCE.md` (patentes de códecs sin evaluar) |

## Requisitos

- Mac con Apple Silicon (M1 o posterior). Intel (x86_64) no está soportado.
- macOS 27.0 o superior: es el mínimo **declarado** y el único verificado con ejecución real (27.0.1).
  El mínimo técnico de los binarios es 15.7.9; no se declara uno menor hasta verificarlo en esa versión.
- Unos 1,2 GB libres: la app ocupa ~650 MB y los datos crecen con los trabajos.
- Un navegador para la interfaz: Safari, Brave o Chromium.
- Internet solo para descargar medios por URL, usar proveedores de IA configurados
  o descargar modelos de transcripción distintos de `base`.

## Contenido del paquete

`snipra-macos-1.0.0-arm64.zip` contiene una carpeta con:

- `Snipra.app`: la aplicación.
- `INSTALL_MACOS.md`: este documento.
- `LICENSE`: licencia del código de Snipra.
- `LICENSES/`: avisos de terceros (`THIRD_PARTY_NOTICES.md`), oferta de código fuente
  (`SOURCES.md`) y textos de licencia. Ver `docs/LICENSE_COMPLIANCE.md`.
- `MANIFEST.json`: versión, componentes, hashes, estado de firma y resultados de verificación.

El archivo `SHA256SUMS` se distribuye junto al ZIP, no dentro de él.
La relación entre commit de compilación, tag y hashes publicados se documenta en
[trazabilidad de 1.0.0](RELEASE_1_0_0_TRACEABILITY.md).

## Instalación

1. Verifica la integridad en la carpeta donde descargaste el ZIP:

   ```sh
   shasum -a 256 -c SHA256SUMS
   ```

2. Extrae el ZIP en cualquier carpeta (no hace falta el checkout de Git):

   ```sh
   ditto -x -k snipra-macos-1.0.0-arm64.zip ~/Descargas
   ```

3. Arrastra `Snipra.app` a `/Applications`. Puedes moverla sin reinstalar nada:
   los datos viven fuera de la app.

## Primera ejecución

La app no está notarizada, así que macOS pide aprobación la primera vez:

1. Abre `Snipra.app` desde `/Applications`. macOS la bloqueará porque no está
   notarizada. Cierra el aviso, ve a **Ajustes del Sistema → Privacidad y
   seguridad** y pulsa **Abrir igualmente** junto a Snipra; confirma con tu
   contraseña. En macOS 15 o posterior, clic derecho → **Abrir** ya no basta.
   No desactives Gatekeeper ni uses comandos que lo desactiven. La firma actual es ad hoc (sin Developer ID
   ni notarización), por eso macOS la rechaza en otros equipos.
2. El lanzador muestra el progreso: comprueba los componentes, inicia la API y el
   worker, y abre la interfaz en tu navegador predeterminado en
   `http://127.0.0.1:8000/`.
3. Si el puerto 8000 está ocupado por otro programa, Snipra usa el 8001. Si ambos
   están ocupados, muestra un aviso y no detiene ningún proceso ajeno.

El primer arranque copia el modelo de transcripción incluido a
`~/Library/Application Support/Snipra/models/`. Tarda unos segundos.

## Sustituir bibliotecas LGPL (derecho de relink)

Las bibliotecas LGPL están como dylibs separadas en `Snipra.app/Contents/Frameworks`.
La app usa firma ad hoc sin Hardened Runtime, así que puedes sustituir una por tu
propia versión compatible y volver a firmar:

```sh
cp tu-libreria.dylib Snipra.app/Contents/Frameworks/<nombre>.dylib
codesign --force --deep -s - Snipra.app
```

El código fuente correspondiente de FFmpeg y sus bibliotecas está en la misma
release (`snipra-macos-1.0.0-corresponding-source.zip`).

## Uso

Desde la interfaz puedes subir un video o pegar una URL pública, elegir formatos
9:16, 1:1 y 16:9, y seguir el progreso del trabajo. Los clips y los renders se
guardan en la base local de Snipra. Los trabajos se procesan en este Mac.

La extensión de navegador está en `Snipra.app/Contents/Resources/app/extension`
(clic derecho sobre `Snipra.app` → **Mostrar contenido del paquete**). Se carga
como extensión descomprimida. Genera el token en **Ajustes → Extensión de
navegador** y pégalo en el popup. La extensión busca Snipra en los puertos 8000 y
8001 (ver *Primera ejecución*).

## Cómo detener Snipra

Usa **Cmd+Q**, el botón **Detener Snipra** o cierra la ventana del lanzador. Snipra
detiene solo sus propios procesos (API y worker) y libera los puertos. Verifica que
no quedan procesos con:

```sh
pgrep -fl "app.worker.runner|uvicorn app.main"
```

## Datos persistentes y ubicaciones

Snipra **no borra tus trabajos automáticamente**: la retención de archivos está desactivada por defecto (0 días) y solo se activa si la configuras en Ajustes.

| Qué | Dónde |
| --- | --- |
| Base de datos, trabajos, clips, composiciones y ajustes | `~/Library/Application Support/Snipra/storage/` |
| Modelo de transcripción base | `~/Library/Application Support/Snipra/models/` |
| Registros | `~/Library/Logs/Snipra/` |
| Caché de fuentes | `~/Library/Caches/Snipra/` |

Nunca se escriben datos dentro de `Snipra.app`.

## Registros

- Botón **Ver registros** en el lanzador, o `open ~/Library/Logs/Snipra`.
- `launcher.log`: arranque, puertos y parada.
- `backend-api.log`: API FastAPI y errores de peticiones.
- `backend-worker.log`: procesamiento de trabajos.

## Actualizar sin perder datos

1. Cierra Snipra (Cmd+Q).
2. Reemplaza `Snipra.app` en `/Applications` por la nueva versión.
3. Abre Snipra de nuevo.

Los trabajos, clips y ajustes están en `Application Support`, que la actualización
no modifica. Antes de una actualización importante, copia esa carpeta:

```sh
cp -R "$HOME/Library/Application Support/Snipra" ~/Desktop/Snipra-respaldo
```

## Desinstalar

Borra `Snipra.app`. Para eliminar también los datos, borra
`~/Library/Application Support/Snipra`, `~/Library/Logs/Snipra` y
`~/Library/Caches/Snipra`. Esta acción no se puede deshacer.

## Limitaciones conocidas

- Solo arm64. Mínimo declarado macOS 27.0 (verificado en 27.0.1); el mínimo técnico de los binarios es 15.7.9.
- El seguimiento de sujeto por visión (extra `vision`: OpenCV, YOLO, MediaPipe) no
  está incluido. El reencuadre usa centrado o desenfoque como respaldo.
- Los modelos de transcripción `small` y `medium` (perfiles equilibrado y preciso)
  se descargan de Hugging Face la primera vez que se eligen. Requieren internet.
- Los proveedores de IA externos solo se usan si los configuras en **Ajustes**. El
  valor por defecto es el análisis heurístico local, sin claves.
- Para descargas por URL, el modo de cookies del navegador puede leer la sesión del
  navegador, igual que en la versión de Windows. Puedes cambiarlo a *ninguno* en
  **Ajustes**.

## Para desarrolladores

El flujo de desarrollo con el checkout (Vite, `uv` y `scripts/start-snipra.sh`) no
cambia y no forma parte de la app empaquetada. El código de Snipra (MIT) va legible dentro de
`Snipra.app/Contents/Resources/app`.
