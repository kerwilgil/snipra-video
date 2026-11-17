# Snipra

Snipra es una aplicaci?n local para importar v?deo, generar clips, componer y exportar medios. Incluye un frontend React, una API FastAPI, un worker y una extensi?n para navegadores Chromium. Los servicios de IA externos son opcionales y requieren configuraci?n y selecci?n expl?citas.

## Desarrollo en Windows

Requisitos: Python 3.11, uv, Node.js, FFmpeg y ffprobe en PATH.

```powershell
git clone https://github.com/kerwilgil/snipra-video.git
cd snipra-video
cd backend
uv sync --locked --python 3.11
cd ../frontend
npm ci
cd ..
powershell -ExecutionPolicy Bypass -File scripts/start-snipra.ps1 -NoVision -NoOpen
```

Abre `http://localhost:5173`. La extensi?n se carga descomprimida desde `extension/` y necesita emparejarse con la aplicaci?n local; consulta [su gu?a](docs/BROWSER_EXTENSION.md). La configuraci?n y los trabajos se guardan localmente y no forman parte de este repositorio.

## Documentaci?n

- [Arquitectura](docs/ARCHITECTURE.md)
- [Cat?logo NVIDIA](docs/NVIDIA_CATALOG.md)
- [Flujo de trabajo con IA](docs/AI_WORKFLOW.md)
- [Caso de estudio con IA](docs/AI_CASE_STUDY.md)

El c?digo se distribuye bajo [MIT](LICENSE). Las dependencias, FFmpeg y los modelos tienen licencias propias. El c?digo fuente no constituye un instalador firmado; el empaquetado macOS requiere revisi?n de licencias, firma y notarizaci?n antes de distribuir binarios.
