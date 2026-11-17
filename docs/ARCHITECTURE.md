# Arquitectura

El frontend React/TypeScript comunica con FastAPI mediante la API local. SQLite
persiste jobs, ajustes y proyectos. El worker `app.worker.runner` consume los jobs
y continúa el procesamiento fuera del ciclo de vida de la petición HTTP.

`backend/app/pipeline/` contiene ingest con yt-dlp, transcripción, detección,
extracción, reencuadre, subtítulos, composición y render con FFmpeg.
`backend/app/editor/` contiene proyectos, timeline, director, generación, tareas
de assets y verificación. `backend/app/providers/` resuelve proveedores explícitos.
`backend/app/video_intelligence/` administra evidencia visual y cachés.

El runtime editorial necesita `.agents/skills/snipra-video-director/`, incluidas sus
referencias. Se conserva como recurso productivo y entra en el packaging.

`storage/` se crea localmente. No forma parte del código ni de Git. El archivo
`backend/.env.example` contiene defaults sin credenciales; la configuración real
de cada instalación se mantiene fuera del repositorio.

La extensión MV3 inspecciona contexto al abrir el popup, valida opciones y envía
la URL de la página al backend local. El emparejamiento requiere aprobación humana
en Snipra. Los tokens de conexión y cookies no se incluyen en el snapshot.
