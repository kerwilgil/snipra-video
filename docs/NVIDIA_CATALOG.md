# Catálogo dinámico NVIDIA NIM

Snipra consulta `GET {NVIDIA_BASE_URL}/models` desde el backend. El catálogo no
contiene una lista compilada de IDs. El selector de texto y el de visión VIA
comparten la misma consulta y filtran por modalidad. La primera opción,
**Automático (recomendado)**, guarda exclusivamente `auto`; los IDs manuales
existentes se conservan, incluidos los retirados.

La resolución ocurre al ejecutar la tarea. Si desaparece el candidato automático,
se elige otro compatible del catálogo actualizado. Un ID manual retirado produce
un error explícito; nunca se reemplaza por otro ID ni por otro proveedor.

## Caché local

- Archivo: `storage/nvidia_catalog.json` (o el directorio `STORAGE_DIR`).
- TTL predeterminado: 300 segundos. El botón **Actualizar catálogo** fuerza una consulta.
- Sin conexión: se permite el último catálogo obtenido durante un máximo de 24 horas,
  con aviso de posible desactualización. Pasado ese límite, o sin caché, la selección
  falla con un mensaje explícito. No hay catálogo ficticio de respaldo.
- Los errores de red se reintentan como máximo cada 30 segundos automáticamente.
- Escrituras mediante archivo temporal y reemplazo atómico, compatibles con Windows/macOS.
- La caché está ligada a la URL del proveedor: cambiarla no reutiliza otro catálogo.

## Política actualizable sin build

Crear o editar `storage/nvidia_policy.json`. Se relee al resolver; no requiere
reiniciar Snipra. Este ejemplo no fija ningún modelo:

```json
{
  "ttl_seconds": 300,
  "max_stale_seconds": 86400,
  "text_priority": [],
  "vision_priority": [],
  "models": {}
}
```

`text_priority` y `vision_priority` aceptan IDs del catálogo en orden de preferencia.
Los retirados se saltan automáticamente. En `models`, cada ID puede tener
`{"modalities": ["text", "vision"], "priority": 10}`. Una lista vacía de modalidades
excluye ese modelo. Los overrides nunca crean disponibilidad: el ID tiene que
existir también en el catálogo recibido de NVIDIA.

Sin preferencias, la política ordena por prioridad, evidencia de modalidad,
indicación de chat/instrucciones y, finalmente, ID para resolver empates. En texto
prefiere modelos sin visión cuando los demás criterios coinciden. Es una regla
determinista de compatibilidad, no una clasificación de calidad basada en benchmarks.

### Límite de los metadatos del proveedor

El endpoint público de NVIDIA actualmente suele devolver solo `id`, `created` y
`owned_by`, sin declarar modalidades. Se usan las modalidades declaradas cuando
existen. En su ausencia se aplican patrones de nombres; la API expone
`compatibility_source: "inferred"` y el selector informa de esa limitación.
**La pertenencia al catálogo no prueba que un modelo acepte imágenes con el contrato
de chat de Snipra ni que esté habilitado para una cuenta concreta.**

`vision_pattern`, `exclude_pattern` y los overrides `models` permiten corregir la
clasificación sin cambiar código ni recompilar. No se han ejecutado inferencias
de prueba contra cada modelo: la clasificación inferida es una limitación conocida.

## Registro y compatibilidad

- Los jobs NVIDIA guardan el ID resuelto en `job.model` antes de detectar y registran
  la correspondencia `auto → ID` en su log. Los ajustes globales continúan en `auto`.
- Las ejecuciones del Agente IA incluyen las resoluciones NVIDIA en
  `EditorAgentRun.verification.model_usage`; VIA incluye la identidad visual en su
  procedencia habitual. También se registra la resolución en el logger del backend.
- Se conservan los modelos manuales guardados previamente y los proveedores existentes.
- NVIDIA para visión sigue siendo una elección explícita para VIA; no habilita el
  reranking visual ni cambia automáticamente a APIs de pago.

Referencia del contrato del proveedor:
[NVIDIA NIM: API Reference](https://docs.nvidia.com/nim/large-language-models/latest/api-reference.html).
