# Desarrollo asistido por IA

Este documento separa la evidencia disponible de los detalles aún no documentados. Snipra es una aplicación de procesamiento de vídeo; este documento acompaña la publicación del código fuente.

## Evidencia verificable

- El commit `0bbd547` atribuye coautoría a Claude Haiku 5.5. Corrige una dependencia de ejecución ausente: `jsonschema` pasó del grupo de desarrollo a las dependencias de la aplicación. El mensaje explica el fallo de importación en instalaciones sin dependencias de desarrollo. La atribución acredita lo registrado en Git, no permite reconstruir una conversación ni cuantificar la contribución.
- Los commits `51f1e89` y `6e3fbfa` también registran coautoría de Claude Haiku 5.5 en el empaquetado y la recuperación del lanzador macOS. Sus mensajes describen implementación y verificaciones; no constituyen por sí solos una reproducción independiente de esos resultados.
- Codex se utilizó en la auditoría que preparó este documento y el caso de estudio: inspección local de código y metadatos, ejecución de comprobaciones y redacción. Los resultados y límites de esa auditoría deben acompañar cualquier publicación; esta sesión no acredita uso de Codex en commits anteriores.

## Método

El flujo observado para esta auditoría fue inspeccionar primero el estado de Git y la evidencia existente, ejecutar las comprobaciones disponibles, contrastar los hallazgos con archivos y commits, y redactar sin exponer código o credenciales. La IA propone y ejecuta pasos dentro de las instrucciones recibidas; la persona responsable decide qué divulgar y conserva el control del repositorio. Una atribución en Git no demuestra por sí sola quién tomó cada decisión o revisó cada línea.

El ejemplo de `jsonschema` muestra una lección concreta: una instalación de desarrollo puede ocultar una dependencia necesaria en producción. Conviene comprobar instalaciones sin extras de desarrollo y probar el arranque del artefacto empaquetado, además de ejecutar la suite habitual.

## Pendiente de acreditar

No se dispone aquí de prompts históricos, transcripciones, tiempos, métricas de productividad ni detalles verificables de revisión humana para los commits citados. No se atribuyen esos datos. Antes de publicar, el responsable debe revisar el texto y confirmar qué información del proyecto autoriza divulgar. Los identificadores de commits privados son referencias de auditoría, no enlaces públicos accesibles.
