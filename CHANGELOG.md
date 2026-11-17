# Changelog

## 1.0.0 — candidato, aún no publicado

- Flujo local de video: ingest por URL pública o archivo, detección de clips,
  scoring determinista y export de resultados.
- Render 9:16, 1:1 y 16:9, subtítulos y composiciones sobre clips existentes.
- Decision Intelligence opcional con evidencia ya extraída, cinco tareas,
  validación de contratos, provenance y fallback offline. OFF por defecto.
- Bridge de Decision para los providers existentes OpenAI, Claude API y NVIDIA;
  `auto` hereda provider/model del Job. Validado automáticamente con mocks;
  inferencia real pendiente de un provider HTTP configurado.
- Codex y Claude Code mantienen sus detectores existentes. En Decision usan
  únicamente fallback Rules; no se anuncia soporte CLI cancelable.
- Extensión MV3 PAGE_URL_ONLY; tests de contexto, submit único y sanitización.
  Gates físicos por sitio pendientes de completar en un navegador con popup
  e instalación controlables.
- Launcher y packaging Windows 1.0.0.0 con Brand V2. Ajustes adaptados a pantallas
  estrechas y score de clip expresado sobre 100, separado de probabilidades.
- macOS conserva sus archivos de desarrollo; soporte físico de app, launcher,
  render, export y paquete pendiente. No se anuncia validación macOS.
El repositorio ofrece código fuente. Los instaladores requieren validación por separado.
