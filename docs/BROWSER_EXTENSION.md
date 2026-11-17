# Browser Extension — Snipra 1.0.0

Estado: pre-release. Extensión Manifest V3 de uso local.

“Detecta videos en la página actual y envía su URL a Snipra para descargarlos y procesarlos.”

Contrato **PAGE_URL_ONLY**:

```text
Browser Extension → detecta contexto/video → envía URL de página
                  → backend Snipra → yt-dlp/pipeline → descarga/procesamiento
```

HTML5, OpenGraph, JSON-LD y embeds aportan contexto. La fuente enviada es la URL
de la pestaña activa. Los blobs o URLs multimedia detectados no reemplazan esa URL.
No es un interceptor de streams, no descarga blobs directamente y no rompe DRM.
El backend depende de la compatibilidad del extractor y del acceso a la fuente.

## Instalar y conectar

1. Inicia Snipra localmente.
2. Abre `chrome://extensions`, `edge://extensions` o `brave://extensions`.
3. Activa modo desarrollador, carga descomprimida y selecciona `extension/` del checkout.
4. Abre el popup y pulsa **Conectar con Snipra**.
5. Aprueba la conexión en la página local de Snipra.

También existe token manual desde Ajustes → Extensión de navegador. Es una
credencial local y no debe incorporarse a Git ni a screenshots.

## Usar

Abre una página con video, pulsa el icono y elige clips, descargar video o audio.
Puedes seleccionar perfiles, formatos, calidad y opciones de subtítulos.
En Instagram con video, la acción predeterminada es **Descargar video**.
El popup presenta aceptación, progreso, resultado y acceso al archivo/trabajo.
Puedes cerrarlo y volver a consultar el job de la misma página.
Las páginas sin video no deben ofrecer descarga activa.

Permisos: `activeTab`, `storage`, `scripting`, y hosts de API local en
localhost/127.0.0.1, puertos 8000 y 8001. Sin permiso `cookies` ni `webRequest`.
La sesión para Instagram se configura en el backend, no se lee desde la extensión.
Consulta el estado físico real en la documentación de Instagram.
