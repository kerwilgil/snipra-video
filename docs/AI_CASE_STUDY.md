# Caso de estudio: una dependencia que faltaba en producción

Snipra es una aplicación para procesar vídeo y producir clips. En una revisión de su distribución para macOS se documentó un problema típico de empaquetado Python: un módulo requerido por la aplicación estaba declarado solo como dependencia de desarrollo. El entorno de desarrollo podía tenerlo instalado, pero una instalación de producción sin ese grupo fallaba al importar el backend.

La corrección trasladó `jsonschema` a las dependencias de ejecución y actualizó el lock sin cambiar versiones resueltas. El commit privado `0bbd547` registra el cambio y atribuye coautoría a Claude Haiku 5.5. Esa atribución es evidencia del registro del proyecto; no revela los prompts, la distribución exacta del trabajo ni una medida de productividad.

El resultado comprobable en el historial es el cambio de clasificación de la dependencia. El mensaje del commit describe el fallo de importación y su causa; esta auditoría no reprodujo independientemente el fallo anterior ni una instalación macOS empaquetada. La enseñanza para otros equipos es verificar el arranque con las dependencias de producción, no solo en entornos que también contienen herramientas de desarrollo.

Codex se utilizó posteriormente para inspeccionar el commit y preparar esta documentación. La revisión y decisión de publicación corresponden a la persona responsable del proyecto. Los identificadores del repositorio privado se incluyen como referencias verificables por sus responsables; el lector no necesita acceso al código para entender el caso.
