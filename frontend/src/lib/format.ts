export function formatDuration(seconds: number | null | undefined): string {
  if (!seconds || seconds < 0) return '—'
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

/** Resolución real descargada a etiqueta corta: 2160 -> "4K · 2160p", 1080 -> "1080p". */
export function formatResolution(height: number | null | undefined): string {
  if (!height || height <= 0) return '—'
  const tier = height >= 2160 ? '4K · ' : height >= 1440 ? '2K · ' : ''
  return `${tier}${height}p`
}

export function formatBytes(bytes: number): string {
  if (!bytes) return '—'
  const mb = bytes / (1024 * 1024)
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${(bytes / 1024).toFixed(0)} KB`
}

const DATE_TIME_FORMATTER = new Intl.DateTimeFormat('es-PA', {
  day: '2-digit',
  month: 'short',
  year: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
})

/** Fecha y hora del backend convertidas a la zona horaria local del navegador. */
export function formatDateTime(value: string | null | undefined): string {
  if (!value) return 'Fecha desconocida'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return 'Fecha desconocida'
  return DATE_TIME_FORMATTER.format(date)
}

const PROVIDER_LABELS: Record<string, string> = {
  claude: 'Claude · API key',
  openai: 'OpenAI · API key',
  nvidia: 'Nvidia NIM (gratis)',
  claude_code: 'Claude Code · suscripción',
  codex: 'Codex · suscripción ChatGPT',
  naive: 'Sin IA · heurístico offline',
}

export function providerLabel(provider: string): string {
  return PROVIDER_LABELS[provider] ?? provider
}

const DETECTION_SOURCE_LABELS: Record<string, string> = {
  llm: 'Deteccion IA',
  fallback: 'Fallback local',
  mixed: 'Seleccion mixta',
  music: 'Modo musica',
  signal: 'Señal de audio/escena',
}

export function detectionSourceLabel(value: string): string {
  return DETECTION_SOURCE_LABELS[value] ?? value
}

// Snipra corre en Windows y en macOS, y los inputs de ruta piden una ruta local
// real del equipo. Un ejemplo con "C:\Users\..." en Mac desorienta, asi que el
// placeholder sigue al SO. userAgentData solo existe en Chromium; el userAgent
// cubre Safari y Firefox.
const IS_MAC = /mac/i.test(
  (navigator as { userAgentData?: { platform?: string } }).userAgentData?.platform ??
    navigator.userAgent,
)

/** Ruta de ejemplo para el input de musica de fondo, segun el SO del usuario. */
export const BGM_PATH_PLACEHOLDER = IS_MAC
  ? '/Users/tu-usuario/Music/musica.mp3'
  : 'C:\\Users\\...\\musica.mp3'

/** Ruta de ejemplo para el input del cookies.txt exportado, segun el SO. */
export const COOKIES_PATH_PLACEHOLDER = IS_MAC
  ? '/Users/tu-usuario/Downloads/cookies.txt'
  : 'C:\\ruta\\cookies.txt'

/** Rango de los sliders de subtitulos, unico para UploadPage y ClipCard. */
export const SUBTITLE_SIZE_RANGE = { min: 0.75, max: 1.3, step: 0.05 } as const
export const SUBTITLE_STROKE_RANGE = { min: 0, max: 14, step: 1 } as const

/**
 * Posicion del slider como "n/total" en vez del valor crudo.
 *
 * El valor real de Tamaño es un multiplicador (0.75-1.3 de 0.05 en 0.05): no se
 * mostraba, asi que no habia forma de saber en que punto estabas ni de repetir
 * un ajuste que habia quedado bien. Se numera la posicion, igual que hace el
 * slider de "Numero de clips".
 */
export function sliderStep(
  value: number,
  range: { min: number; max: number; step: number },
): { index: number; total: number } {
  const total = Math.round((range.max - range.min) / range.step) + 1
  const raw = Math.round((value - range.min) / range.step) + 1
  return { index: Math.min(total, Math.max(1, raw)), total }
}
