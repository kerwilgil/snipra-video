import { useEffect, useState } from 'react'
import { IconMonitor, IconMoon, IconSun } from './icons'

type Theme = 'dark' | 'light' | 'system'
const STORAGE_KEY = 'snipra-theme'
const DARK_MEDIA_QUERY = '(prefers-color-scheme: dark)'

function getInitialTheme(): Theme {
  const stored = localStorage.getItem(STORAGE_KEY)
  if (stored === 'light' || stored === 'dark' || stored === 'system') return stored
  return 'system'
}

function systemPrefersDark(): boolean {
  return window.matchMedia(DARK_MEDIA_QUERY).matches
}

function applyTheme(theme: Theme) {
  const effectiveLight = theme === 'light' || (theme === 'system' && !systemPrefersDark())
  // dark es el default sin atributo; light se activa con data-theme="light"
  if (effectiveLight) document.documentElement.dataset.theme = 'light'
  else delete document.documentElement.dataset.theme
}

const CYCLE: Record<Theme, Theme> = { system: 'light', light: 'dark', dark: 'system' }
const LABELS: Record<Theme, string> = { system: 'Sistema', light: 'Claro', dark: 'Oscuro' }

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme)

  useEffect(() => {
    applyTheme(theme)
    localStorage.setItem(STORAGE_KEY, theme)
  }, [theme])

  // En modo "sistema", sigue los cambios de tema del SO en vivo (sin recargar).
  useEffect(() => {
    if (theme !== 'system') return
    const media = window.matchMedia(DARK_MEDIA_QUERY)
    const onChange = () => applyTheme('system')
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [theme])

  const next = CYCLE[theme]
  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      aria-label={`Tema: ${LABELS[theme]}. Cambiar a ${LABELS[next]}`}
      title={`Tema: ${LABELS[theme]} (clic para ${LABELS[next]})`}
      className="flex h-9 w-9 shrink-0 cursor-pointer items-center justify-center rounded-lg border border-border text-ink-soft transition-colors hover:bg-raised/60 hover:text-ink"
    >
      {theme === 'system' && <IconMonitor className="h-[18px] w-[18px] text-ink-soft" />}
      {theme === 'light' && <IconSun className="h-[18px] w-[18px] text-amber" />}
      {theme === 'dark' && <IconMoon className="h-[18px] w-[18px] text-blue" />}
    </button>
  )
}
