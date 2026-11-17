import { ThemeToggle } from './ThemeToggle'

interface ShellTopbarProps {
  titleRef: (node: HTMLDivElement | null) => void
  actionsRef: (node: HTMLDivElement | null) => void
  searchRef: (node: HTMLDivElement | null) => void
}

/** Zona superior unica de la app: marca, contexto de pagina, estado y acciones principales. */
export function ShellTopbar({ titleRef, actionsRef, searchRef }: ShellTopbarProps) {
  return (
    <header className="app-topbar">
      <div ref={searchRef} className="app-topbar-search" />
      <div ref={titleRef} className="app-topbar-title" />
      <div className="app-topbar-side">
        <span className="app-status-chip" title="FFmpeg / Whisper / YOLO">
          <i className="app-status-dot" aria-hidden="true" />
          Procesamiento local
        </span>
        <div ref={actionsRef} className="app-topbar-actions" />
        <ThemeToggle />
      </div>
    </header>
  )
}
