import type { JobStatus } from '../lib/types'

const STATUS_LABEL: Record<JobStatus, string> = {
  queued: 'En cola',
  ingesting: 'Ingiriendo',
  transcribing: 'Transcribiendo',
  detecting: 'Detectando',
  rendering: 'Renderizando',
  done: 'Listo',
  failed: 'Error',
  canceled: 'Cancelado',
}

const STATUS_STYLE: Record<JobStatus, string> = {
  queued: 'bg-white/5 text-ink-soft ring-white/10',
  ingesting: 'bg-cyan/10 text-cyan ring-cyan/20',
  transcribing: 'bg-cyan/10 text-cyan ring-cyan/20',
  detecting: 'bg-violet/10 text-violet ring-violet/25',
  rendering: 'bg-warn/10 text-warn ring-warn/20',
  done: 'bg-good/10 text-good ring-good/20',
  failed: 'bg-bad/10 text-bad ring-bad/20',
  canceled: 'bg-white/5 text-ink-mute ring-white/10',
}

const ACTIVE: JobStatus[] = ['queued', 'ingesting', 'transcribing', 'detecting', 'rendering']

export function StatusBadge({ status }: { status: JobStatus }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${STATUS_STYLE[status]}`}
    >
      {ACTIVE.includes(status) && (
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
      )}
      {STATUS_LABEL[status]}
    </span>
  )
}

export function ProgressBar({ value }: { value: number }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100)
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-white/[0.06]">
      <div className="grad-brand h-full rounded-full transition-all duration-500" style={{ width: `${pct}%` }} />
    </div>
  )
}

export function Spinner() {
  return (
    <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/25 border-t-white" />
  )
}
