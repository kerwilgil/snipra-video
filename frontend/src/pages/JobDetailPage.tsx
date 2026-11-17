import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ClipCard } from '../components/ClipCard'
import { IconArrowLeft, IconDownload, IconSparkles, IconTrash } from '../components/icons'
import { ProgressBar, StatusBadge } from '../components/ui'
import { TopbarActions, TopbarTitle } from '../components/topbar'
import { api } from '../lib/api'
import { processingModeLabel } from '../lib/gaming'
import { detectionSourceLabel, formatDuration, formatResolution } from '../lib/format'
import { displayJobProgress, displayJobStage } from '../lib/jobProgress'
import type { Clip, Job, JobLog } from '../lib/types'

const PIPELINE_STEPS = [
  { key: 'ingest', label: 'Fuente', match: ['Analizando', 'Descargando', 'Descarga'] },
  { key: 'transcribe', label: 'Transcripcion', match: ['Transcribiendo', 'Transcripcion'] },
  { key: 'detect', label: 'Deteccion', match: ['Detectando', 'Detectados', 'Deteccion'] },
  { key: 'render', label: 'Render', match: ['Generando', 'Listo'] },
]

const ACTIVE_STATUSES = new Set<Job['status']>(['queued', 'ingesting', 'transcribing', 'detecting', 'rendering'])

function formatElapsed(milliseconds: number) {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000))
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60
  return [hours, minutes, seconds].map((value) => String(value).padStart(2, '0')).join(':')
}

function StageTimer({ job }: { job: Job }) {
  const active = ACTIVE_STATUSES.has(job.status)
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!active) return undefined
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [active])

  if (!active) return null
  const startedAt = Date.parse(job.stage_started_at || job.updated_at)
  const elapsed = Number.isFinite(startedAt) ? now - startedAt : 0
  return (
    <div className="text-right">
      <div className="text-[10px] uppercase tracking-[0.12em] text-ink-mute">Tiempo en etapa</div>
      <div className="num text-sm font-semibold text-ink">{formatElapsed(elapsed)}</div>
    </div>
  )
}

function stepState(job: Job, logs: JobLog[], index: number): 'done' | 'active' | 'pending' {
  if (job.status === 'done') return 'done'
  if (job.status === 'failed' || job.status === 'canceled') {
    return index < PIPELINE_STEPS.findIndex((step) => job.stage.includes(step.label)) ? 'done' : 'pending'
  }
  const current = PIPELINE_STEPS.findIndex((step) => step.match.some((token) => job.stage.includes(token)))
  if (current >= 0) {
    if (index < current) return 'done'
    if (index === current) return 'active'
    return 'pending'
  }
  const hasLog = logs.some((log) => PIPELINE_STEPS[index].match.some((token) => log.message.includes(token)))
  return hasLog ? 'done' : 'pending'
}

export function JobDetailPage() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const qc = useQueryClient()

  const del = useMutation({
    mutationFn: api.deleteJob,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jobs'] })
      navigate('/jobs')
    },
  })

  const retry = useMutation({
    mutationFn: api.retryJob,
    onSuccess: (updated) => {
      qc.setQueryData(['job', id], updated)
      qc.invalidateQueries({ queryKey: ['jobs'] })
      qc.invalidateQueries({ queryKey: ['job', id] })
      qc.invalidateQueries({ queryKey: ['logs', id] })
    },
  })

  const cancel = useMutation({
    mutationFn: api.cancelJob,
    onSuccess: (updated) => {
      qc.setQueryData(['job', id], updated)
      qc.invalidateQueries({ queryKey: ['jobs'] })
      qc.invalidateQueries({ queryKey: ['job', id] })
      qc.invalidateQueries({ queryKey: ['logs', id] })
    },
  })

  const manifest = useMutation({
    mutationFn: api.getExportManifest,
    onSuccess: (data) => {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `snipra-${id}-manifest.json`
      link.click()
      URL.revokeObjectURL(url)
    },
  })

  const { data: job, isLoading: jobLoading, isError: jobError, isFetching: jobFetching, refetch: refetchJob } = useQuery({
    queryKey: ['job', id],
    queryFn: () => api.getJob(id),
    refetchInterval: (q) => {
      const s = q.state.data?.status
      return s && s !== 'done' && s !== 'failed' && s !== 'canceled' ? 2000 : false
    },
  })

  const { data: logs } = useQuery({
    queryKey: ['logs', id],
    queryFn: () => api.getJobLogs(id),
    refetchInterval:
      job?.status && job.status !== 'done' && job.status !== 'failed' && job.status !== 'canceled'
        ? 2500
        : false,
  })

  const { data: clips } = useQuery({
    queryKey: ['clips', id],
    queryFn: () => api.getClips(id),
    enabled: job?.status === 'done',
  })

  if (!job) return <div className="page-container section-stack"><TopbarTitle><Link to="/jobs" className="topbar-context">Trabajos</Link><strong>Detalle</strong></TopbarTitle><header className="page-header"><h1>Detalle del trabajo</h1></header>{jobError ? <div role="alert" className="page-error"><span>No se pudo cargar el trabajo.</span><button type="button" className="btn-secondary" disabled={jobFetching} onClick={() => void refetchJob()}>{jobFetching ? 'Reintentando...' : 'Reintentar'}</button></div> : jobLoading && <p role="status" className="page-description">Cargando trabajo...</p>}</div>

  const processing = !['done', 'failed', 'canceled'].includes(job.status)
  const progress = displayJobProgress(job)
  const orderedClips = clips ?? []

  return (
    <div className="page-container section-stack">
      <TopbarTitle>
        <Link to="/jobs" className="topbar-context transition-colors hover:text-ink-soft">
          <IconArrowLeft className="inline h-3.5 w-3.5 align-[-2px]" /> Trabajos
        </Link>
        <strong>{job.title || job.source_name}</strong>
      </TopbarTitle>
      <TopbarActions>
        {processing && (
          <button
            type="button"
            onClick={() => {
              if (confirm('Cancelar este procesamiento?')) cancel.mutate(job.id)
            }}
            disabled={cancel.isPending}
            className="btn-secondary disabled:opacity-50"
          >
            {cancel.isPending ? 'Cancelando...' : 'Cancelar'}
          </button>
        )}
        {(job.status === 'failed' || job.status === 'done' || job.status === 'canceled') && (
          <button
            type="button"
            onClick={() => retry.mutate(job.id)}
            disabled={retry.isPending}
            className="btn-secondary disabled:opacity-50"
          >
            {retry.isPending ? 'Reintentando...' : 'Reintentar job'}
          </button>
        )}
        {job.status === 'done' && job.output_mode === 'clips' && (
          <button
            type="button"
            onClick={() => manifest.mutate(job.id)}
            disabled={manifest.isPending}
            className="btn-secondary disabled:opacity-50"
          >
            {manifest.isPending ? 'Exportando...' : 'Manifest JSON'}
          </button>
        )}
        <button
          type="button"
          onClick={() => {
            if (confirm('Borrar este trabajo y sus clips?')) del.mutate(job.id)
          }}
          disabled={del.isPending}
          className="btn-danger disabled:opacity-50"
        >
          <IconTrash className="h-3.5 w-3.5" />
          Borrar
        </button>
      </TopbarActions>

      <header className="page-header">
        <h1>{job.title || job.source_name}</h1>
        <p className="page-description">Estado y resultados del trabajo.</p>
      </header>

      <section className="section-card">
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge status={job.status} />
          <Meta>{job.output_mode === 'download' ? 'Descarga original' : job.output_mode === 'audio' ? 'Extraccion de audio' : processingModeLabel(job.processing_mode)}</Meta>
          {job.output_mode === 'clips' && <Meta>{job.captions_enabled ? job.caption_style : 'Sin subtitulos'}</Meta>}
          {job.output_mode === 'clips' && job.detection_source && <Meta>{detectionSourceLabel(job.detection_source)}</Meta>}
        </div>
        <p className="num mt-3 text-sm text-ink-mute">
          {job.output_mode === 'download'
            ? `Original / ${job.download_quality === 'auto' ? 'mejor disponible' : `hasta ${job.download_quality}p`} / ${formatDuration(job.duration)}`
            : job.output_mode === 'audio'
              ? `Audio / ${(job.audio_format || 'mp3').toUpperCase()} / ${formatDuration(job.duration)}`
              : `${job.formats.join(' / ')} / ${job.num_clips} clips / IA: ${job.provider} / ${formatDuration(job.duration)}`}
          {job.source_height > 0 && ` / fuente ${formatResolution(job.source_height)}`}
        </p>
      </section>

      <JobProgressStrip job={job} logs={logs ?? []} progress={progress} />

      <QualityWarningBanner logs={logs ?? []} />

      {job.status === 'done' && (job.output_mode === 'download'
        ? <DownloadResult job={job} />
        : job.output_mode === 'audio'
          ? <AudioResult job={job} />
          : <ClipLibrary clips={orderedClips} jobId={job.id} />)}

      {job.status === 'failed' && (
        <div className="rounded-lg border border-bad/30 bg-bad/[0.06] p-5 text-sm">
          <div className="font-semibold text-bad">El procesamiento fallo</div>
          <div className="mt-2 overflow-auto rounded-md bg-base/60 p-3 font-mono text-xs text-ink-soft">
            {job.error}
          </div>
        </div>
      )}

      {job.status === 'canceled' && (
        <div className="section-card text-sm text-ink-soft">
          Este trabajo fue cancelado. Puedes reintentarlo cuando quieras.
        </div>
      )}

      <JobLogs logs={logs ?? []} />
    </div>
  )
}

function ClipLibrary({ clips, jobId }: { clips: Clip[]; jobId: string }) {
  return (
    <section className="space-y-4">
      <div className="flex items-center gap-2">
        <IconSparkles className="h-5 w-5 text-violet" />
        <h2 className="text-lg font-semibold text-ink">Clips generados ({clips.length})</h2>
      </div>

      {clips.length === 0 && <p className="text-sm text-ink-mute">No se generaron clips para este video.</p>}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {clips.map((clip) => <ClipCard key={clip.id} clip={clip} jobId={jobId} />)}
      </div>
    </section>
  )
}

function DownloadResult({ job }: { job: Job }) {
  return (
    <section className="panel flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <div className="text-lg font-semibold text-ink">Video listo</div>
        <p className="mt-1 text-sm text-ink-soft">El archivo original se conservo sin clips, IA ni subtitulos.</p>
      </div>
      <a
        href={api.downloadOriginalUrl(job.id)}
        download
        className="btn-primary shrink-0"
      >
        <IconDownload className="h-[18px] w-[18px]" /> Descargar video
      </a>
    </section>
  )
}

function AudioResult({ job }: { job: Job }) {
  return (
    <section className="panel flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <div className="text-lg font-semibold text-ink">Audio listo</div>
        <p className="mt-1 text-sm text-ink-soft">Solo la pista de audio, sin video ni subtitulos.</p>
      </div>
      <a
        href={api.downloadOriginalUrl(job.id)}
        download
        className="btn-primary shrink-0"
      >
        <IconDownload className="h-[18px] w-[18px]" /> Descargar audio ({(job.audio_format || 'mp3').toUpperCase()})
      </a>
    </section>
  )
}

function JobProgressStrip({ job, logs, progress }: { job: Job; logs: JobLog[]; progress: number }) {
  const pct = Math.round(Math.min(1, Math.max(0, progress)) * 100)
  if (job.output_mode === 'download' || job.output_mode === 'audio') {
    const isAudio = job.output_mode === 'audio'
    return (
      <section className="panel p-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <div className="text-sm font-semibold text-ink">{displayJobStage(job)}</div>
            <div className="mt-1 text-xs text-ink-mute">
              {isAudio ? 'Extraccion de audio, sin IA ni render.' : 'Descarga directa sin IA ni render.'}
              {!isAudio && job.source_height > 0 && ` Resolucion real: ${formatResolution(job.source_height)}.`}
            </div>
          </div>
          <div className="flex items-center gap-4">
            <StageTimer job={job} />
            <div className="num text-2xl font-semibold text-cyan">{pct}%</div>
          </div>
        </div>
        <ProgressBar value={progress} />
      </section>
    )
  }
  return (
    <section className="panel p-4">
      <div className="mb-3 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge status={job.status} />
            <div className="text-sm font-semibold text-ink">{displayJobStage(job)}</div>
          </div>
          <div className="num mt-1 truncate text-xs text-ink-mute">
            {job.formats.join(' / ')} / {job.num_clips} clips / IA: {job.provider} / {job.model || 'default'}
            {job.source_height > 0 && ` / fuente ${formatResolution(job.source_height)}`}
          </div>
        </div>
        <div className="flex items-center gap-4">
          <StageTimer job={job} />
          <div className="num text-2xl font-semibold text-cyan">{pct}%</div>
        </div>
      </div>

      <ProgressBar value={progress} />

      <div className="mt-4 grid gap-2 md:grid-cols-4">
        {PIPELINE_STEPS.map((step, index) => {
          const state = stepState(job, logs, index)
          return (
            <div
              key={step.key}
              className={`rounded-lg border px-3 py-2 ${
                state === 'done'
                  ? 'border-good/25 bg-good/[0.06]'
                  : state === 'active'
                    ? 'border-cyan/35 bg-cyan/[0.08]'
                    : 'border-border bg-base/30'
              }`}
            >
              <div className="text-xs font-semibold text-ink">{step.label}</div>
              <div className="mt-0.5 text-[10px] uppercase tracking-[0.12em] text-ink-mute">{state}</div>
            </div>
          )
        })}
      </div>
    </section>
  )
}

function QualityWarningBanner({ logs }: { logs: JobLog[] }) {
  const warning = logs.find((log) => log.message.startsWith('Calidad pedida:'))
  if (!warning) return null
  return (
    <div
      role="alert"
      className="rounded-lg border border-warn/30 bg-warn/[0.08] px-4 py-3 text-sm text-warn"
    >
      <span className="font-semibold">Calidad de la fuente por debajo de lo pedido. </span>
      {warning.message.replace('Calidad pedida:', '').trim()}
    </div>
  )
}

function JobLogs({ logs }: { logs: Array<{ id: string; level: string; message: string; created_at: string }> }) {
  if (logs.length === 0) return null
  const recent = logs.slice(-8).reverse()
  return (
    <details className="panel p-4">
      <summary className="cursor-pointer text-sm font-semibold text-ink">Logs tecnicos ({logs.length})</summary>
      <div className="mt-3 space-y-2">
        {recent.map((log) => (
          <div key={log.id} className="grid gap-2 text-xs text-ink-mute md:grid-cols-[132px_1fr]">
            <span className="num">{new Date(log.created_at).toLocaleTimeString()}</span>
            <span className={log.level === 'error' ? 'text-bad' : log.level === 'warning' ? 'text-warn' : ''}>
              {log.message}
            </span>
          </div>
        ))}
      </div>
    </details>
  )
}

function Meta({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-md border border-border bg-base/40 px-2 py-1 text-[11px] text-ink-mute">
      {children}
    </span>
  )
}
