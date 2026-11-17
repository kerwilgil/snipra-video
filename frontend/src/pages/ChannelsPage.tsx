import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { ConfirmDialog } from '../components/ConfirmDialog'
import { IconLink, IconPlay, IconPlus, IconTrash, IconX } from '../components/icons'
import { Select } from '../components/Select'
import { ToggleSwitch } from '../components/ToggleSwitch'
import { Spinner } from '../components/ui'
import { TopbarTitle } from '../components/topbar'
import { api } from '../lib/api'
import { providerLabel } from '../lib/format'
import {
  GAMING_OUTPUT_OPTIONS,
  isInternalGamingMode,
  resolveGamingProcessingMode,
  type GamingOutput,
} from '../lib/gaming'
import type { ProcessingMode, TrackedChannel } from '../lib/types'

const POLL_INTERVAL_OPTIONS = [
  { value: '15', label: 'Cada 15 minutos' },
  { value: '30', label: 'Cada 30 minutos' },
  { value: '60', label: 'Cada hora' },
  { value: '180', label: 'Cada 3 horas' },
  { value: '360', label: 'Cada 6 horas' },
  { value: '1440', label: 'Una vez al dia' },
]

const FORMAT_OPTIONS = ['9x16', '1x1', '16x9']

function sourceHost(url: string) {
  try { return new URL(url).hostname.replace(/^www\./, '') }
  catch { return 'Fuente' }
}

export function ChannelsPage() {
  const qc = useQueryClient()
  const { data: settings } = useQuery({ queryKey: ['settings'], queryFn: api.getSettings })
  const { data: channels, isLoading, isError: channelsError, isFetching, refetch } = useQuery({
    queryKey: ['channels'],
    queryFn: api.listChannels,
    refetchInterval: 15000,
  })

  const [url, setUrl] = useState('')
  const [label, setLabel] = useState('')
  const [formats, setFormats] = useState<string[]>(['9x16'])
  const [numClips, setNumClips] = useState(5)
  const [pollInterval, setPollInterval] = useState('60')
  const [provider, setProvider] = useState('')
  const [processingMode, setProcessingMode] = useState<ProcessingMode>('auto_smart')
  const [gamingOutput, setGamingOutput] = useState<GamingOutput>('auto')
  const [error, setError] = useState<string | null>(null)
  const [channelActionError, setChannelActionError] = useState<string | null>(null)
  const [showForm, setShowForm] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<TrackedChannel | null>(null)
  const [checkFeedback, setCheckFeedback] = useState<{
    channelId: string
    success: boolean
    message: string
  } | null>(null)

  const providers = settings?.providers ?? ['naive']
  const processingModes = (settings?.processing_modes ?? []).filter(
    (mode) => !isInternalGamingMode(mode.value),
  )
  const isGaming = processingMode === 'gaming'
  const isSingleGamingEdit = isGaming && (gamingOutput === 'hook' || gamingOutput === 'highlights')
  const effectiveNumClips = isSingleGamingEdit ? 1 : numClips
  const effectiveProcessingMode = resolveGamingProcessingMode(processingMode, gamingOutput)

  const createChannel = useMutation({
    mutationFn: api.createChannel,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['channels'] })
      setUrl('')
      setLabel('')
      setError(null)
      setShowForm(false)
    },
    onError: (e) => setError(String(e instanceof Error ? e.message : e)),
  })

  const toggleChannel = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      api.updateChannel(id, { enabled }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['channels'] }),
    onError: (e) => setChannelActionError(e instanceof Error ? e.message : String(e)),
  })

  const deleteChannel = useMutation({
    mutationFn: api.deleteChannel,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['channels'] }),
    onError: (e) => setChannelActionError(e instanceof Error ? e.message : String(e)),
  })

  const checkNow = useMutation({
    mutationFn: api.checkChannelNow,
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ['channels'] })
      const res = data.check_result as Record<string, unknown> | undefined
      if (res) {
        const isSuccess = Boolean(res.success)
        const msg = isSuccess
          ? `Chequeo exitoso: ${res.videos_seen ?? 0} videos vistos, ${res.new_videos ?? 0} nuevos, ${res.jobs_created ?? 0} jobs creados${res.first_check ? ' (primer chequeo)' : ''}.`
          : `Fallo en el chequeo: ${res.error_message ?? res.error_code ?? 'Error desconocido'}`
        setCheckFeedback({ channelId: data.id, success: isSuccess, message: msg })
      }
    },
    onError: (e) => {
      setCheckFeedback({
        channelId: '',
        success: false,
        message: `Error al chequear canal: ${e instanceof Error ? e.message : String(e)}`,
      })
    },
  })

  function toggleFormat(f: string) {
    setFormats((prev) => (prev.includes(f) ? prev.filter((x) => x !== f) : [...prev, f]))
  }

  function submit() {
    setError(null)
    if (!url.trim()) return setError('Pega la URL del canal, playlist o VODs.')
    if (formats.length === 0) return setError('Elige al menos un formato.')
    createChannel.mutate({
      url: url.trim(),
      label: label.trim() || undefined,
      formats,
      num_clips: effectiveNumClips,
      poll_interval_minutes: Number(pollInterval),
      provider: provider || undefined,
      processing_mode: effectiveProcessingMode,
    })
  }

  return (
    <div className="page-container section-stack channels-page">
      <TopbarTitle>
        <span className="topbar-context">Automatizacion</span>
        <strong>Canales trackeados</strong>
      </TopbarTitle>
      <header className="page-header channels-header">
        <div>
          <h1>Canales trackeados</h1>
          <p className="page-description">Snipra revisa estos canales y encola clips cuando publican un video nuevo. El historial anterior no se procesa.</p>
        </div>
        <button type="button" className={showForm ? 'btn-secondary' : 'btn-primary'} aria-expanded={showForm} aria-controls="add-channel-form" onClick={() => { if (!showForm) createChannel.reset(); setShowForm((value) => !value) }}>
          {showForm ? <IconX /> : <IconPlus />} {showForm ? 'Cerrar formulario' : 'Agregar canal'}
        </button>
      </header>

      {channels && <div className="channels-overview" aria-label="Resumen de canales">
        <div><strong>{channels.filter((channel) => channel.enabled).length}</strong><span>Canales activos</span></div>
        <div><strong>{channels.filter((channel) => channel.last_check_status === 'error').length}</strong><span>Canales con error</span></div>
        <div><strong className="channels-overview-date">{channels.some((channel) => channel.last_checked_at) ? new Date(Math.max(...channels.map((channel) => channel.last_checked_at ? new Date(channel.last_checked_at).getTime() : 0))).toLocaleString() : 'Pendiente'}</strong><span>Último chequeo</span></div>
      </div>}

      {showForm && <section id="add-channel-form" className="section-card section-stack form-card">
        <h2 className="section-heading">Agregar canal</h2>
        <div className="form-section">
          <h3>Identidad</h3>
          <div className="form-grid">
            <label className="form-field">
              <span>URL del canal</span>
              <span className="relative">
                <IconLink className="pointer-events-none absolute left-3.5 top-1/2 h-[18px] w-[18px] -translate-y-1/2 text-ink-mute" />
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://youtube.com/@canal"
                  className="field h-12 w-full pl-11 pr-4 text-sm"
                />
              </span>
            </label>
            <label className="form-field">
              <span>Nombre para identificarlo <span className="field-optional">(opcional)</span></span>
              <input
                type="text"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="Ej. Mi canal"
                className="field h-12 w-full px-4 text-sm"
              />
            </label>
          </div>
        </div>

        <div className="form-section">
          <h3>Clips y revisión</h3>
          <div className="channel-review-layout">
            <div className="channel-format-field">
              <span className="channel-field-label">Formatos de salida</span>
              <div className="channel-format-options" role="group" aria-label="Formatos de salida">
              {FORMAT_OPTIONS.map((f) => (
                <button
                  key={f}
                  type="button"
                  onClick={() => toggleFormat(f)}
                  aria-pressed={formats.includes(f)}
                  className={formats.includes(f) ? 'active' : ''}
                >
                  {f}
                </button>
              ))}
              </div>
            </div>
            <div className="channel-review-row">
              <label className="channel-range-field">
                <span className="channel-field-label">Clips por video nuevo</span>
                <span className="channel-range-control">
                  <input
                    type="range"
                    min={1}
                    max={12}
                    value={effectiveNumClips}
                    disabled={isSingleGamingEdit}
                    onChange={(e) => setNumClips(Number(e.target.value))}
                  />
                  <output className="num">{isSingleGamingEdit ? '1 edit final' : `${effectiveNumClips} clips máx.`}</output>
                </span>
              </label>
              <div className="channel-select-field">
                <span className="channel-field-label">Frecuencia de revisión</span>
                <Select
                  ariaLabel="Frecuencia de revisión"
                  value={pollInterval}
                  onChange={setPollInterval}
                  options={POLL_INTERVAL_OPTIONS}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="form-section">
          <h3>Procesamiento</h3>
          <div className="form-grid">
          <div className="channel-select-field">
            <span className="channel-field-label">IA de detección</span>
            <Select
              ariaLabel="IA de detección"
              value={provider}
              onChange={setProvider}
              options={[
                { value: '', label: `Por defecto · ${providerLabel(settings?.provider ?? 'naive')}` },
                ...providers.map((p) => ({ value: p, label: providerLabel(p) })),
              ]}
            />
          </div>
          <div className="channel-select-field">
            <span className="channel-field-label">Modo de procesamiento</span>
            <Select
              ariaLabel="Modo de procesamiento"
              value={processingMode}
              onChange={(v) => setProcessingMode(v as ProcessingMode)}
              options={processingModes.map((mode) => ({ value: mode.value, label: mode.label }))}
            />
            {isGaming && (
              <div className="channel-gaming-field">
                <span className="channel-field-label">Tipo de salida gaming</span>
                <Select
                  ariaLabel="Tipo de salida gaming"
                  value={gamingOutput}
                  onChange={(value) => setGamingOutput(value as GamingOutput)}
                  options={GAMING_OUTPUT_OPTIONS.map((option) => ({
                    value: option.value,
                    label: option.label,
                  }))}
                />
              </div>
            )}
          </div>
          </div>
        </div>

        {error && (
          <div role="alert" className="rounded-lg border border-bad/30 bg-bad/[0.06] px-4 py-3 text-sm text-bad">
            {error}
          </div>
        )}

        <div className="action-row">
          <button
            type="button"
            onClick={submit}
            disabled={createChannel.isPending}
            className="btn-primary disabled:cursor-not-allowed"
          >
            {createChannel.isPending ? <Spinner /> : <IconPlus className="h-[18px] w-[18px]" />}
            Agregar canal
          </button>
        </div>
      </section>}

      {createChannel.isSuccess && !showForm && <div role="status" className="rounded-lg border border-good/30 bg-good/[0.06] px-4 py-3 text-sm text-good">Canal guardado. Aparece en la lista y se revisará según la frecuencia elegida.</div>}

      {checkFeedback && (
        <div
          role="status"
          className={`flex items-center justify-between rounded-lg border px-4 py-3 text-sm ${
            checkFeedback.success
              ? 'border-good/30 bg-good/[0.06] text-good'
              : 'border-bad/30 bg-bad/[0.06] text-bad'
          }`}
        >
          <span>{checkFeedback.message}</span>
          <button
            type="button"
            onClick={() => setCheckFeedback(null)}
            className="btn-ghost"
          >
            Cerrar
          </button>
        </div>
      )}

      <h2 className="section-heading">Canales configurados</h2>
      {isLoading && <p className="page-description" role="status">Cargando canales...</p>}
      {channelsError && <div role="alert" className="page-error"><span>No se pudieron cargar los canales.</span><button type="button" className="btn-secondary" disabled={isFetching} onClick={() => void refetch()}>{isFetching ? 'Reintentando...' : 'Reintentar'}</button></div>}
      {channelActionError && <div role="alert" className="page-error"><span>{channelActionError}</span><button type="button" className="btn-secondary" onClick={() => setChannelActionError(null)}>Cerrar</button></div>}

      {channels?.length === 0 && (
        <div className="section-card flex min-h-[160px] flex-col items-center justify-center gap-2 text-center">
          <p className="text-sm text-ink-soft">Aun no trackeas ningun canal.</p>
        </div>
      )}

      <div className="channels-list grid gap-3" aria-label="Lista de canales">
        {channels?.map((channel) => (
          <ChannelRow
            key={channel.id}
            channel={channel}
            onToggle={(enabled) => toggleChannel.mutate({ id: channel.id, enabled })}
            onDelete={() => setPendingDelete(channel)}
            onCheckNow={() => checkNow.mutate(channel.id)}
            checking={checkNow.isPending && checkNow.variables === channel.id}
          />
        ))}
      </div>

      <ConfirmDialog
        open={pendingDelete !== null}
        title="Dejar de trackear canal"
        message={`Dejar de trackear "${pendingDelete?.label || pendingDelete?.url}"? Los jobs ya creados no se borran.`}
        confirmLabel="Dejar de trackear"
        onConfirm={() => {
          if (pendingDelete) deleteChannel.mutate(pendingDelete.id)
          setPendingDelete(null)
        }}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  )
}

function ChannelRow({
  channel,
  onToggle,
  onDelete,
  onCheckNow,
  checking,
}: {
  channel: TrackedChannel
  onToggle: (enabled: boolean) => void
  onDelete: () => void
  onCheckNow: () => void
  checking: boolean
}) {
  const lastChecked = channel.last_checked_at ? new Date(channel.last_checked_at) : null
  const lastSuccess = channel.last_success_at ? new Date(channel.last_success_at) : null
  const nextCheck = lastChecked
    ? new Date(lastChecked.getTime() + channel.poll_interval_minutes * 60_000)
    : null

  const status = channel.last_check_status || (channel.last_checked_at ? 'healthy' : 'pending')

  return (
    <article className="section-card channel-card">
      <div className="grid gap-4 lg:grid-cols-[1fr_auto] lg:items-center">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="channel-name">
              {channel.label || channel.channel_title || channel.url}
            </h3>
            {!channel.enabled && <span className="channel-paused">Pausado</span>}
            <HealthBadge status={status} failures={channel.consecutive_failures} />
          </div>
          <div className="channel-origin">
            <span>{sourceHost(channel.canonical_url || channel.url)}</span>
            <a href={channel.canonical_url || channel.url} target="_blank" rel="noreferrer" className="truncate text-cyan hover:underline">{channel.canonical_url || channel.url}</a>
          </div>
          <p className="channel-configuration">{channel.formats.join(' / ')} · {channel.num_clips} clips/video · cada {channel.poll_interval_minutes} min · {channel.processing_mode.replaceAll('_', ' ')}</p>
          <p className="channel-checks">Último chequeo: {lastChecked ? lastChecked.toLocaleString() : 'aún no revisado'}{nextCheck && channel.enabled ? ` · Próximo: ${nextCheck.toLocaleString()}` : ''}{lastSuccess && status === 'error' ? ` · Último exitoso: ${lastSuccess.toLocaleString()}` : ''}</p>
          {status === 'error' && channel.last_check_error && (
            <div className="mt-2 rounded-md border border-bad/30 bg-bad/[0.06] p-2 text-xs text-bad">
              {channel.last_check_error}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 lg:justify-end">
          <button
            type="button"
            onClick={onCheckNow}
            disabled={checking}
            title="Revisar ahora"
            className="btn-secondary"
          >
            {checking ? <Spinner /> : <IconPlay className="h-3.5 w-3.5" />}
            {status === 'error' ? 'Reintentar' : 'Ejecutar ahora'}
          </button>
          <ToggleSwitch
            checked={channel.enabled}
            onChange={onToggle}
            title={channel.enabled ? 'Pausar' : 'Activar'}
            ariaLabel={`${channel.enabled ? 'Pausar' : 'Activar'} canal ${channel.label || channel.url}`}
          />
          <button
            type="button"
            onClick={onDelete}
            title="Dejar de trackear"
            aria-label={`Borrar canal ${channel.label || channel.url}`}
            className="icon-button channel-delete"
          >
            <IconTrash className="h-4 w-4" />
          </button>
        </div>
      </div>
    </article>
  )
}

function HealthBadge({ status, failures }: { status: string; failures: number }) {
  if (status === 'healthy') {
    return (
      <span className="inline-flex items-center gap-1 rounded-md border border-good/40 bg-good/10 px-2 py-0.5 text-[11px] font-medium text-good">
        ● Saludable
      </span>
    )
  }
  if (status === 'error') {
    return (
      <span className="inline-flex items-center gap-1 rounded-md border border-bad/40 bg-bad/10 px-2 py-0.5 text-[11px] font-medium text-bad">
        ● Error{failures > 0 ? ` (${failures} fallo${failures === 1 ? '' : 's'})` : ''}
      </span>
    )
  }
  if (status === 'warning') {
    return (
      <span className="inline-flex items-center gap-1 rounded-md border border-warn/40 bg-warn/10 px-2 py-0.5 text-[11px] font-medium text-warn">
        ● Advertencia
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-border bg-base/40 px-2 py-0.5 text-[11px] font-medium text-ink-mute">
      Pendiente
    </span>
  )
}
