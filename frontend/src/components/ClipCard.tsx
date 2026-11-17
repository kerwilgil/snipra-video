import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import {
  IconDownload,
  IconPencil,
  IconPlay,
  IconRefresh,
  IconScissors,
  IconSparkles,
} from './icons'
import { Spinner } from './ui'
import { api } from '../lib/api'
import {
  SUBTITLE_SIZE_RANGE,
  SUBTITLE_STROKE_RANGE,
  detectionSourceLabel,
  formatBytes,
  formatDuration,
  sliderStep,
} from '../lib/format'
import type { Clip, DeliveryQCResult, SubtitleBlock, SubtitleStyleConfig } from '../lib/types'

const DEFAULT_STYLE_CONFIG: SubtitleStyleConfig = {
  style: 'viral_bold',
  size: 1,
  position: 'lower_middle',
  text_color: '#FFFFFF',
  highlight_color: '#FFD21F',
  secondary_highlight_color: '#00D4FF',
  stroke_color: '#000000',
  stroke_width: 8,
  shadow_enabled: true,
  background_box: false,
  uppercase: true,
  emoji_enabled: false,
  highlight_max_words: 3,
}

const LANGUAGE_OPTIONS = [
  { value: 'Ingles', label: 'Ingles' },
  { value: 'Portugues', label: 'Portugues' },
  { value: 'Frances', label: 'Frances' },
  { value: 'Aleman', label: 'Aleman' },
  { value: '__custom__', label: 'Otro...' },
]

export function ClipCard({ clip, jobId }: { clip: Clip; jobId: string }) {
  const qc = useQueryClient()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState({
    start: clip.start,
    end: clip.end,
    title: clip.title,
    hook: clip.hook,
  })
  const asset = clip.assets[0]
  const score = Math.round(clip.score * 100)
  const [copied, setCopied] = useState(false)
  const [transcriptOpen, setTranscriptOpen] = useState(false)
  const [styleOpen, setStyleOpen] = useState(false)
  const [subtitleOpen, setSubtitleOpen] = useState(false)
  const [timelineOpen, setTimelineOpen] = useState(false)
  const [translateOpen, setTranslateOpen] = useState(false)
  const [styleDraft, setStyleDraft] = useState<SubtitleStyleConfig>(DEFAULT_STYLE_CONFIG)

  const copyCaption = async () => {
    const text = [clip.description || clip.hook, clip.hashtags.join(' ')].filter(Boolean).join('\n\n')
    if (!text) return
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  const update = useMutation({
    mutationFn: () => api.updateClip(clip.id, draft),
    onSuccess: () => {
      setEditing(false)
      qc.invalidateQueries({ queryKey: ['clips', jobId] })
      qc.invalidateQueries({ queryKey: ['logs', jobId] })
    },
  })

  const rerender = useMutation({
    mutationFn: () => api.rerenderClip(clip.id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['clips', jobId] })
      qc.invalidateQueries({ queryKey: ['logs', jobId] })
    },
  })

  const removeFillers = useMutation({
    mutationFn: () => api.reeditClip(clip.id, { remove_fillers: true }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['clips', jobId] })
      qc.invalidateQueries({ queryKey: ['logs', jobId] })
    },
  })

  const removeRepeatedTakes = useMutation({
    mutationFn: () => api.reeditClip(clip.id, { remove_repeated_takes: true }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['clips', jobId] })
      qc.invalidateQueries({ queryKey: ['logs', jobId] })
    },
  })

  const removeLongPauses = useMutation({
    mutationFn: () => api.reeditClip(clip.id, { remove_long_pauses: true }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['clips', jobId] })
      qc.invalidateQueries({ queryKey: ['logs', jobId] })
    },
  })

  const rewriteCaptions = useMutation({
    mutationFn: () => api.rewriteClipCaptions(clip.id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['clips', jobId] })
      qc.invalidateQueries({ queryKey: ['logs', jobId] })
      qc.invalidateQueries({ queryKey: ['timeline', clip.id] })
    },
  })

  const translateCaptions = useMutation({
    mutationFn: (vars: { targetLanguage: string; applyToAll: boolean }) =>
      api.translateClipCaptions(clip.id, {
        target_language: vars.targetLanguage,
        apply_to_all: vars.applyToAll,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['clips', jobId] })
      qc.invalidateQueries({ queryKey: ['logs', jobId] })
      qc.invalidateQueries({ queryKey: ['timeline', clip.id] })
      setTranslateOpen(false)
    },
  })

  const applyStyle = useMutation({
    mutationFn: (applyToAll: boolean) =>
      api.editClipStyle(clip.id, {
        style: styleDraft.style,
        style_config: styleDraft,
        apply_to_all: applyToAll,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['clips', jobId] })
      qc.invalidateQueries({ queryKey: ['logs', jobId] })
    },
  })

  const applyPrivacy = useMutation({
    mutationFn: () => api.applyClipPrivacy(clip.id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['clips', jobId] })
      qc.invalidateQueries({ queryKey: ['logs', jobId] })
      qc.invalidateQueries({ queryKey: ['timeline', clip.id] })
    },
  })

  const isProcessing =
    rerender.isPending ||
    removeFillers.isPending ||
    removeRepeatedTakes.isPending ||
    removeLongPauses.isPending ||
    rewriteCaptions.isPending ||
    translateCaptions.isPending ||
    applyPrivacy.isPending
  const processingLabel = removeFillers.isPending
    ? 'Recortando muletillas...'
    : removeRepeatedTakes.isPending
      ? 'Quitando repeticiones...'
    : removeLongPauses.isPending
      ? 'Comprimiendo pausas...'
    : applyPrivacy.isPending
      ? 'Aplicando privacidad...'
    : rewriteCaptions.isPending
      ? 'Reescribiendo captions...'
      : translateCaptions.isPending
        ? 'Traduciendo subtitulos...'
        : 'Generando video...'

  return (
    <article className="card overflow-hidden transition-colors hover:border-border-strong">
      {/* Barra indeterminada en el borde superior mientras se procesa */}
      <div className="relative h-[3px] overflow-hidden bg-transparent">
        {isProcessing && (
          <div
            className="absolute top-0 h-full rounded-full bg-cyan"
            style={{ animation: 'indeterminate 1.4s ease-in-out infinite' }}
          />
        )}
      </div>

      <div className="relative aspect-[9/16] bg-black">
        {asset ? (
          <video
            src={asset.video_url}
            poster={asset.thumb_url ?? undefined}
            controls
            preload="metadata"
            className="h-full w-full object-contain"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-ink-mute">sin video</div>
        )}
        {/* Overlay de procesamiento sobre el video */}
        {isProcessing && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-black/75 backdrop-blur-[2px]">
            <Spinner />
            <span className="text-xs font-medium text-white/80">{processingLabel}</span>
          </div>
        )}
        <div className="absolute left-2 top-2 rounded-md bg-black/70 px-2 py-1 text-[11px] font-semibold text-white ring-1 ring-inset ring-white/10">
          Clip {clip.index + 1}
        </div>
        <div title="Score determinista; no es una probabilidad" className="num absolute right-2 top-2 rounded-md bg-cyan/90 px-2 py-1 text-[11px] font-semibold text-black">
          {score}/100
        </div>
      </div>

      <div className="space-y-3 p-4">
        {editing ? (
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <NumberInput label="Inicio" value={draft.start} onChange={(value) => setDraft({ ...draft, start: value })} />
              <NumberInput label="Fin" value={draft.end} onChange={(value) => setDraft({ ...draft, end: value })} />
            </div>
            <input
              className="field w-full px-3 py-2 text-xs"
              value={draft.title}
              onChange={(e) => setDraft({ ...draft, title: e.target.value })}
              placeholder="Titulo"
            />
            <textarea
              className="field min-h-20 w-full px-3 py-2 text-xs"
              value={draft.hook}
              onChange={(e) => setDraft({ ...draft, hook: e.target.value })}
              placeholder="Hook"
            />
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => update.mutate()}
                disabled={update.isPending}
                className="rounded-lg border border-cyan/40 px-3 py-2 text-xs text-cyan disabled:opacity-50"
              >
                Guardar
              </button>
              <button
                type="button"
                onClick={() => setEditing(false)}
                className="rounded-lg border border-border px-3 py-2 text-xs text-ink-soft"
              >
                Cancelar
              </button>
            </div>
          </div>
        ) : (
          <>
            <div>
              <div className="line-clamp-2 text-sm font-semibold text-ink">{clip.title || 'Clip'}</div>
              <div className="num mt-1 text-[11px] text-ink-mute">
                {formatDuration(clip.duration)} /{' '}
                {clip.source_ranges?.length > 1
                  ? `${clip.source_ranges.length} cortes`
                  : `${clip.start.toFixed(0)}s-${clip.end.toFixed(0)}s`}
              </div>
            </div>
            {clip.hook && <p className="line-clamp-2 text-xs leading-5 text-ink-soft">{clip.hook}</p>}
            {clip.detection_source && (
              <div className="text-[11px] font-medium text-ink-mute">
                Deteccion: {detectionSourceLabel(clip.detection_source)}
              </div>
            )}
            {clip.reason && <p className="line-clamp-2 text-[11px] leading-4 text-ink-mute">{clip.reason}</p>}
            {(clip.editorial_cards.length > 0 || clip.privacy_suggestions.length > 0 || clip.editorial_qa.passed != null) && (
              <div className="flex flex-wrap gap-1.5 text-[10px]">
                {clip.editorial_cards.length > 0 && (
                  <span className="rounded bg-cyan/10 px-2 py-1 text-cyan">
                    {clip.editorial_cards.length} tarjeta(s) verificadas
                  </span>
                )}
                {clip.privacy_suggestions.length > 0 && (
                  <span className="rounded bg-warn/10 px-2 py-1 text-warn">
                    {clip.privacy_suggestions.length} dato(s) por revisar
                  </span>
                )}
                {clip.editorial_qa.passed != null && (
                  <span className={`rounded px-2 py-1 ${clip.editorial_qa.passed ? 'bg-good/10 text-good' : 'bg-bad/10 text-bad'}`}>
                    QA editorial {clip.editorial_qa.passed ? 'OK' : 'con avisos'}
                  </span>
                )}
              </div>
            )}
            {(clip.description || clip.hashtags.length > 0) && (
              <div className="rounded-lg border border-border bg-base/40 p-2.5">
                {clip.description && (
                  <p className="line-clamp-3 text-[11px] leading-4 text-ink-soft">{clip.description}</p>
                )}
                {clip.hashtags.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {clip.hashtags.map((t) => (
                      <span key={t} className="rounded bg-cyan/10 px-1.5 py-0.5 text-[10px] text-cyan">
                        {t}
                      </span>
                    ))}
                  </div>
                )}
                <button
                  type="button"
                  onClick={copyCaption}
                  className="mt-2 cursor-pointer text-[11px] text-ink-mute transition-colors hover:text-cyan"
                >
                  {copied ? 'Copiado ✓' : 'Copiar caption + hashtags'}
                </button>
              </div>
            )}
          </>
        )}

        <div className="grid gap-1.5">
          {clip.assets.map((a) => (
            <a
              key={a.id}
              href={a.video_url}
              download
              className="num inline-flex cursor-pointer items-center justify-between gap-2 rounded-lg border border-border px-2.5 py-2 text-[11px] text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan"
            >
              <span className="inline-flex items-center gap-1.5">
                <IconDownload className="h-3.5 w-3.5" />
                {a.format}
              </span>
              <span>{formatBytes(a.size_bytes)}</span>
            </a>
          ))}
        </div>

        <div className="clip-primary-actions">
          {clip.privacy_suggestions.length > 0 && clip.privacy_masks.length === 0 && (
            <button
              type="button"
              onClick={() => applyPrivacy.mutate()}
              disabled={applyPrivacy.isPending}
              title="Confirma y aplica desenfoque únicamente a las zonas sugeridas por OCR"
              className="col-span-2 inline-flex items-center justify-center gap-1.5 rounded-lg border border-warn/40 px-3 py-2 text-xs text-warn transition-colors hover:bg-warn/10 disabled:opacity-50"
            >
              <IconSparkles className="h-3.5 w-3.5" />
              {applyPrivacy.isPending ? 'Aplicando...' : 'Confirmar y desenfocar sugerencias'}
            </button>
          )}
          <a
            href={asset?.video_url}
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan"
          >
            <IconPlay className="h-3.5 w-3.5" />
            Reproducir
          </a>
          <button
            type="button"
            onClick={() => setEditing((value) => !value)}
            title="Editar título y gancho del clip"
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-border-strong hover:text-ink"
          >
            <IconPencil className="h-3.5 w-3.5" />
            Editar texto
          </button>
        </div>
        <details className="clip-tools">
          <summary>Más herramientas <span aria-hidden="true">⌄</span></summary>
          <div className="clip-tool-groups">
            <section className="clip-tool-group" aria-label="Edición del clip">
              <h3>Edición</h3>
              <div className="clip-tool-grid">
          <button
            type="button"
            onClick={() => setStyleOpen((value) => !value)}
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan"
          >
            <IconPencil className="h-3.5 w-3.5" />
            Editar estilo
          </button>
          <button
            type="button"
            onClick={() => setSubtitleOpen((value) => !value)}
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan"
          >
            <IconPencil className="h-3.5 w-3.5" />
            Editar subtitulos
          </button>
              </div>
            </section>
            <section className="clip-tool-group" aria-label="Procesamiento del clip">
              <h3>Procesamiento</h3>
              <div className="clip-tool-grid">
          <button
            type="button"
            onClick={() => rerender.mutate()}
            disabled={rerender.isPending}
            title="Vuelve a renderizar el clip con los ajustes actuales (reframe, subtítulos)"
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan disabled:opacity-50"
          >
            <IconRefresh className="h-3.5 w-3.5" />
            {rerender.isPending ? 'Generando...' : 'Regenerar render'}
          </button>
          <button
            type="button"
            onClick={() => removeFillers.mutate()}
            disabled={removeFillers.isPending}
            title="Recorta automáticamente las muletillas (eh, este, o sea...) y vuelve a generar el clip"
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan disabled:opacity-50"
          >
            <IconScissors className="h-3.5 w-3.5" />
            {removeFillers.isPending ? 'Recortando...' : 'Quitar muletillas'}
          </button>
          <button
            type="button"
            onClick={() => removeRepeatedTakes.mutate()}
            disabled={removeRepeatedTakes.isPending}
            title="Detecta frases dichas dos veces seguidas (tomas repetidas) y conserva la segunda"
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan disabled:opacity-50"
          >
            <IconScissors className="h-3.5 w-3.5" />
            {removeRepeatedTakes.isPending ? 'Quitando...' : 'Quitar repeticiones'}
          </button>
          <button
            type="button"
            onClick={() => removeLongPauses.mutate()}
            disabled={removeLongPauses.isPending}
            title="Comprime las pausas largas entre palabras a 0.3s en vez de quitarlas del todo"
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan disabled:opacity-50"
          >
            <IconScissors className="h-3.5 w-3.5" />
            {removeLongPauses.isPending ? 'Comprimiendo...' : 'Comprimir pausas'}
          </button>
          <button
            type="button"
            onClick={() => rewriteCaptions.mutate()}
            disabled={rewriteCaptions.isPending}
            title="Reescribe los captions para lectura social sin cambiar el audio"
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan disabled:opacity-50"
          >
            <IconSparkles className="h-3.5 w-3.5" />
            {rewriteCaptions.isPending ? 'Reescribiendo...' : 'Rewrite captions'}
          </button>
              </div>
            </section>
            <section className="clip-tool-group" aria-label="Otras herramientas del clip">
              <h3>Más opciones</h3>
              <div className="clip-tool-grid">
          <button
            type="button"
            onClick={() => setTranslateOpen((value) => !value)}
            title="Traduce los subtitulos a otro idioma; el audio original no cambia"
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan"
          >
            <IconSparkles className="h-3.5 w-3.5" />
            Traducir subtitulos
          </button>
          <button
            type="button"
            onClick={() => setTranscriptOpen((value) => !value)}
            title="Edita el clip por su texto: tachá las palabras que querés cortar"
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan"
          >
            <IconPencil className="h-3.5 w-3.5" />
            {transcriptOpen ? 'Cerrar texto' : 'Cortar palabras'}
          </button>
          <button
            type="button"
            onClick={() => setTimelineOpen((value) => !value)}
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan"
          >
            <IconPlay className="h-3.5 w-3.5" />
            Timeline QA
          </button>
          <button
            type="button"
            onClick={() => applyStyle.mutate(true)}
            disabled={applyStyle.isPending}
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan disabled:opacity-50"
          >
            <IconSparkles className="h-3.5 w-3.5" />
            Estilo a todos
          </button>
              </div>
            </section>
          </div>
        </details>
        {styleOpen && (
          <StyleEditor
            value={styleDraft}
            onChange={setStyleDraft}
            onApply={() => applyStyle.mutate(false)}
            onPreset={() => setStyleDraft(DEFAULT_STYLE_CONFIG)}
            saving={applyStyle.isPending}
          />
        )}
        {subtitleOpen && <SubtitleBlockEditor clip={clip} jobId={jobId} onDone={() => setSubtitleOpen(false)} />}
        {translateOpen && (
          <TranslateCaptionsPanel
            pending={translateCaptions.isPending}
            error={translateCaptions.isError ? translateCaptions.error : null}
            onTranslate={(targetLanguage, applyToAll) =>
              translateCaptions.mutate({ targetLanguage, applyToAll })
            }
          />
        )}
        {timelineOpen && <TimelinePanel clip={clip} />}
        {transcriptOpen && (
          <TranscriptEditor clip={clip} jobId={jobId} onDone={() => setTranscriptOpen(false)} />
        )}
      </div>
    </article>
  )
}

function TimelinePanel({ clip }: { clip: Clip }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['timeline', clip.id],
    queryFn: () => api.getClipTimeline(clip.id),
  })

  if (isLoading) return <p className="text-[11px] text-ink-mute">Cargando timeline QA...</p>
  if (error) return <p className="text-[11px] text-warn">Timeline QA no disponible para este clip.</p>
  if (!data) return null

  const firstFormat = data.formats[0]

  return (
    <div className="rounded-lg border border-border bg-base/40 p-3">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <div className="text-xs font-semibold text-ink">Timeline QA</div>
          <div className="num mt-0.5 text-[11px] text-ink-mute">
            {data.clip.source_start.toFixed(1)}s-{data.clip.source_end.toFixed(1)}s / {data.processing.requested_reframe}
          </div>
        </div>
        {firstFormat && (
          <span
            className={`rounded-md px-2 py-1 text-[10px] font-semibold ${
              firstFormat.quality.passed ? 'bg-good/10 text-good' : 'bg-warn/10 text-warn'
            }`}
          >
            QA {firstFormat.quality.passed ? 'OK' : 'Revisar'}
          </span>
        )}
      </div>

      {firstFormat && (
        <>
          <div className="grid grid-cols-4 gap-1.5">
            {firstFormat.frames.map((frame) => (
              <div key={frame.filename} className="overflow-hidden rounded-md border border-border bg-black">
                <img src={frame.url} alt="" className="aspect-[9/16] w-full object-cover" loading="lazy" />
                <div className="num border-t border-white/10 px-1 py-0.5 text-center text-[10px] text-ink-mute">
                  {frame.time.toFixed(1)}s
                </div>
              </div>
            ))}
          </div>

          {firstFormat.audio_energy && firstFormat.audio_energy.length > 0 && (
            <div className="mt-3 rounded-md border border-border bg-black/10 p-2">
              <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-mute">
                Energia de audio
              </div>
              <div className="flex h-12 items-end gap-[2px]">
                {firstFormat.audio_energy.map((sample, index) => (
                  <div
                    key={`${sample.time}-${index}`}
                    title={`${sample.time.toFixed(1)}s`}
                    className="min-w-[3px] flex-1 rounded-t bg-cyan/75"
                    style={{ height: `${Math.max(8, Math.round(sample.rms * 100))}%` }}
                  />
                ))}
              </div>
            </div>
          )}

          <div className="mt-3 grid grid-cols-2 gap-2 text-[11px]">
            <TimelineMetric label="Formato" value={`${firstFormat.format} / ${firstFormat.mode}`} />
            <TimelineMetric label="Intentos QA" value={String(firstFormat.quality.attempts)} />
            <TimelineMetric label="Resolucion" value={`${firstFormat.width}x${firstFormat.height}`} />
            <TimelineMetric label="Audio" value={String(firstFormat.quality.checks.has_audio ?? 'n/a')} />
            {data.editorial && (
              <>
                <TimelineMetric label="Tarjetas" value={String(data.editorial.cards.length)} />
                <TimelineMetric
                  label="Zonas protegidas"
                  value={String(data.editorial.protected_zones.length)}
                />
                <TimelineMetric
                  label="Privacidad"
                  value={`${data.editorial.privacy_masks.length} aplicadas / ${data.editorial.privacy_suggestions.length} sugeridas`}
                />
                <TimelineMetric
                  label="QA editorial"
                  value={data.editorial.qa.passed === false ? 'Revisar' : 'OK'}
                />
              </>
            )}
          </div>

          {firstFormat.delivery_qc && <DeliveryCheck qc={firstFormat.delivery_qc} />}
        </>
      )}

      {data.subtitles.length > 0 && (
        <div className="mt-3 border-t border-border pt-3">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-ink-mute">
            Subtitulos
          </div>
          <div className="max-h-28 space-y-1.5 overflow-y-auto">
            {data.subtitles.slice(0, 5).map((block, index) => (
              <div key={`${block.start}-${index}`} className="rounded border border-border bg-black/10 p-2">
                <div className="num text-[10px] text-ink-mute">
                  {block.start.toFixed(1)}s-{block.end.toFixed(1)}s / {block.word_count} palabras
                </div>
                <div className="mt-1 line-clamp-2 text-[11px] leading-4 text-ink-soft">{block.text}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function TimelineMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-black/10 p-2">
      <div className="text-[10px] text-ink-mute">{label}</div>
      <div className="mt-0.5 break-all font-semibold text-ink">{value}</div>
    </div>
  )
}

const DELIVERY_CHECK_LABELS: Record<string, string> = {
  file: 'Archivo',
  resolution: 'Resolución',
  aspect: 'Aspecto',
  codec: 'Codec',
  pixel_format: 'Formato píxel',
  fps: 'FPS',
  vfr: 'VFR',
  audio: 'Audio',
  audio_codec: 'Codec audio',
  sample_rate: 'Sample rate',
  channels: 'Canales',
  loudness: 'Loudness',
  true_peak: 'True peak',
  duration: 'Duración',
  duration_max: 'Duración máx.',
  color: 'Color',
  faststart: 'Faststart',
  safe_zone: 'Safe zone',
}

function DeliveryCheck({ qc }: { qc: Partial<DeliveryQCResult> }) {
  const checks = Array.isArray(qc.checks) ? qc.checks.filter((check) => check != null) : []
  const status = qc.passed === false || checks.some((check) => check.status === 'fail')
    ? 'FAIL'
    : checks.some((check) => check.status === 'warn')
      ? 'WARN'
      : qc.passed !== true || checks.length === 0 || checks.some((check) => check.status !== 'pass')
        ? 'Desconocido'
        : 'PASS'
  const statusClass = status === 'PASS' ? 'text-good' : status === 'WARN' ? 'text-warn' : status === 'FAIL' ? 'text-bad' : 'text-ink-mute'
  return (
    <div className="mt-3 border-t border-border pt-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[0.12em] text-ink-mute">
          Delivery Check
        </span>
        <span className={`rounded-md bg-black/10 px-2 py-0.5 text-[10px] font-semibold ${statusClass}`}>
          {status} · {qc.profile ?? 'Destino desconocido'}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] sm:grid-cols-3">
        {checks.map((check) => (
          <div
            key={check.id}
            title={check.message || undefined}
            className="flex items-center justify-between gap-2"
          >
            <span className="truncate text-ink-mute">
              {DELIVERY_CHECK_LABELS[check.id] ?? check.id}
            </span>
            <span
              className={
                check.status === 'pass'
                  ? 'font-semibold text-good'
                  : check.status === 'warn'
                    ? 'font-semibold text-warn'
                    : check.status === 'fail'
                      ? 'font-semibold text-bad'
                      : 'font-semibold text-ink-mute'
              }
            >
              {check.status === 'pass' ? 'PASS' : check.status === 'warn' ? 'WARN' : check.status === 'fail' ? 'FAIL' : 'Desconocido'}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

function StyleEditor({
  value,
  onChange,
  onApply,
  onPreset,
  saving,
}: {
  value: SubtitleStyleConfig
  onChange: (value: SubtitleStyleConfig) => void
  onApply: () => void
  onPreset: () => void
  saving: boolean
}) {
  return (
    <div className="rounded-lg border border-border bg-base/40 p-2.5">
      <div className="grid grid-cols-2 gap-2">
        <select value={value.style} onChange={(e) => onChange({ ...value, style: e.target.value })} className="field px-2 py-2 text-xs">
          {['viral_bold', 'clean', 'neon', 'minimal', 'gaming', 'corporate'].map((style) => (
            <option key={style} value={style}>{style.replace('_', ' ')}</option>
          ))}
        </select>
        <select value={value.highlight_color} onChange={(e) => onChange({ ...value, highlight_color: e.target.value })} className="field px-2 py-2 text-xs">
          <option value="#FFD21F">Amarillo</option>
          <option value="#00D4FF">Cian</option>
          <option value="#37E26B">Verde</option>
          <option value="#FF3B30">Rojo</option>
          <option value="#FF4FD8">Rosa</option>
          <option value="#FFFFFF">Blanco</option>
        </select>
        <label className="text-[11px] text-ink-mute">
          <span className="flex items-center justify-between">
            <span>Tamaño</span>
            <span className="num text-cyan">
              {sliderStep(value.size, SUBTITLE_SIZE_RANGE).index}/
              {sliderStep(value.size, SUBTITLE_SIZE_RANGE).total}
            </span>
          </span>
          <input type="range" min={SUBTITLE_SIZE_RANGE.min} max={SUBTITLE_SIZE_RANGE.max} step={SUBTITLE_SIZE_RANGE.step} value={value.size} onChange={(e) => onChange({ ...value, size: Number(e.target.value) })} className="w-full accent-[#3fa9f5]" />
        </label>
        <label className="text-[11px] text-ink-mute">
          <span className="flex items-center justify-between">
            <span>Contorno</span>
            <span className="num text-cyan">
              {sliderStep(value.stroke_width, SUBTITLE_STROKE_RANGE).index}/
              {sliderStep(value.stroke_width, SUBTITLE_STROKE_RANGE).total}
            </span>
          </span>
          <input type="range" min={SUBTITLE_STROKE_RANGE.min} max={SUBTITLE_STROKE_RANGE.max} step={SUBTITLE_STROKE_RANGE.step} value={value.stroke_width} onChange={(e) => onChange({ ...value, stroke_width: Number(e.target.value) })} className="w-full accent-[#3fa9f5]" />
        </label>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-ink-soft">
        <label><input type="checkbox" checked={value.shadow_enabled} onChange={(e) => onChange({ ...value, shadow_enabled: e.target.checked })} /> Sombra</label>
        <label><input type="checkbox" checked={value.background_box} onChange={(e) => onChange({ ...value, background_box: e.target.checked })} /> Caja</label>
        <label><input type="checkbox" checked={value.uppercase} onChange={(e) => onChange({ ...value, uppercase: e.target.checked })} /> Mayusculas</label>
        <label><input type="checkbox" checked={value.emoji_enabled} onChange={(e) => onChange({ ...value, emoji_enabled: e.target.checked })} /> Emojis</label>
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        <button type="button" onClick={onApply} disabled={saving} className="rounded-lg bg-cyan/15 px-3 py-1.5 text-[11px] font-semibold text-cyan disabled:opacity-50">
          {saving ? 'Aplicando...' : 'Aplicar a este clip'}
        </button>
        <button type="button" onClick={onPreset} className="rounded-lg border border-border px-3 py-1.5 text-[11px] text-ink-soft">
          Restaurar preset original
        </button>
        <button type="button" onClick={onApply} disabled={saving} className="rounded-lg border border-border px-3 py-1.5 text-[11px] text-ink-soft">
          Guardar estilo actual como preset
        </button>
      </div>
    </div>
  )
}

function TranslateCaptionsPanel({
  onTranslate,
  pending,
  error,
}: {
  onTranslate: (targetLanguage: string, applyToAll: boolean) => void
  pending: boolean
  error: unknown
}) {
  const [language, setLanguage] = useState(LANGUAGE_OPTIONS[0].value)
  const [customLanguage, setCustomLanguage] = useState('')
  const [applyToAll, setApplyToAll] = useState(false)

  const isCustom = language === '__custom__'
  const resolvedLanguage = (isCustom ? customLanguage : language).trim()
  const canSubmit = resolvedLanguage.length > 0 && !pending

  return (
    <div className="rounded-lg border border-border bg-base/40 p-2.5">
      <div className="grid grid-cols-2 gap-2">
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="field px-2 py-2 text-xs"
        >
          {LANGUAGE_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        {isCustom && (
          <input
            className="field px-2 py-2 text-xs"
            value={customLanguage}
            onChange={(e) => setCustomLanguage(e.target.value)}
            placeholder="Idioma destino"
          />
        )}
      </div>
      <label className="mt-2 flex items-center gap-1.5 text-[11px] text-ink-soft">
        <input type="checkbox" checked={applyToAll} onChange={(e) => setApplyToAll(e.target.checked)} />
        Aplicar a todos los clips
      </label>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => canSubmit && onTranslate(resolvedLanguage, applyToAll)}
          disabled={!canSubmit}
          className="rounded-lg bg-cyan/15 px-3 py-1.5 text-[11px] font-semibold text-cyan disabled:opacity-50"
        >
          {pending ? 'Traduciendo...' : 'Traducir'}
        </button>
        {error != null && (
          <span className="text-[11px] text-bad">
            {error instanceof Error ? error.message : 'No se pudo traducir los subtitulos.'}
          </span>
        )}
      </div>
    </div>
  )
}

function formatBlockTime(seconds: number): string {
  const safe = Math.max(0, seconds)
  const m = Math.floor(safe / 60)
  const s = safe % 60
  return `${m}:${s.toFixed(2).padStart(5, '0')}`
}

/**
 * Timeline QA (GET /clips/{id}/timeline) ya trae, por bloque de subtitulo, sus
 * palabras con timestamps (backend/app/pipeline/render.py::_write_timeline_artifact).
 * Reusamos esa misma fuente para editar en vez de re-derivar bloques arbitrarios
 * desde el transcript plano.
 */
function SubtitleBlockEditor({ clip, jobId, onDone }: { clip: Clip; jobId: string; onDone: () => void }) {
  const qc = useQueryClient()
  const { data, isLoading, error } = useQuery({
    queryKey: ['timeline', clip.id],
    queryFn: () => api.getClipTimeline(clip.id),
  })

  const initialBlocks: SubtitleBlock[] = useMemo(
    () =>
      (data?.subtitles ?? []).map((block) => ({
        start: block.start,
        end: block.end,
        text: block.text,
        words:
          block.words && block.words.length > 0
            ? block.words
            : [{ word: block.text, start: block.start, end: block.end }],
      })),
    [data],
  )
  // blocks = lo editado por el usuario (split/merge/borrar/texto), o los iniciales
  // derivados del timeline (sin setState en effect: se recalcula si cambia data).
  const [edited, setEdited] = useState<SubtitleBlock[] | null>(null)
  const [applyToAll, setApplyToAll] = useState(false)
  const blocks = edited ?? initialBlocks

  const save = useMutation({
    mutationFn: () => api.editClipSubtitles(clip.id, { subtitles: blocks, apply_to_all: applyToAll }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['clips', jobId] })
      qc.invalidateQueries({ queryKey: ['logs', jobId] })
      qc.invalidateQueries({ queryKey: ['timeline', clip.id] })
      onDone()
    },
  })

  if (isLoading) return <p className="text-[11px] text-ink-mute">Cargando subtitulos...</p>
  if (error) return <p className="text-[11px] text-warn">No se pudo cargar el timeline de este clip para editar subtitulos.</p>
  if (initialBlocks.length === 0) {
    return <p className="text-[11px] text-ink-mute">Este clip no tiene bloques de subtitulos para editar.</p>
  }

  // Edicion de texto libre: ya no hay mapeo palabra-por-palabra confiable, se
  // colapsa a una unica palabra sintetica que cubre todo el rango del bloque
  // (misma degradacion que usa el backend en subtitle_blocks_to_segments/_fallback_lines).
  const updateText = (i: number, text: string) =>
    setEdited(
      blocks.map((block, index) =>
        index === i
          ? { ...block, text, words: [{ word: text, start: block.start, end: block.end }] }
          : block,
      ),
    )

  const updateTime = (i: number, patch: Partial<Pick<SubtitleBlock, 'start' | 'end'>>) =>
    setEdited(
      blocks.map((block, index) => {
        if (index !== i) return block
        const next = { ...block, ...patch }
        const words =
          block.words && block.words.length === 1
            ? [{ ...block.words[0], start: next.start, end: next.end }]
            : block.words
        return { ...next, words }
      }),
    )

  const splitAt = (i: number, wordIndex: number) => {
    const block = blocks[i]
    const words = block.words ?? []
    if (words.length < 2 || wordIndex <= 0 || wordIndex >= words.length) return
    const left = words.slice(0, wordIndex)
    const right = words.slice(wordIndex)
    const a: SubtitleBlock = {
      start: left[0].start,
      end: left[left.length - 1].end,
      text: left.map((w) => w.word).join(' '),
      words: left,
    }
    const b: SubtitleBlock = {
      start: right[0].start,
      end: right[right.length - 1].end,
      text: right.map((w) => w.word).join(' '),
      words: right,
    }
    setEdited([...blocks.slice(0, i), a, b, ...blocks.slice(i + 1)])
  }

  const mergeWithNext = (i: number) => {
    if (i >= blocks.length - 1) return
    const a = blocks[i]
    const b = blocks[i + 1]
    const merged: SubtitleBlock = {
      start: a.start,
      end: b.end,
      text: [a.text, b.text].filter(Boolean).join(' '),
      words: [...(a.words ?? []), ...(b.words ?? [])],
    }
    setEdited([...blocks.slice(0, i), merged, ...blocks.slice(i + 2)])
  }

  const deleteBlock = (i: number) => setEdited(blocks.filter((_, index) => index !== i))

  return (
    <div className="rounded-lg border border-border bg-base/40 p-2.5">
      <div className="max-h-64 space-y-2 overflow-y-auto">
        {blocks.map((block, i) => (
          <div key={i} className="space-y-1.5 rounded border border-border p-2">
            <textarea
              value={block.text}
              onChange={(e) => updateText(i, e.target.value)}
              className="field min-h-14 w-full px-2 py-1.5 text-xs"
            />

            {(block.words?.length ?? 0) > 1 && (
              <div className="flex flex-wrap items-center gap-1">
                <span className="text-[10px] text-ink-mute">Dividir despues de:</span>
                {block.words!.slice(0, -1).map((w, wi) => (
                  <button
                    key={wi}
                    type="button"
                    onClick={() => splitAt(i, wi + 1)}
                    title={`Dividir el bloque despues de "${w.word}"`}
                    className="cursor-pointer rounded bg-white/5 px-1.5 py-0.5 text-[11px] text-ink-soft transition-colors hover:bg-cyan/10 hover:text-cyan"
                  >
                    {w.word} |
                  </button>
                ))}
              </div>
            )}

            <div className="grid grid-cols-2 gap-1.5">
              <label>
                <span className="mb-1 block text-[10px] text-ink-mute">Inicio ({formatBlockTime(block.start)})</span>
                <input
                  type="number"
                  step="0.1"
                  value={block.start}
                  onChange={(e) => updateTime(i, { start: Number(e.target.value) })}
                  className="field w-full px-2 py-1.5 text-xs"
                />
              </label>
              <label>
                <span className="mb-1 block text-[10px] text-ink-mute">Fin ({formatBlockTime(block.end)})</span>
                <input
                  type="number"
                  step="0.1"
                  value={block.end}
                  onChange={(e) => updateTime(i, { end: Number(e.target.value) })}
                  className="field w-full px-2 py-1.5 text-xs"
                />
              </label>
            </div>

            <div className="flex flex-wrap gap-1.5">
              <button
                type="button"
                onClick={() => mergeWithNext(i)}
                disabled={i >= blocks.length - 1}
                className="rounded border border-border px-2 py-1 text-[10px] text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan disabled:opacity-40"
              >
                Fusionar con siguiente
              </button>
              <button
                type="button"
                onClick={() => deleteBlock(i)}
                className="rounded border border-border px-2 py-1 text-[10px] text-bad/80 transition-colors hover:border-bad/50 hover:text-bad"
              >
                Borrar linea
              </button>
            </div>
          </div>
        ))}
      </div>

      <label className="mt-2 flex items-center gap-2 text-[11px] text-ink-soft">
        <input type="checkbox" checked={applyToAll} onChange={(e) => setApplyToAll(e.target.checked)} />
        Aplicar a todos los clips
      </label>

      <div className="mt-2 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => save.mutate()}
          disabled={save.isPending || blocks.length === 0}
          className="rounded-lg bg-cyan/15 px-3 py-1.5 text-[11px] font-semibold text-cyan disabled:opacity-50"
        >
          {save.isPending ? 'Guardando...' : 'Guardar cambios'}
        </button>
        {save.isError && <span className="text-[11px] text-bad">No se pudieron guardar los subtitulos.</span>}
      </div>
    </div>
  )
}

function TranscriptEditor({ clip, jobId, onDone }: { clip: Clip; jobId: string; onDone: () => void }) {
  const qc = useQueryClient()
  const [removed, setRemoved] = useState<Set<number>>(new Set())
  const { data, isLoading } = useQuery({
    queryKey: ['transcript', clip.id],
    queryFn: () => api.getClipTranscript(clip.id),
  })

  const apply = useMutation({
    mutationFn: () => {
      const words = data?.words ?? []
      const remove = [...removed].map((i) => [words[i].start, words[i].end])
      return api.reeditClip(clip.id, { remove })
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['clips', jobId] })
      qc.invalidateQueries({ queryKey: ['logs', jobId] })
      onDone()
    },
  })

  if (isLoading) return <p className="text-[11px] text-ink-mute">Cargando texto…</p>
  if (!data?.has_words)
    return (
      <p className="text-[11px] text-ink-mute">
        Este clip no tiene transcripción palabra por palabra para editar.
      </p>
    )

  const toggle = (i: number) =>
    setRemoved((prev) => {
      const next = new Set(prev)
      if (next.has(i)) next.delete(i)
      else next.add(i)
      return next
    })

  return (
    <div className="rounded-lg border border-border bg-base/40 p-2.5">
      <p className="mb-2 text-[11px] text-ink-mute">Tocá las palabras que querés cortar:</p>
      <div className="flex max-h-40 flex-wrap gap-1 overflow-y-auto">
        {data.words.map((w, i) => (
          <button
            key={i}
            type="button"
            onClick={() => toggle(i)}
            className={`cursor-pointer rounded px-1.5 py-0.5 text-[11px] transition-colors ${
              removed.has(i)
                ? 'bg-bad/20 text-bad line-through'
                : 'bg-white/5 text-ink-soft hover:bg-cyan/10 hover:text-cyan'
            }`}
          >
            {w.word}
          </button>
        ))}
      </div>
      <div className="mt-2 flex items-center justify-between gap-2">
        <span className="text-[11px] text-ink-mute">{removed.size} para cortar</span>
        <button
          type="button"
          onClick={() => apply.mutate()}
          disabled={apply.isPending || removed.size === 0}
          className="rounded-lg bg-cyan/15 px-3 py-1.5 text-[11px] font-semibold text-cyan transition-colors hover:bg-cyan/25 disabled:opacity-50"
        >
          {apply.isPending ? 'Recortando…' : 'Aplicar y regenerar'}
        </button>
      </div>
    </div>
  )
}

function NumberInput({
  label,
  value,
  onChange,
}: {
  label: string
  value: number
  onChange: (value: number) => void
}) {
  return (
    <label>
      <span className="mb-1 block text-[10px] text-ink-mute">{label}</span>
      <input
        type="number"
        step="0.1"
        min="0"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="field w-full px-2 py-2 text-xs"
      />
    </label>
  )
}
