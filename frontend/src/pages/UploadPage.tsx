import { useQuery } from '@tanstack/react-query'
import { type ReactNode, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { useNavigate } from 'react-router-dom'
import { IconDownload, IconLink, IconSparkles, IconUpload } from '../components/icons'
import { Select } from '../components/Select'
import { ConfigCard } from '../components/ConfigCard'
import { ToggleField } from '../components/ToggleSwitch'
import { TopbarActions, TopbarTitle } from '../components/topbar'
import { Spinner } from '../components/ui'
import { api } from '../lib/api'
import {
  GAMING_OUTPUT_OPTIONS,
  isInternalGamingMode,
  resolveGamingProcessingMode,
  type GamingOutput,
} from '../lib/gaming'
import {
  SUBTITLE_SIZE_RANGE,
  SUBTITLE_STROKE_RANGE,
  providerLabel,
  sliderStep,
} from '../lib/format'
import type { JobOutputMode, MediaOptions, ProcessingMode, ProcessingModeOption, SubtitleStyleConfig } from '../lib/types'

// w/h en px: la forma se inscribe en un bounding box de 40x40 conservando la proporcion real,
// asi las tres cards lucen parejas y comunican el aspect ratio de un vistazo.
const FORMAT_LABELS: Record<string, { title: string; sub: string; w: number; h: number }> = {
  '9x16': { title: '9:16', sub: 'TikTok · Reels · Shorts', w: 23, h: 40 },
  '1x1': { title: '1:1', sub: 'Feed', w: 40, h: 40 },
  '16x9': { title: '16:9', sub: 'YouTube', w: 40, h: 23 },
}

const FALLBACK_PROCESSING_MODES: ProcessingModeOption[] = [
  { value: 'auto_smart', label: 'Auto', description: 'El LLM detecta el tipo de contenido del transcript y elige el enfoque más adecuado.' },
  { value: 'podcast_viral', label: 'Podcast / Entrevista', description: '2 o más personas, debate, entrevista, mesa redonda. Prioriza ideas virales y el hablante activo.' },
  { value: 'vlog', label: 'Vlog / Directo a cámara', description: 'Una persona hablando, estilo personal y cotidiano. Busca anécdotas, humor y momentos de conexión.' },
  { value: 'gaming', label: 'Gaming', description: 'Gameplay, kills, reacciones en tiempo real. Prioriza acción, tensión y victorias o derrotas.' },
  { value: 'tutorial_screen', label: 'Tutorial / Educativo', description: 'Instrucciones paso a paso o contenido con pantalla. Protege texto y UI visible.' },
  { value: 'music', label: 'Música / Ambiental', description: 'Sin diálogo principal o con música como protagonista. Corta por escena o energía, sin subtítulos.' },
  { value: 'keep_original', label: 'Técnico: conservar encuadre', description: 'Respeta el cuadro completo siempre. Útil para contenido con texto en bordes o composición fija.' },
]

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

const DOWNLOAD_QUALITY_OPTIONS = [
  { value: 'auto', label: 'Auto - mejor disponible' },
  { value: '720', label: 'Hasta 720p' },
  { value: '1080', label: 'Hasta 1080p' },
  { value: '1440', label: 'Hasta 1440p (2K)' },
  { value: '2160', label: 'Hasta 2160p (4K)' },
]

const AUDIO_FORMAT_OPTIONS = [
  { value: 'mp3', label: 'MP3' },
  { value: 'wav', label: 'WAV (sin compresion)' },
  { value: 'm4a', label: 'M4A' },
  { value: 'opus', label: 'Opus' },
  { value: 'flac', label: 'FLAC (sin perdida)' },
]

const TRANSCRIPTION_QUALITY_OPTIONS = [
  { value: 'default', label: 'Transcripción · Predeterminada' },
  { value: 'fast', label: 'Transcripción · Rápida' },
  { value: 'balanced', label: 'Transcripción · Equilibrada' },
  { value: 'precise', label: 'Transcripción · Precisa' },
]

const COLOR_HANDLING_OPTIONS = [
  { value: 'auto', label: 'Color HDR · Auto' },
  { value: 'sdr', label: 'Color HDR · Convertir a SDR' },
  { value: 'preserve', label: 'Color HDR · Intentar preservar' },
]

const STABILIZATION_OPTIONS = [
  { value: 'off', label: 'Estabilización · Desactivada' },
  { value: 'auto', label: 'Estabilización · Auto' },
  { value: 'light', label: 'Estabilización · Suave' },
  { value: 'medium', label: 'Estabilización · Media' },
  { value: 'strong', label: 'Estabilización · Fuerte' },
]

const VIDEO_DENOISE_OPTIONS = [
  { value: 'off', label: 'Ruido de video · Desactivado' },
  { value: 'light', label: 'Ruido de video · Suave' },
  { value: 'medium', label: 'Ruido de video · Medio' },
  { value: 'strong', label: 'Ruido de video · Fuerte' },
]

const AUDIO_CLEANUP_OPTIONS = [
  { value: 'off', label: 'Limpieza de audio · Desactivada' },
  { value: 'voice_light', label: 'Limpieza de audio · Voz suave' },
  { value: 'voice', label: 'Limpieza de audio · Voz' },
  { value: 'voice_strong', label: 'Limpieza de audio · Voz fuerte' },
]

export function UploadPage() {
  const navigate = useNavigate()
  const { data: settings } = useQuery({ queryKey: ['settings'], queryFn: api.getSettings })

  const [file, setFile] = useState<File | null>(null)
  const [url, setUrl] = useState('')
  const [batchMode, setBatchMode] = useState(false)
  const [batchUrls, setBatchUrls] = useState('')
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | null>(null)
  const [batchResult, setBatchResult] = useState<{ okCount: number; failed: Array<{ url: string; reason: string }> } | null>(null)
  const [numClips, setNumClips] = useState(5)
  const [formats, setFormats] = useState<string[]>(['9x16'])
  const [outputMode, setOutputMode] = useState<JobOutputMode>('clips')
  const [captionsEnabled, setCaptionsEnabled] = useState(true)
  const [sponsorblockEnabled, setSponsorblockEnabled] = useState(true)
  const [downloadQuality, setDownloadQuality] = useState('auto')
  const [audioFormat, setAudioFormat] = useState('mp3')
  const [processingMode, setProcessingMode] = useState<ProcessingMode>('auto_smart')
  const [gamingOutput, setGamingOutput] = useState<GamingOutput>('auto')
  const [captionStyle, setCaptionStyle] = useState('viral_bold')
  const [styleConfig, setStyleConfig] = useState<SubtitleStyleConfig>(DEFAULT_STYLE_CONFIG)
  const [provider, setProvider] = useState('')
  const [compactEditing, setCompactEditing] = useState(false)
  const [semanticCards, setSemanticCards] = useState(false)
  const [privacyAssist, setPrivacyAssist] = useState(false)
  const [protectedZones, setProtectedZones] = useState(true)
  const [overlayQa, setOverlayQa] = useState(true)
  const [transcriptionQuality, setTranscriptionQuality] = useState<'default' | 'fast' | 'balanced' | 'precise'>('default')
  const [mediaOverrides, setMediaOverrides] = useState<Partial<MediaOptions>>({})
  const mediaOptions: MediaOptions = {
    destination_profile: 'auto',
    color_handling: 'auto',
    stabilization: 'off',
    video_denoise: 'off',
    audio_cleanup: 'off',
    ...settings?.media_defaults,
    ...mediaOverrides,
  }

  function updateMedia<K extends keyof MediaOptions>(key: K, value: MediaOptions[K]) {
    setMediaOverrides((previous) => ({ ...previous, [key]: value }))
  }
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'video/*': [] },
    multiple: false,
    onDrop: (files) => {
      if (files[0]) {
        setFile(files[0])
        setUrl('')
      }
    },
  })

  const availableFormats = settings?.available_formats ?? ['9x16', '1x1', '16x9']
  const processingModes = (settings?.processing_modes ?? FALLBACK_PROCESSING_MODES).filter(
    (mode) => !isInternalGamingMode(mode.value),
  )
  const captionStyles = settings?.caption_styles ?? []
  const highlightColors = settings?.highlight_colors ?? {
    yellow: '#FFD21F',
    cyan: '#00D4FF',
    green: '#37E26B',
    red: '#FF3B30',
    pink: '#FF4FD8',
    white: '#FFFFFF',
  }
  const effectiveStyleConfig = { ...(settings?.default_style_config ?? DEFAULT_STYLE_CONFIG), ...styleConfig, style: captionStyle }
  const activeCaptionStyle =
    captionStyles.find((style) => style.value === (captionStyle || settings?.caption_style)) ??
    captionStyles[0]
  const activeProcessingMode =
    processingModes.find((mode) => mode.value === processingMode) ?? processingModes[0]
  const providers = settings?.providers ?? ['naive']
  const downloadOnly = outputMode === 'download'
  const audioOnly = outputMode === 'audio'
  const isClipsMode = outputMode === 'clips'
  const isGaming = processingMode === 'gaming'
  const isSingleGamingEdit = isGaming && (gamingOutput === 'hook' || gamingOutput === 'highlights')
  const effectiveNumClips = isSingleGamingEdit ? 1 : numClips
  const effectiveProcessingMode = resolveGamingProcessingMode(processingMode, gamingOutput)
  const activeGamingOutput = GAMING_OUTPUT_OPTIONS.find((option) => option.value === gamingOutput)

  const toggleFormat = (f: string) =>
    setFormats((prev) => (prev.includes(f) ? prev.filter((x) => x !== f) : [...prev, f]))

  async function submitBatch() {
    const lines = batchUrls
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
    if (lines.length === 0) return setError('Pega al menos una URL, una por linea.')
    if (formats.length === 0) return setError('Elige al menos un formato.')

    const validUrls: string[] = []
    const failed: Array<{ url: string; reason: string }> = []
    for (const line of lines) {
      if (/^https?:\/\//i.test(line)) validUrls.push(line)
      else failed.push({ url: line, reason: 'No parece una URL valida (debe empezar con http:// o https://)' })
    }
    if (validUrls.length === 0) {
      return setError('Ninguna linea parece una URL valida.')
    }

    const opts = {
      num_clips: effectiveNumClips,
      formats,
      processing_mode: effectiveProcessingMode,
      caption_style: captionStyle || settings?.caption_style || 'viral_bold',
      style_config: effectiveStyleConfig,
      provider: provider || undefined,
      output_mode: outputMode,
      captions_enabled: captionsEnabled,
      download_quality: downloadQuality,
      audio_format: audioFormat,
      sponsorblock_enabled: sponsorblockEnabled,
      editorial_options: {
        compact_editing: compactEditing,
        semantic_cards: semanticCards,
        privacy_assist: privacyAssist,
        protected_zones: protectedZones,
        overlay_qa: overlayQa,
        transcription_quality: transcriptionQuality,
      },
      media_options: mediaOptions,
    }

    setSubmitting(true)
    let okCount = 0
    for (let i = 0; i < validUrls.length; i++) {
      setBatchProgress({ current: i + 1, total: validUrls.length })
      try {
        await api.createFromUrl(validUrls[i], opts)
        okCount += 1
      } catch (e) {
        failed.push({ url: validUrls[i], reason: String(e instanceof Error ? e.message : e) })
      }
    }
    setBatchProgress(null)
    setBatchResult({ okCount, failed })
    setSubmitting(false)
  }

  async function submit() {
    setError(null)
    setBatchResult(null)
    if (isClipsMode && batchMode) return submitBatch()
    if (!isClipsMode && !url.trim()) {
      return setError(
        audioOnly ? 'Pega una URL publica para extraer el audio.' : 'Pega una URL publica para descargar el video.',
      )
    }
    if (isClipsMode && !file && !url.trim()) return setError('Sube un archivo o pega una URL.')
    if (isClipsMode && formats.length === 0) return setError('Elige al menos un formato.')
    setSubmitting(true)
    try {
      const opts = {
        num_clips: effectiveNumClips,
        formats,
        processing_mode: effectiveProcessingMode,
        caption_style: captionStyle || settings?.caption_style || 'viral_bold',
        style_config: effectiveStyleConfig,
        provider: provider || undefined,
        output_mode: outputMode,
        captions_enabled: captionsEnabled,
        download_quality: downloadQuality,
        audio_format: audioFormat,
        sponsorblock_enabled: sponsorblockEnabled,
        editorial_options: {
          compact_editing: compactEditing,
          semantic_cards: semanticCards,
          privacy_assist: privacyAssist,
          protected_zones: protectedZones,
          overlay_qa: overlayQa,
          transcription_quality: transcriptionQuality,
        },
        media_options: mediaOptions,
      }
      const res = isClipsMode && file
        ? await api.createFromFile(file, opts)
        : await api.createFromUrl(url.trim(), opts)
      navigate(`/jobs/${res.id}`)
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
      setSubmitting(false)
    }
  }

  return (
    <div className="page-container page-container--form section-stack upload-page">
      <TopbarTitle>
        <span className="topbar-context">Nuevo trabajo</span>
        <strong>{audioOnly ? 'Extraer audio' : downloadOnly ? 'Descargar video' : 'Crear clips'}</strong>
      </TopbarTitle>
      <TopbarActions>
        <button
          type="button"
          onClick={submit}
          disabled={submitting}
          className="btn-primary disabled:cursor-not-allowed"
        >
          {submitting ? <Spinner /> : downloadOnly || audioOnly ? <IconDownload className="h-[18px] w-[18px]" /> : <IconSparkles className="h-[18px] w-[18px]" />}
          {submitting
            ? batchProgress
              ? `Creando job ${batchProgress.current}/${batchProgress.total}...`
              : 'Encolando...'
            : audioOnly
              ? 'Extraer audio'
              : downloadOnly
                ? 'Descargar video'
                : batchMode
                  ? 'Crear jobs'
                  : 'Generar clips'}
        </button>
      </TopbarActions>
      <header className="page-header">
        <span className="page-eyebrow">Nuevo trabajo</span>
        <h1>{audioOnly ? 'Extraer audio' : downloadOnly ? 'Descargar video' : 'Crear clips'}</h1>
        <p className="page-description">{audioOnly
          ? 'Extrae solo el audio de una URL publica compatible, en el formato que elijas.'
          : downloadOnly
            ? 'Guarda el video original desde una URL publica compatible, sin IA ni reencode.'
            : 'Pega una URL o sube un video. Snipra lo descarga, transcribe y genera clips por formato.'}</p>
      </header>

      {error && (
        <div role="alert" className="page-error">{error}</div>
      )}

      <section className="form-grid form-grid--3 operation-grid" aria-label="Tipo de trabajo">
        <h2 className="section-heading operation-heading">Tipo de operación</h2>
        <button
          type="button"
          aria-pressed={isClipsMode}
          onClick={() => setOutputMode('clips')}
          className={`flex min-h-20 flex-col items-start justify-center rounded-lg border px-4 text-left transition-colors ${
            isClipsMode ? 'border-cyan/60 bg-cyan/[0.07] text-ink' : 'border-border bg-base/40 text-ink-soft hover:border-border-strong'
          }`}
        >
          <span className="text-sm font-semibold">Crear clips</span>
          <span className="mt-1 text-xs leading-5 text-ink-mute">Detecta momentos y genera formatos sociales.</span>
        </button>
        <button
          type="button"
          aria-pressed={downloadOnly}
          onClick={() => {
            setOutputMode('download')
            setFile(null)
            setBatchMode(false)
            setBatchResult(null)
          }}
          className={`flex min-h-20 flex-col items-start justify-center rounded-lg border px-4 text-left transition-colors ${
            downloadOnly ? 'border-cyan/60 bg-cyan/[0.07] text-ink' : 'border-border bg-base/40 text-ink-soft hover:border-border-strong'
          }`}
        >
          <span className="inline-flex items-center gap-2 text-sm font-semibold"><IconDownload className="h-4 w-4" /> Descargar original</span>
          <span className="mt-1 text-xs leading-5 text-ink-mute">Conserva el video descargado, sin procesarlo.</span>
        </button>
        <button
          type="button"
          aria-pressed={audioOnly}
          onClick={() => {
            setOutputMode('audio')
            setFile(null)
            setBatchMode(false)
            setBatchResult(null)
          }}
          className={`flex min-h-20 flex-col items-start justify-center rounded-lg border px-4 text-left transition-colors ${
            audioOnly ? 'border-cyan/60 bg-cyan/[0.07] text-ink' : 'border-border bg-base/40 text-ink-soft hover:border-border-strong'
          }`}
        >
          <span className="text-sm font-semibold">Extraer audio</span>
          <span className="mt-1 text-xs leading-5 text-ink-mute">Solo el audio: MP3, WAV y mas.</span>
        </button>
      </section>

      <section>
        <div className="section-card">
          <div className="mb-4 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-semibold text-ink">Fuente</div>
              <div className="mt-1 text-xs text-ink-mute">{isClipsMode ? 'URL publica o archivo local' : 'URL publica compatible'}</div>
            </div>
            <div className="flex items-center gap-3">
              {isClipsMode && (
                <button
                  type="button"
                  onClick={() => {
                    setBatchMode((value) => !value)
                    setBatchResult(null)
                    setError(null)
                    if (!batchMode) setFile(null)
                  }}
                  className="text-xs text-ink-mute underline-offset-2 hover:text-cyan hover:underline"
                >
                  {batchMode ? 'Volver a una URL' : 'Pegar varias URLs'}
                </button>
              )}
              {file && (
                <button
                  type="button"
                  onClick={() => setFile(null)}
                  className="rounded-lg border border-border px-3 py-1.5 text-xs text-ink-soft hover:border-border-strong hover:text-ink"
                >
                  Quitar archivo
                </button>
              )}
            </div>
          </div>

          {batchMode && isClipsMode ? (
            <div className="relative">
              <textarea
                value={batchUrls}
                onChange={(e) => setBatchUrls(e.target.value)}
                placeholder={'https://youtube.com/watch?v=...\nhttps://tiktok.com/@user/video/...'}
                rows={6}
                className="field w-full px-4 py-3 text-sm"
              />
              <p className="mt-2 text-[11px] text-ink-mute">
                Una URL por linea. Se crea un job por cada una, en orden (no en paralelo).
              </p>
            </div>
          ) : (
            <>
              <div className={`grid gap-3 ${isClipsMode ? 'lg:grid-cols-[1fr_220px]' : ''}`}>
                <div className="relative">
                  <IconLink className="pointer-events-none absolute left-3.5 top-1/2 h-[18px] w-[18px] -translate-y-1/2 text-ink-mute" />
                  <input
                    type="url"
                    value={url}
                    onChange={(e) => {
                      setUrl(e.target.value)
                      if (e.target.value) setFile(null)
                    }}
                    placeholder="https://youtube.com/watch?v=..."
                    className="field h-12 w-full pl-11 pr-4 text-sm"
                  />
                </div>

                {isClipsMode && <div
                  {...getRootProps()}
                  className={`btn-secondary flex h-12 cursor-pointer items-center justify-center gap-2 px-3 ${
                    isDragActive ? 'ring-2 ring-cyan ring-offset-2 ring-offset-surface' : ''
                  }`}
                >
                  <input {...getInputProps()} />
                  <IconUpload className="h-[18px] w-[18px]" />
                  <span className="truncate">{file ? file.name : 'Subir video'}</span>
                </div>}
              </div>
              <p className="mt-2 text-[11px] text-ink-mute">
                Soporta YouTube, TikTok, Instagram, Twitch, Facebook y 1800+ sitios más vía yt-dlp.
              </p>
            </>
          )}
        </div>

      </section>

      {isClipsMode && <>
      <section className="form-grid upload-settings-grid" aria-label="Configuración principal">
        <div className="section-card flex min-w-0 flex-col">
          <div className="mb-3 text-sm font-semibold text-ink">Formatos de salida</div>
          <div className="grid grid-cols-3 gap-2">
            {availableFormats.map((f) => {
              const on = formats.includes(f)
              const meta = FORMAT_LABELS[f] ?? { title: f, sub: '', w: 32, h: 32 }
              return (
                <button
                  key={f}
                  type="button"
                  onClick={() => toggleFormat(f)}
                  aria-pressed={on}
                  className={`flex cursor-pointer flex-col items-center gap-2 rounded-lg border p-3 text-center transition-colors ${
                    on
                      ? 'border-cyan/60 bg-cyan/[0.07] text-ink'
                      : 'border-border bg-base/40 text-ink-soft hover:border-border-strong'
                  }`}
                >
                  <span className="flex h-10 w-10 items-center justify-center">
                    <span
                      style={{ width: meta.w, height: meta.h }}
                      className={`rounded-[3px] border-2 transition-colors ${
                        on ? 'border-cyan bg-cyan/20' : 'border-ink-mute/60'
                      }`}
                    />
                  </span>
                  <span className="num text-base font-semibold leading-none text-ink">{meta.title}</span>
                  <span className={`text-[11px] leading-tight ${on ? 'text-ink-soft' : 'text-ink-mute'}`}>
                    {meta.sub}
                  </span>
                </button>
              )
            })}
          </div>
        </div>

        <div className="section-card min-w-0 space-y-4">
          <div>
            <div className="mb-2 flex items-center justify-between text-sm font-semibold text-ink">
              <span>
                {isSingleGamingEdit
                  ? 'Edit final'
                  : isGaming && gamingOutput === 'auto'
                    ? 'Maximo de clips'
                    : 'Numero de clips'}
              </span>
              <span className="num text-cyan">{effectiveNumClips}</span>
            </div>
            <input
              type="range"
              min={1}
              max={12}
              value={effectiveNumClips}
              disabled={isSingleGamingEdit}
              onChange={(e) => setNumClips(Number(e.target.value))}
              className="w-full cursor-pointer"
            />
          </div>
          <div>
            <div className="mb-2 text-sm font-semibold text-ink">IA de deteccion</div>
            <Select
              ariaLabel="IA de deteccion"
              value={provider}
              onChange={setProvider}
              options={[
                { value: '', label: `Por defecto · ${providerLabel(settings?.provider ?? 'naive')}` },
                ...providers.map((p) => ({ value: p, label: providerLabel(p) })),
              ]}
            />
          </div>
        </div>

        <ConfigCard title="Modo de procesamiento" description={activeProcessingMode?.description} extra={isGaming && (
            <div>
              <div className="mb-2 text-xs text-ink-mute">Tipo de salida gaming</div>
              <Select
                ariaLabel="Tipo de salida gaming"
                value={gamingOutput}
                onChange={(value) => setGamingOutput(value as GamingOutput)}
                options={GAMING_OUTPUT_OPTIONS.map((option) => ({ value: option.value, label: option.label }))}
              />
              <p className="mt-2 text-xs leading-5 text-cyan">{activeGamingOutput?.description}</p>
            </div>
          )}>
            <Select
              ariaLabel="Modo de procesamiento"
              value={processingMode}
              onChange={(v) => setProcessingMode(v as ProcessingMode)}
              options={processingModes.map((mode) => ({ value: mode.value, label: mode.label }))}
            />
        </ConfigCard>

        <ConfigCard title="Calidad de descarga" description="Resolución del video fuente antes de recortar a 9:16. Auto = la mejor que tenga el video. Más resolución aquí = clips más nítidos.">
          <Select
            ariaLabel="Calidad de descarga"
            value={downloadQuality}
            onChange={setDownloadQuality}
            options={DOWNLOAD_QUALITY_OPTIONS}
          />
        </ConfigCard>

      </section>

      <section className="section-card">
        <div className="mb-1 text-sm font-semibold text-ink">Edición inteligente nativa</div>
        <p className="mb-4 text-xs leading-5 text-ink-soft">
          Controles locales del montaje. Las sugerencias de privacidad nunca se aplican sin tu confirmación.
        </p>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <ToggleField
            label="Edición compacta (Podcast/Vlog)"
            checked={compactEditing}
            onChange={setCompactEditing}
          />
          <ToggleField label="Tarjetas semánticas verificadas" checked={semanticCards} onChange={setSemanticCards} />
          <ToggleField label="Sugerir datos privados por OCR" checked={privacyAssist} onChange={setPrivacyAssist} />
          <ToggleField label="Proteger sujeto y subtítulos" checked={protectedZones} onChange={setProtectedZones} />
          <ToggleField label="QA de overlays y límites" checked={overlayQa} onChange={setOverlayQa} />
          <Select
            ariaLabel="Calidad de transcripción"
            value={transcriptionQuality}
            onChange={(value) => setTranscriptionQuality(value as typeof transcriptionQuality)}
            options={TRANSCRIPTION_QUALITY_OPTIONS}
          />
        </div>
      </section>

      <details className="section-card media-enhancements">
        <summary className="cursor-pointer select-none text-sm font-semibold text-ink">
          Mejoras de medios <span aria-hidden="true" className="text-ink-soft">⌄</span>
        </summary>
        <p className="mt-2 text-xs leading-5 text-ink-soft">
          Destino y mejoras opcionales para este trabajo.
        </p>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <Field label="Destino">
            <Select
              ariaLabel="Destino de publicación"
              value={mediaOptions.destination_profile}
              onChange={(v) => updateMedia('destination_profile', v)}
              options={[
                { value: 'auto', label: 'Auto (según formato)' },
                ...(settings?.destination_profiles ?? []).map((profile) => ({
                  value: profile.id,
                  label: `${profile.display_name} (${profile.frame})`,
                })),
              ]}
            />
          </Field>
          <Field label="Color">
            <Select
              ariaLabel="Manejo de color HDR"
              value={mediaOptions.color_handling}
              onChange={(v) => updateMedia('color_handling', v as MediaOptions['color_handling'])}
              options={COLOR_HANDLING_OPTIONS}
            />
          </Field>
          <Field label="Estabilización">
            <Select
              ariaLabel="Estabilización de video"
              value={mediaOptions.stabilization}
              onChange={(v) => updateMedia('stabilization', v as MediaOptions['stabilization'])}
              options={STABILIZATION_OPTIONS}
            />
          </Field>
          <Field label="Reducción de ruido de video">
            <Select
              ariaLabel="Reducción de ruido de video"
              value={mediaOptions.video_denoise}
              onChange={(v) => updateMedia('video_denoise', v as MediaOptions['video_denoise'])}
              options={VIDEO_DENOISE_OPTIONS}
            />
          </Field>
          <Field label="Limpieza de audio">
            <Select
              ariaLabel="Limpieza de audio"
              value={mediaOptions.audio_cleanup}
              onChange={(v) => updateMedia('audio_cleanup', v as MediaOptions['audio_cleanup'])}
              options={AUDIO_CLEANUP_OPTIONS}
            />
          </Field>
        </div>
      </details>

      <section className="upload-subtitles section-stack">
        <div className="section-card">
          <div className="mb-4 text-sm font-semibold text-ink">Configuracion de subtitulos</div>

          <div className="grid gap-3 sm:grid-cols-2">
            <ToggleField label="Subtitulos quemados" checked={captionsEnabled} onChange={setCaptionsEnabled} />
            <ToggleField
              label="Evitar sponsors (SponsorBlock)"
              checked={sponsorblockEnabled}
              onChange={setSponsorblockEnabled}
            />
          </div>

          {!captionsEnabled && (
            <p className="mt-4 text-sm leading-5 text-ink-soft">
              El video se renderizara sin texto ni archivos de subtitulos.
            </p>
          )}

          {captionsEnabled && <>
          <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <Field label="Estilo">
              <Select
                ariaLabel="Estilo de subtitulos"
                value={captionStyle}
                onChange={(v) => {
                  setCaptionStyle(v)
                  setStyleConfig((prev) => ({ ...prev, style: v }))
                }}
                options={captionStyles.map((style) => ({ value: style.value, label: style.label }))}
              />
            </Field>
            <Field
              label="Tamaño"
              value={`${sliderStep(styleConfig.size, SUBTITLE_SIZE_RANGE).index}/${
                sliderStep(styleConfig.size, SUBTITLE_SIZE_RANGE).total
              }`}
            >
              <div className="flex h-10 items-center">
                <input
                  type="range"
                  min={SUBTITLE_SIZE_RANGE.min}
                  max={SUBTITLE_SIZE_RANGE.max}
                  step={SUBTITLE_SIZE_RANGE.step}
                  value={styleConfig.size}
                  onChange={(e) => setStyleConfig((prev) => ({ ...prev, size: Number(e.target.value) }))}
                  className="w-full cursor-pointer"
                />
              </div>
            </Field>

            <Field label="Posicion">
              <Select
                ariaLabel="Posicion"
                value={styleConfig.position}
                onChange={(v) => setStyleConfig((prev) => ({ ...prev, position: v }))}
                options={[
                  { value: 'lower_middle', label: 'Lower-middle' },
                  { value: 'middle', label: 'Centro' },
                  { value: 'top', label: 'Arriba' },
                ]}
              />
            </Field>

            <Field label="Color destacado">
              <Select
                ariaLabel="Color destacado"
                value={styleConfig.highlight_color}
                onChange={(v) => setStyleConfig((prev) => ({ ...prev, highlight_color: v }))}
                options={Object.entries(highlightColors).map(([name, value]) => ({
                  value,
                  swatch: value,
                  label:
                    name === 'yellow'
                      ? 'Amarillo'
                      : name === 'cyan'
                        ? 'Cian'
                        : name === 'green'
                          ? 'Verde'
                          : name === 'red'
                            ? 'Rojo'
                            : name === 'pink'
                              ? 'Rosa'
                              : 'Blanco',
                }))}
              />
            </Field>

            <Field label="Color principal">
              <input
                type="color"
                value={styleConfig.text_color}
                onChange={(e) => setStyleConfig((prev) => ({ ...prev, text_color: e.target.value }))}
                className="h-10 w-full cursor-pointer rounded-lg border border-border bg-base p-1"
              />
            </Field>

            <Field
              label="Contorno"
              value={`${sliderStep(styleConfig.stroke_width, SUBTITLE_STROKE_RANGE).index}/${
                sliderStep(styleConfig.stroke_width, SUBTITLE_STROKE_RANGE).total
              }`}
            >
              <div className="flex h-10 items-center">
                <input
                  type="range"
                  min={SUBTITLE_STROKE_RANGE.min}
                  max={SUBTITLE_STROKE_RANGE.max}
                  step={SUBTITLE_STROKE_RANGE.step}
                  value={styleConfig.stroke_width}
                  onChange={(e) => setStyleConfig((prev) => ({ ...prev, stroke_width: Number(e.target.value) }))}
                  className="w-full cursor-pointer"
                />
              </div>
            </Field>
          </div>

          <p className="mt-3 text-xs leading-relaxed text-ink-soft">
            {activeCaptionStyle?.description ?? 'Preset visual para los subtitulos quemados.'}
          </p>

          <div className="mt-5">
            <div className="mb-2 text-xs text-ink-mute">Opciones</div>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <ToggleField label="Sombra" checked={styleConfig.shadow_enabled} onChange={(v) => setStyleConfig((prev) => ({ ...prev, shadow_enabled: v }))} />
              <ToggleField label="Caja" checked={styleConfig.background_box} onChange={(v) => setStyleConfig((prev) => ({ ...prev, background_box: v }))} />
              <ToggleField label="Mayusculas" checked={styleConfig.uppercase} onChange={(v) => setStyleConfig((prev) => ({ ...prev, uppercase: v }))} />
              <ToggleField label="Emojis" checked={styleConfig.emoji_enabled} onChange={(v) => setStyleConfig((prev) => ({ ...prev, emoji_enabled: v }))} />
            </div>
          </div>

          <div className="mt-5">
            <div className="mb-2 text-xs text-ink-mute">Vista previa del subtitulo</div>
            <SubtitlePreview config={effectiveStyleConfig} />
          </div>
          </>}
        </div>

        <PipelineSummary settings={settings ?? null} processingMode={effectiveProcessingMode} captionsEnabled={captionsEnabled} />
      </section>
      </>}

      {downloadOnly && (
        <section className="panel flex flex-col gap-3 p-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="text-sm font-semibold text-ink">Video original, sin procesamiento</div>
            <p className="mt-1 text-sm leading-5 text-ink-soft">Auto usa la mejor calidad que el video realmente tenga. Las demas opciones fijan un maximo.</p>
          </div>
          <div className="w-full shrink-0 sm:w-56">
            <Select
              ariaLabel="Calidad de descarga"
              value={downloadQuality}
              onChange={setDownloadQuality}
              options={DOWNLOAD_QUALITY_OPTIONS}
            />
          </div>
        </section>
      )}

      {audioOnly && (
        <section className="panel flex flex-col gap-3 p-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="text-sm font-semibold text-ink">Solo audio, sin video</div>
            <p className="mt-1 text-sm leading-5 text-ink-soft">Se descarga unicamente la pista de audio y se convierte al formato elegido.</p>
          </div>
          <div className="w-full shrink-0 sm:w-56">
            <Select
              ariaLabel="Formato de audio"
              value={audioFormat}
              onChange={setAudioFormat}
              options={AUDIO_FORMAT_OPTIONS}
            />
          </div>
        </section>
      )}

      {batchResult && (
        <div className="card space-y-3 p-5">
          <div className="text-sm font-semibold text-ink">Resultado del batch</div>
          <p className="text-sm leading-5 text-ink-soft">
            {batchResult.okCount} job{batchResult.okCount === 1 ? '' : 's'} creado{batchResult.okCount === 1 ? '' : 's'} correctamente.
            {batchResult.failed.length > 0 && ` ${batchResult.failed.length} fallaron.`}
          </p>
          {batchResult.failed.length > 0 && (
            <ul className="max-h-40 space-y-1 overflow-y-auto rounded-lg border border-bad/30 bg-bad/[0.06] p-3 text-xs text-bad">
              {batchResult.failed.map((item, index) => (
                <li key={`${item.url}-${index}`} className="break-all">
                  {item.url} — {item.reason}
                </li>
              ))}
            </ul>
          )}
          <button
            type="button"
            onClick={() => navigate('/jobs')}
            className="btn-primary"
          >
            Ir a Mis trabajos
          </button>
        </div>
      )}
    </div>
  )
}

function SubtitlePreview({ config }: { config: SubtitleStyleConfig }) {
  const fontSize = Math.round(30 * config.size)

  const textColor = config.text_color || '#FFFFFF'
  const highlightColor = config.highlight_color || '#FFD21F'
  const strokeColor = config.stroke_color || '#000000'
  const strokeWidth = Math.max(2, Math.round(config.stroke_width * 0.5))

  return (
    <div className="flex min-h-[120px] items-center justify-center rounded-lg border border-border bg-black/50 px-4 py-6 text-center">
      <div
        className="font-extrabold leading-tight"
        style={{
          color: textColor,
          WebkitTextFillColor: textColor,
          fontSize,
          WebkitTextStroke: `${strokeWidth}px ${strokeColor}`,
          paintOrder: 'stroke fill',
          textShadow: config.shadow_enabled ? '0 4px 8px rgba(0,0,0,.85)' : 'none',
          background: config.background_box ? 'rgba(0,0,0,.62)' : 'transparent',
          padding: config.background_box ? '6px 12px' : 0,
          borderRadius: config.background_box ? 8 : 0,
        }}
      >
        {config.uppercase ? 'PORQUE' : 'Porque'}
        <br />
        {config.uppercase ? 'ESTÁ ' : 'Está '}
        <span style={{ color: highlightColor, WebkitTextFillColor: highlightColor }}>
          {config.uppercase ? 'LLENO' : 'lleno'}
        </span>
      </div>
    </div>
  )
}

function Field({
  label,
  value,
  children,
}: {
  label: string
  /** Valor a la derecha del label (p. ej. "7/12" de un slider). */
  value?: ReactNode
  children: ReactNode
}) {
  return (
    <label className="block">
      <span className="mb-1.5 flex items-center justify-between text-xs text-ink-mute">
        <span>{label}</span>
        {value != null && <span className="num text-cyan">{value}</span>}
      </span>
      {children}
    </label>
  )
}


function PipelineSummary({
  settings,
  processingMode,
  captionsEnabled,
}: {
  settings: import('../lib/types').AppSettings | null
  processingMode: ProcessingMode
  captionsEnabled: boolean
}) {
  const reframeLabels: Record<string, string> = {
    center: 'Centrar en cara (estático)',
    track: 'Seguimiento (animado)',
    auto: 'Auto',
    preserve: 'Conservar encuadre',
    blur: 'Fondo difuminado',
  }
  const whisperLabels: Record<string, string> = {
    'large-v3': 'large-v3 · máx. precisión',
    'large-v2': 'large-v2',
    medium: 'medium · equilibrado',
    small: 'small · rápido',
    base: 'base · muy rápido',
    tiny: 'tiny',
  }

  const reframe = settings?.reframe_mode ?? 'auto'
  const whisper = settings?.whisper_model ?? '—'
  const minS = settings?.min_clip_seconds ?? '—'
  const maxS = settings?.max_clip_seconds ?? '—'
  const quality = settings?.download_quality ?? '—'
  const signals = settings?.enable_media_signals ?? false
  const visualRerank = settings?.enable_visual_rerank ?? false

  const isPreserve = processingMode === 'tutorial_screen' || processingMode === 'keep_original'
  const effectiveReframe = isPreserve ? 'preserve' : reframe

  // Resumen compacto de la config global (vive en Ajustes). Solo informativo:
  // muestra de un vistazo Whisper/reencuadre/duracion sin ocupar media fila.
  const chips: Array<{ label: string; value: string }> = [
    { label: 'Transcripción', value: whisperLabels[whisper] ?? whisper },
    {
      label: 'Reencuadre',
      value: (reframeLabels[effectiveReframe] ?? effectiveReframe) + (isPreserve ? ' (forzado)' : ''),
    },
    { label: 'Duración', value: `${minS}–${maxS}s` },
    { label: 'Calidad', value: quality },
    { label: 'Subtitulos', value: captionsEnabled ? 'Quemados' : 'Off' },
  ]
  if (signals) chips.push({ label: 'Análisis', value: 'Escena' })
  if (visualRerank) chips.push({ label: 'Análisis', value: 'Rerank visual IA' })

  return (
    <div className="panel flex flex-wrap items-center gap-x-2 gap-y-1.5 px-4 py-3">
      <span className="text-xs font-semibold text-ink">Pipeline activo</span>
      {chips.map(({ label, value }) => (
        <span
          key={label + value}
          className="inline-flex items-center gap-1 rounded-md border border-border bg-base/40 px-2 py-1 text-[11px] text-ink-mute"
        >
          {label}: <span className="font-medium text-ink-soft">{value}</span>
        </span>
      ))}
      <a
        href="/settings"
        className="ml-auto text-[11px] text-ink-mute underline-offset-2 hover:text-ink hover:underline"
      >
        Editar en Ajustes →
      </a>
    </div>
  )
}
