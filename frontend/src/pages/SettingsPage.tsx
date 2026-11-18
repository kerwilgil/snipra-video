import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Select } from '../components/Select'
import { NvidiaModelField } from '../components/NvidiaModelField'
import { ToggleField } from '../components/ToggleSwitch'
import { TopbarActions, TopbarTitle } from '../components/topbar'
import { api, type ApiKeysUpdate, type SettingsUpdate } from '../lib/api'
import { COOKIES_PATH_PLACEHOLDER, providerLabel } from '../lib/format'
import type { AppSettings, DurationPreset, MediaCapabilities, StorageStatus } from '../lib/types'

// Modelos sugeridos por proveedor. El primero de cada lista es el default del backend
// (providers/registry.py). El note describe el trade-off del modelo.
type ModelOption = { id: string; note: string }
const MODEL_SUGGESTIONS: Record<string, ModelOption[]> = {
  claude: [
    { id: 'claude-haiku-4-5', note: 'rápido, más barato' },
    { id: 'claude-sonnet-4-6', note: 'equilibrado' },
    { id: 'claude-opus-4-8', note: 'máxima calidad' },
  ],
  openai: [
    { id: 'gpt-4o-mini', note: 'rápido, barato' },
    { id: 'gpt-5.4-nano', note: 'el más rápido y barato' },
    { id: 'gpt-5.4-mini', note: 'rápido, equilibrado' },
    { id: 'gpt-5.4', note: 'máxima calidad' },
    { id: 'gpt-4.1-mini', note: 'rápido, barato' },
  ],
  // Suscripcion via CLI local (no API key). Claude Code acepta alias (haiku/sonnet/
  // opus) o nombre completo del modelo.
  // IDs completos (no alias) verificados en vivo con el CLI: modelUsage en la
  // respuesta confirma que usa exactamente ese modelo, sin ambiguedad.
  claude_code: [
    { id: 'claude-haiku-4-5-20251001', note: 'rápido, menos cuota' },
    { id: 'claude-sonnet-5', note: 'equilibrado' },
    { id: 'claude-opus-4-8', note: 'máxima calidad' },
    { id: 'claude-fable-5', note: 'Fable 5' },
  ],
  // Verificados contra el catalogo real del CLI (models_cache.json), no inventados.
  // Solo la generacion 5.6 mas reciente (Kerwil pidio quitar 5.4/5.5, ya viejos).
  codex: [
    { id: 'gpt-5.6-luna', note: 'rápido y económico' },
    { id: 'gpt-5.6-terra', note: 'equilibrado, uso diario' },
    { id: 'gpt-5.6-sol', note: 'máxima calidad, frontier' },
  ],
}

export function SettingsPage() {
  const qc = useQueryClient()
  const { data: s, isLoading, isError, error, isFetching, refetch } = useQuery({ queryKey: ['settings'], queryFn: api.getSettings })
  const { data: viaAvailability } = useQuery({ queryKey: ['via-availability'], queryFn: api.getViaAvailability })
  const { data: storage } = useQuery({ queryKey: ['storage'], queryFn: api.getStorage })
  const [form, setForm] = useState<SettingsUpdate | null>(null)

  const save = useMutation({
    mutationFn: api.updateSettings,
    onSuccess: (updated) => {
      qc.setQueryData(['settings'], updated)
      setForm(settingsToForm(updated))
      qc.invalidateQueries({ queryKey: ['settings'] })
      qc.invalidateQueries({ queryKey: ['via-availability'] })
    },
  })

  const cleanup = useMutation({
    mutationFn: api.cleanupStorage,
    onSuccess: (result) => {
      qc.setQueryData(['storage'], result.status)
      qc.invalidateQueries({ queryKey: ['storage'] })
    },
  })

  const [revealedToken, setRevealedToken] = useState<string | null>(null)
  const regenerateToken = useMutation({
    mutationFn: api.regenerateExtensionToken,
    onSuccess: (updated) => {
      // el valor crudo solo viaja en esta respuesta puntual; el GET normal de
      // /api/settings nunca lo vuelve a exponer (ver extension_token_configured)
      setRevealedToken(updated.extension_token ?? null)
      qc.invalidateQueries({ queryKey: ['settings'] })
    },
  })

  if (isLoading) return <div className="page-container page-container--form"><TopbarTitle><span className="topbar-context">Sistema</span><strong>Ajustes</strong></TopbarTitle><p role="status" className="page-description">Cargando ajustes...</p></div>
  if (isError || !s) return (
    <div className="page-container page-container--form section-stack">
      <TopbarTitle><span className="topbar-context">Sistema</span><strong>Ajustes</strong></TopbarTitle>
      <header className="page-header"><h1>Ajustes</h1><p className="page-description">Configura el procesamiento y las preferencias de Snipra.</p></header>
      <div role="alert" className="page-error">
        <div><strong>No se pudieron cargar los ajustes.</strong><p>Comprueba que Snipra esté en marcha. {error instanceof Error ? error.message : ''}</p></div>
        <button type="button" className="btn-secondary" disabled={isFetching} onClick={() => void refetch()}>{isFetching ? 'Reintentando...' : 'Reintentar'}</button>
      </div>
    </div>
  )
  const current = form ?? settingsToForm(s)
  const selectedVisionProvider = current.vision_provider ?? s.vision_provider ?? 'auto'
  const effectiveVisionProvider =
    selectedVisionProvider === 'auto' ? (current.provider ?? s.provider) : selectedVisionProvider
  const visionModels =
    s.vision_models_by_provider?.[effectiveVisionProvider] ??
    (effectiveVisionProvider === s.vision_effective?.provider ? (s.vision_models ?? []) : [])
  const effectiveVisionModel =
    current.vision_model ||
    (selectedVisionProvider === 'auto' ? current.model : s.vision_default_models?.[effectiveVisionProvider]) ||
    ''
  const visionSupported = Boolean(s.vision_default_models?.[effectiveVisionProvider])

  function update<K extends keyof SettingsUpdate>(key: K, value: SettingsUpdate[K]) {
    setForm((prev) => ({ ...(prev ?? current), [key]: value }))
  }

  // Al cambiar de proveedor, el modelo guardado (de OTRO proveedor) ya no aplica --
  // limpiarlo para que ModelField muestre el default del proveedor nuevo en vez de
  // arrastrar "personalizado" con un modelo que no le corresponde.
  function updateProvider(provider: string) {
    setForm((prev) => ({ ...(prev ?? current), provider, model: provider === 'nvidia' ? 'auto' : '' }))
  }

  function updateVisionProvider(provider: string) {
    setForm((prev) => ({
      ...(prev ?? current),
      vision_provider: provider,
      vision_model: '',
    }))
  }

  return (
    <div className="page-container page-container--form section-stack">
      <TopbarTitle>
        <span className="topbar-context">Sistema</span>
        <strong>Ajustes</strong>
      </TopbarTitle>
      <TopbarActions>
        <button
          type="button"
          onClick={() => save.mutate(current)}
          disabled={save.isPending}
          className="btn-primary disabled:opacity-60"
        >
          {save.isPending ? 'Guardando...' : 'Guardar ajustes'}
        </button>
      </TopbarActions>
      <header className="page-header">
        <h1>Ajustes</h1>
        <p className="page-description">Estos cambios se guardan en el almacenamiento local de Snipra y se aplican a nuevos trabajos.</p>
      </header>

      {save.isError && (
        <div className="rounded-lg border border-bad/30 bg-bad/[0.06] px-4 py-3 text-sm text-bad">
          No se pudieron guardar los ajustes.
          {save.error instanceof Error ? ` ${save.error.message}` : ''}
        </div>
      )}
      {save.isSuccess && (
        <div className="rounded-lg border border-good/30 bg-good/[0.06] px-4 py-3 text-sm text-good">
          Ajustes guardados.
        </div>
      )}

      <div className="config-grid grid grid-cols-1 gap-4 lg:grid-cols-2">
          <section className="card min-w-0 space-y-4 p-5">
            <h2 className="text-sm font-semibold text-ink">IA de deteccion</h2>
            {viaAvailability && !viaAvailability.available && (
              <div className="rounded-lg border border-warn/30 bg-warn/[0.06] px-3 py-2.5 text-xs leading-relaxed text-ink-soft">
                <strong>VIA no está configurado</strong>
                <br />
                {viaAvailability.code === 'VIA_MISSING_PROVIDER_CREDENTIALS' ? (
                  'Claude está seleccionado, pero falta ANTHROPIC_API_KEY.'
                ) : viaAvailability.code === 'VIA_VISUAL_PROVIDER_UNSUPPORTED' ? (
                  'Configura un proveedor de IA compatible (Claude, OpenAI) para usar el análisis inteligente de video. Snipra puede seguir utilizándose sin VIA.'
                ) : (
                  'Configura un proveedor de IA compatible para usar el análisis inteligente de video. Snipra puede seguir utilizándose sin VIA.'
                )}
              </div>
            )}
            <SelectField
              label="Proveedor"
              value={current.provider ?? ''}
              onChange={updateProvider}
              options={s.providers.map((provider) => ({ value: provider, label: providerLabel(provider) }))}
            />
            <ModelField
              provider={current.provider ?? ''}
              value={current.model ?? ''}
              onChange={(value) => update('model', value)}
            />
            <SubscriptionStatus provider={current.provider ?? ''} settings={s} />
            <ToggleField
              label="Analisis audio/escena"
              checked={current.enable_media_signals ?? true}
              onChange={(value) => update('enable_media_signals', value)}
            />
            <ToggleField
              label="Analisis visual IA (mira frames, usa mas tokens)"
              checked={current.enable_visual_rerank ?? false}
              onChange={(value) => update('enable_visual_rerank', value)}
            />
            {(current.enable_visual_rerank ?? false) && (
              <div className="space-y-2">
                <SelectField
                  label="Proveedor de vision"
                  value={selectedVisionProvider}
                  onChange={updateVisionProvider}
                  options={(s.vision_providers ?? ['auto']).map((p) => ({
                    value: p,
                    label: p === 'auto' ? 'Igual que el de texto (auto)' : p,
                  }))}
                />
                {/* Solo los modelos del proveedor de vision efectivo. La union de
                    todos mostraba claude-sonnet-5 dos veces y gpt-4o-mini aun
                    habiendo elegido codex. */}
                {selectedVisionProvider === 'nvidia' ? <NvidiaModelField
                  label="Modelo de visión NVIDIA (VIA)"
                  vision
                  value={current.nvidia_vision_model ?? 'auto'}
                  onChange={(value) => update('nvidia_vision_model', value)}
                /> : <SelectField
                  label="Modelo de vision"
                  value={current.vision_model ?? s.vision_model ?? ''}
                  onChange={(value) => update('vision_model', value)}
                  options={[
                    { value: '', label: 'Por defecto del proveedor' },
                    ...visionModels.map((m) => ({ value: m, label: m })),
                  ]}
                />}
                {/* "auto" por si solo no dice que se va a usar: se muestra el
                    proveedor y modelo EFECTIVOS que resuelve el backend. */}
                {selectedVisionProvider === 'nvidia' ? (
                  <p className="text-xs text-ink-mute">Visión NVIDIA se aplica a VIA. El reranking visual permanece deshabilitado para este proveedor.</p>
                ) : visionSupported ? (
                  <p className="text-[11px] leading-tight text-ink-mute">
                    En uso al guardar: <strong>{effectiveVisionProvider}</strong>
                    {effectiveVisionModel ? ` / ${effectiveVisionModel}` : ''}
                  </p>
                ) : (
                  <p className="text-xs leading-tight text-warn">
                    El proveedor de texto actual no tiene vision: el analisis visual se omitira.
                  </p>
                )}
                <p className="text-xs leading-tight text-ink-mute">
                  Con <strong>auto</strong> la vision usa el mismo proveedor que el texto, para que
                  elegir una suscripcion (claude_code / codex) no termine llamando a una API de pago.
                </p>
              </div>
            )}
            <ApiKeysPanel settings={s} />
          </section>

        <section className="card min-w-0 space-y-4 p-5">
          <h2 className="text-sm font-semibold text-ink">Procesamiento</h2>
          <TextField
            label="Whisper"
            value={current.whisper_model ?? ''}
            onChange={(value) => update('whisper_model', value)}
          />
          <SelectField
            label="Reframe"
            value={current.reframe_mode ?? 'auto'}
            onChange={(value) => update('reframe_mode', value)}
            options={['auto', 'preserve', 'center', 'blur', 'track', 'split'].map((value) => ({
              value,
              label: value === 'split' ? 'split (2 personas, una arriba y otra abajo)' : value,
            }))}
          />
          <NumberField
            label="Clips por defecto"
            value={current.default_num_clips ?? 5}
            min={1}
            max={12}
            onChange={(value) => update('default_num_clips', value)}
          />
          <DurationPresetField
            presets={s.duration_presets ?? []}
            value={current.clip_duration_preset ?? 'standard'}
            onChange={(preset) => {
              update('clip_duration_preset', preset.value)
              update('min_clip_seconds', preset.min)
              update('max_clip_seconds', preset.max)
            }}
          />
          <div className="grid gap-3 sm:grid-cols-2">
            <NumberField
              label="Min segundos por clip"
              value={current.min_clip_seconds ?? 45}
              min={3}
              max={120}
              step={1}
              onChange={(value) => update('min_clip_seconds', value)}
            />
            <NumberField
              label="Max segundos por clip"
              value={current.max_clip_seconds ?? 90}
              min={8}
              max={300}
              step={1}
              onChange={(value) => update('max_clip_seconds', value)}
            />
          </div>
          <CheckboxGroup
            label="Formatos por defecto"
            values={current.default_formats ?? []}
            options={s.available_formats}
            onChange={(values) => update('default_formats', values)}
          />
          <ToggleField
            label={
              s.ai_upscale_available
                ? 'Super-resolucion IA (GPU, mas nitido pero lento)'
                : 'Super-resolucion IA (requiere Real-ESRGAN)'
            }
            checked={current.enable_ai_upscale ?? false}
            onChange={(value) => update('enable_ai_upscale', value)}
          />
          {!s.ai_upscale_available && (
            <div className="rounded-lg border border-warn/30 bg-warn/[0.06] px-3 py-2.5 text-xs leading-relaxed text-ink-soft">
              No se detecto Real-ESRGAN. Descargalo, descomprimilo y pon la ruta del binario en{' '}
              <code className="text-ink">REALESRGAN_PATH</code> (backend/.env), luego reinicia.
              <a
                href="https://github.com/xinntao/Real-ESRGAN/releases"
                target="_blank"
                rel="noopener noreferrer"
                className="mt-1.5 inline-flex items-center gap-1 font-semibold text-cyan hover:underline"
              >
                Descargar Real-ESRGAN ↗
              </a>
            </div>
          )}
        </section>
      </div>

      <ExtensionTokenPanel
        token={revealedToken}
        configured={s.extension_token_configured}
        regenerating={regenerateToken.isPending}
        onRegenerate={() => regenerateToken.mutate()}
      />

      <section className="card grid gap-x-6 gap-y-4 p-5 lg:grid-cols-2">
        <div className="flex flex-col gap-4">
          <h2 className="text-sm font-semibold text-ink">Subtitulos</h2>
          <SelectField
            label="Estilo por defecto"
            value={current.caption_style ?? 'viral_bold'}
            onChange={(value) => update('caption_style', value)}
            options={s.caption_styles.map((style) => ({ value: style.value, label: style.label }))}
          />
          <div className="flex-1 rounded-lg border border-border bg-base/40 p-3 text-xs leading-5 text-ink-mute">
            {s.caption_styles.find((style) => style.value === current.caption_style)?.description ??
              'Preset visual para subtitulos quemados.'}
          </div>
          <div>
            <h3 className="mb-2 text-xs font-semibold text-ink">SponsorBlock</h3>
            <CheckboxGroup
              label="Categorias excluidas del pool de clips (YouTube)"
              values={current.sponsorblock_categories ?? []}
              options={s.sponsorblock_categories_available}
              onChange={(values) => update('sponsorblock_categories', values)}
            />
            <p className="mt-1.5 text-xs leading-tight text-ink-mute">
              Tramos marcados por SponsorBlock en estas categorias se recortan o descartan del pool de
              candidatos antes de generar clips. Solo aplica a videos de YouTube.
            </p>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <h2 className="text-sm font-semibold text-ink">Cookies / descarga</h2>
          <SelectField
            label="Calidad de descarga"
            value={current.download_quality ?? '1080'}
            onChange={(value) => update('download_quality', value)}
            options={[
              { value: 'auto', label: 'Auto (máxima disponible)' },
              { value: '720', label: '720p (más liviano y rápido)' },
              { value: '1080', label: '1080p (recomendado)' },
              { value: '1440', label: '2K / 1440p (más pesado)' },
              { value: '2160', label: '4K / 2160p (ideal para 9:16 nítido)' },
            ]}
          />
          <DownloadCookiesFields
            browser={current.yt_dlp_cookies_from_browser ?? ''}
            path={current.yt_dlp_cookies_path ?? ''}
            onBrowserChange={(value) => update('yt_dlp_cookies_from_browser', value)}
            onPathChange={(value) => update('yt_dlp_cookies_path', value)}
          />
          <NumberField
            label="Retencion de archivos (dias; 0 desactiva)"
            value={current.storage_retention_days ?? 0}
            min={0}
            max={3650}
            step={1}
            onChange={(value) => update('storage_retention_days', value)}
          />
        </div>
      </section>

      <section className="card grid gap-x-6 gap-y-4 p-5 lg:grid-cols-2">
        <div className="flex flex-col gap-4">
          <h2 className="text-sm font-semibold text-ink">Mejoras de medios</h2>
          <p className="text-xs leading-5 text-ink-mute">
            Valores por defecto para nuevos trabajos. Cada trabajo puede
            sobrescribirlos desde la pantalla de creación.
          </p>
          <SelectField
            label="Destino por defecto"
            value={current.destination_profile ?? 'auto'}
            onChange={(value) => update('destination_profile', value)}
            options={[
              { value: 'auto', label: 'Auto (según formato)' },
              ...(s.destination_profiles ?? []).map((profile) => ({
                value: profile.id,
                label: `${profile.display_name} (${profile.frame})`,
              })),
            ]}
          />
          <SelectField
            label="Color HDR"
            value={current.color_handling ?? 'auto'}
            onChange={(value) => update('color_handling', value as SettingsUpdate['color_handling'])}
            options={[
              { value: 'auto', label: 'Auto (convierte HDR si el destino lo pide)' },
              { value: 'sdr', label: 'Siempre SDR (tone map)' },
              { value: 'preserve', label: 'Intentar preservar color' },
            ]}
          />
        </div>
        <div className="flex flex-col gap-4">
          <SelectField
            label="Estabilización"
            value={current.stabilization ?? 'off'}
            onChange={(value) => update('stabilization', value as SettingsUpdate['stabilization'])}
            options={[
              { value: 'off', label: 'Desactivada' },
              { value: 'auto', label: 'Auto' },
              { value: 'light', label: 'Suave' },
              { value: 'medium', label: 'Media' },
              { value: 'strong', label: 'Fuerte' },
            ]}
          />
          <SelectField
            label="Reducción de ruido de video"
            value={current.video_denoise ?? 'off'}
            onChange={(value) => update('video_denoise', value as SettingsUpdate['video_denoise'])}
            options={[
              { value: 'off', label: 'Desactivada' },
              { value: 'light', label: 'Suave' },
              { value: 'medium', label: 'Media' },
              { value: 'strong', label: 'Fuerte' },
            ]}
          />
          <SelectField
            label="Limpieza de audio"
            value={current.audio_cleanup ?? 'off'}
            onChange={(value) => update('audio_cleanup', value as SettingsUpdate['audio_cleanup'])}
            options={[
              { value: 'off', label: 'Desactivada' },
              { value: 'voice_light', label: 'Voz suave' },
              { value: 'voice', label: 'Voz' },
              { value: 'voice_strong', label: 'Voz fuerte' },
            ]}
          />
        </div>
      </section>

      <DiagnosticsPanel
        settings={s}
        storage={storage}
        cleanupPending={cleanup.isPending}
        onCleanup={() => cleanup.mutate()}
      />

      <MediaDiagnosticsPanel />
    </div>
  )
}

function settingsToForm(settings: AppSettings): SettingsUpdate {
  return {
    provider: settings.provider,
    model: settings.model,
    whisper_model: settings.whisper_model,
    default_num_clips: settings.default_num_clips,
    clip_duration_preset: settings.clip_duration_preset,
    min_clip_seconds: settings.min_clip_seconds,
    max_clip_seconds: settings.max_clip_seconds,
    enable_media_signals: settings.enable_media_signals,
    enable_visual_rerank: settings.enable_visual_rerank,
    vision_provider: settings.vision_provider,
    vision_model: settings.vision_model,
    enable_ai_upscale: settings.enable_ai_upscale,
    nvidia_vision_model: settings.nvidia_vision_model,
    default_formats: settings.default_formats,
    sponsorblock_categories: settings.sponsorblock_categories,
    reframe_mode: settings.reframe_mode,
    yt_dlp_cookies_from_browser: settings.yt_dlp_cookies_from_browser,
    yt_dlp_cookies_path: settings.yt_dlp_cookies_path,
    caption_style: settings.caption_style,
    download_quality: settings.download_quality,
    storage_retention_days: settings.storage_retention_days,
    destination_profile: settings.media_defaults?.destination_profile ?? 'auto',
    color_handling: settings.media_defaults?.color_handling ?? 'auto',
    stabilization: settings.media_defaults?.stabilization ?? 'off',
    video_denoise: settings.media_defaults?.video_denoise ?? 'off',
    audio_cleanup: settings.media_defaults?.audio_cleanup ?? 'off',
  }
}

export function DownloadCookiesFields({ browser, path, onBrowserChange, onPathChange }: {
  browser: string
  path: string
  onBrowserChange: (value: string) => void
  onPathChange: (value: string) => void
}) {
  const mode = browser === 'detect' ? 'auto' : browser || (path ? 'cookies_file' : 'auto')
  const options = [
    { value: 'auto', label: 'Automático' },
    { value: 'none', label: 'Ninguno' },
    { value: 'chrome', label: 'Chrome' },
    { value: 'brave', label: 'Brave' },
    { value: 'edge', label: 'Edge' },
    { value: 'firefox', label: 'Firefox' },
    { value: 'cookies_file', label: 'Archivo cookies.txt' },
  ]
  // Conserva perfiles configurados anteriormente, sin cambiar la sesión elegida.
  if (!options.some((option) => option.value === mode)) {
    options.push({ value: mode, label: 'Perfil de navegador configurado' })
  }
  return (
    <div className="space-y-3">
      <span className="block text-xs text-ink-mute">Cookies para sitios con sesión</span>
      <Select ariaLabel="Cookies para sitios con sesión" value={mode} options={options} onChange={onBrowserChange} />
      <p className="text-[11px] leading-tight text-ink-mute">
        Permite utilizar una sesión ya iniciada para descargar contenido que requiera autenticación.
      </p>
      {mode !== 'none' && (
        <TextField label={mode === 'cookies_file' ? 'Ruta cookies.txt' : 'Ruta cookies.txt (fallback opcional)'}
          value={path} placeholder={COOKIES_PATH_PLACEHOLDER} onChange={onPathChange}
          hint="Ruta absoluta a un archivo local existente. Se guarda solo la ruta; el contenido no se muestra."
        />
      )}
      {mode === 'auto' && <p className="text-[11px] leading-tight text-ink-mute">Prueba los perfiles disponibles de Brave, Edge, Chrome y Firefox; después el archivo configurado.</p>}
    </div>
  )
}

function TextField({
  label,
  value,
  placeholder,
  hint,
  onChange,
}: {
  label: string
  value: string
  placeholder?: string
  hint?: string
  onChange: (value: string) => void
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-xs text-ink-mute">{label}</span>
      <input
        className="field w-full px-3 py-2.5 text-sm"
        aria-label={label}
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
      {hint && <span className="mt-1.5 block text-[11px] leading-tight text-ink-mute">{hint}</span>}
    </label>
  )
}

function ModelField({
  provider,
  value,
  onChange,
}: {
  provider: string
  value: string
  onChange: (value: string) => void
}) {
  if (provider === 'nvidia') return <NvidiaModelField value={value} onChange={onChange} />
  const suggestions = MODEL_SUGGESTIONS[provider] ?? []
  const defaultModel = suggestions[0]?.id

  if (provider === 'naive') {
    return (
      <label className="block">
        <span className="mb-2 block text-xs text-ink-mute">Modelo</span>
        <input
          className="field w-full px-3 py-2.5 text-sm opacity-50"
          value=""
          disabled
          placeholder="naive no usa modelo de IA"
        />
      </label>
    )
  }

  // Vacio = usar el default del backend; mostramos ese modelo seleccionado.
  // Si el valor guardado es custom (no esta en la lista), lo agregamos para no perderlo.
  const isCustom = value !== '' && !suggestions.some((m) => m.id === value)

  const options = [
    ...suggestions.map((model) => ({ value: model.id, label: `${model.id} (${model.note})` })),
    ...(isCustom ? [{ value, label: `${value} (personalizado)` }] : []),
  ]

  return (
    <label className="block">
      <span className="mb-2 block text-xs text-ink-mute">Modelo</span>
      <Select
        ariaLabel="Modelo"
        value={value || defaultModel || ''}
        options={options}
        onChange={onChange}
      />
    </label>
  )
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-xs text-ink-mute">{label}</span>
      <input
        className="field w-full px-3 py-2.5 text-sm"
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  )
}

function ExtensionTokenPanel({
  token,
  configured,
  regenerating,
  onRegenerate,
}: {
  /** Valor crudo solo disponible justo despues de generar/regenerar (revelado una vez). */
  token: string | null
  /** Si ya hay un token guardado en el servidor, aunque este componente no lo pueda leer. */
  configured: boolean
  regenerating: boolean
  onRegenerate: () => void
}) {
  const [copied, setCopied] = useState(false)

  async function copy() {
    if (!token) return
    await navigator.clipboard.writeText(token)
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  const displayValue = token || (configured ? 'Token configurado (oculto por seguridad)' : 'Sin token generado')

  return (
    <section className="card flex flex-1 flex-col space-y-3 p-5">
      <div>
        <h2 className="text-sm font-semibold text-ink">Extension de navegador</h2>
        <p className="mt-1 text-xs text-ink-mute">
          Token requerido para que la extension de Snipra (popup, uso local) pueda crear jobs. Pegalo
          una sola vez en la extension. Sin token configurado, la extension no puede mutar nada. Por
          seguridad, el valor solo se muestra una vez, justo despues de generarlo o regenerarlo — si
          lo perdiste, regeneralo y pegalo de nuevo en la extension.
        </p>
      </div>

      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <input
          className="field w-full px-3 py-2.5 text-xs"
          value={displayValue}
          readOnly
        />
        <div className="flex shrink-0 gap-2">
          <button
            type="button"
            onClick={copy}
            disabled={!token}
            className="rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan disabled:opacity-50"
          >
            {copied ? 'Copiado ✓' : 'Copiar'}
          </button>
          <button
            type="button"
            onClick={onRegenerate}
            disabled={regenerating}
            className="rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan disabled:opacity-50"
          >
            {regenerating ? 'Generando...' : configured ? 'Regenerar' : 'Generar token'}
          </button>
        </div>
      </div>
    </section>
  )
}

function DiagnosticsPanel({
  settings,
  storage,
  cleanupPending,
  onCleanup,
}: {
  settings: AppSettings
  storage?: StorageStatus
  cleanupPending: boolean
  onCleanup: () => void
}) {
  const bins = Object.entries(settings.binaries)
  return (
    <section className="card space-y-4 p-5">
      <div>
        <h2 className="text-sm font-semibold text-ink">Diagnostico local</h2>
        <p className="mt-1 text-xs text-ink-mute">
          Estos checks ayudan a detectar problemas antes de procesar videos.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        {bins.map(([name, bin]) => (
          <div
            key={name}
            className="rounded-lg border border-border bg-base/40 px-3 py-3 text-xs"
          >
            <div className="flex items-center justify-between gap-3">
              <span className="font-semibold text-ink">{name}</span>
              <span className={bin.available ? 'text-good' : 'text-bad'}>
                {bin.available ? 'OK' : 'Falta'}
              </span>
            </div>
            <div className="num mt-2 truncate text-ink-mute">{bin.path}</div>
            {bin.version && <div className="mt-1 truncate text-ink-mute">{bin.version}</div>}
          </div>
        ))}
      </div>

      <div className="rounded-lg border border-border bg-base/40 px-3 py-3 text-xs text-ink-mute">
        Cookies yt-dlp:{' '}
        <span className={settings.yt_dlp_cookies_configured ? 'text-good' : 'text-warn'}>
          {settings.yt_dlp_cookies_configured ? 'configuradas' : 'sin configurar'}
        </span>
      </div>

      {storage && (
        <>
          <div className="grid gap-2 md:grid-cols-3">
            <Metric label="Work (jobs)" value={formatBytes(storage.work_bytes)} />
            <Metric label="Jobs" value={String(storage.job_count)} />
            <Metric label="Huerfanos" value={String(storage.orphan_count)} />
          </div>
          <div className="grid gap-2 md:grid-cols-3">
            <Metric label="Compilaciones" value={formatBytes(storage.compositions_bytes)} />
            <Metric label="Compuestas" value={String(storage.compilation_count)} />
            <Metric
              label="Huerfanas"
              value={String(storage.orphan_composition_count)}
            />
          </div>
          <p className="text-xs leading-relaxed text-ink-soft">
            Las compilaciones <strong className="text-ink">no</strong> se borran solas: la
            retencion automatica solo aplica a artefactos de jobs, porque una compilacion es
            tu resultado final y no un temporal. Se borran una por una desde{' '}
            <Link to="/compose" className="text-cyan hover:underline">
              Componer
            </Link>
            . La limpieza de aqui solo elimina carpetas huerfanas (sin compilacion asociada).
          </p>
        </>
      )}

      <button
        type="button"
        onClick={onCleanup}
        disabled={cleanupPending}
        className="rounded-lg border border-border px-3 py-2 text-sm text-ink-soft hover:border-cyan/50 hover:text-cyan disabled:opacity-50"
      >
        {cleanupPending ? 'Limpiando...' : 'Limpiar archivos huerfanos'}
      </button>
    </section>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-base/40 px-3 py-2">
      <div className="text-[10px] text-ink-mute">{label}</div>
      <div className="num mt-1 text-sm font-semibold text-ink">{value}</div>
    </div>
  )
}

function formatBytes(bytes: number) {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let value = bytes
  let idx = 0
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024
    idx += 1
  }
  return `${value.toFixed(value >= 10 || idx === 0 ? 0 : 1)} ${units[idx]}`
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: Array<{ value: string; label: string }>
  onChange: (value: string) => void
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-xs text-ink-mute">{label}</span>
      <Select ariaLabel={label} value={value} options={options} onChange={onChange} />
    </label>
  )
}

function CheckboxGroup({
  label,
  values,
  options,
  onChange,
}: {
  label: string
  values: string[]
  options: string[]
  onChange: (values: string[]) => void
}) {
  return (
    <div>
      <div className="mb-2 text-xs text-ink-mute">{label}</div>
      <div className="flex flex-wrap gap-2">
        {options.map((option) => {
          const checked = values.includes(option)
          return (
            <button
              key={option}
              type="button"
              onClick={() =>
                onChange(checked ? values.filter((value) => value !== option) : [...values, option])
              }
              className={`rounded-lg border px-3 py-2 text-xs ${
                checked
                  ? 'border-cyan/50 bg-cyan/[0.08] text-cyan'
                  : 'border-border text-ink-soft hover:border-border-strong'
              }`}
            >
              {option}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function DurationPresetField({
  presets,
  value,
  onChange,
}: {
  presets: DurationPreset[]
  value: string
  onChange: (preset: DurationPreset) => void
}) {
  const active = presets.find((p) => p.value === value)
  return (
    <div>
      <div className="mb-2 text-xs text-ink-mute">Duración de clips</div>
      <div className="flex flex-col gap-2">
        {presets.map((p) => {
          const selected = p.value === value
          return (
            <button
              key={p.value}
              type="button"
              onClick={() => onChange(p)}
              className={`flex items-center justify-between gap-3 rounded-lg border px-3 py-2.5 text-left text-sm transition-colors ${
                selected
                  ? 'border-cyan/40 bg-cyan/[0.08] text-ink'
                  : 'border-border text-ink-soft hover:border-border-strong'
              }`}
            >
              <span className="min-w-0 font-medium">{p.label}</span>
              <span className={`shrink-0 rounded-md px-2 py-0.5 text-[11px] font-semibold ${selected ? 'bg-cyan/15 text-cyan' : 'bg-raised text-ink-mute'}`}>
                {p.min}–{p.max}s
              </span>
            </button>
          )
        })}
      </div>
      {active && (
        <p className="mt-1.5 text-xs leading-tight text-ink-mute">{active.description}</p>
      )}
    </div>
  )
}

const API_KEY_LABELS: Record<keyof ApiKeysUpdate, string> = {
  claude: 'Claude / Anthropic',
  openai: 'OpenAI',
  nvidia: 'NVIDIA NIM',
}

function ApiKeysPanel({ settings }: { settings: AppSettings }) {
  const qc = useQueryClient()
  const [draft, setDraft] = useState<ApiKeysUpdate>({})
  const mutation = useMutation({
    mutationFn: api.updateApiKeys,
    onSuccess: (updated) => {
      qc.setQueryData(['settings'], updated)
      setDraft({})
    },
  })

  function submit(key: keyof ApiKeysUpdate, value: string) {
    mutation.mutate({ [key]: value })
  }

  return (
    <div className="space-y-2">
      <div className="text-xs text-ink-mute">Claves API</div>
      <p className="text-[11px] leading-tight text-ink-mute">
        Se guardan solo en este equipo. El servidor nunca devuelve el valor: únicamente indica si
        hay una clave configurada.
      </p>
      <div className="space-y-2">
        {(Object.keys(API_KEY_LABELS) as Array<keyof ApiKeysUpdate>).map((key) => {
          const present = settings.keys_present[key] === true
          const value = draft[key] ?? ''
          const pending = mutation.isPending && mutation.variables?.[key] !== undefined
          return (
            <div key={key} className="rounded-lg border border-border p-2.5">
              <div className="mb-2 flex items-center justify-between gap-2 text-xs">
                <span className="font-medium text-ink">{API_KEY_LABELS[key]}</span>
                <span className={present ? 'text-good' : 'text-ink-mute'}>
                  {present ? 'Configurada' : 'Sin configurar'}
                </span>
              </div>
              <div className="flex flex-col gap-2 sm:flex-row">
                <input
                  type="password"
                  autoComplete="off"
                  value={value}
                  placeholder={present ? 'Nueva clave (la actual no se muestra)' : 'Pega la clave'}
                  onChange={(event) => setDraft((current) => ({ ...current, [key]: event.target.value }))}
                  className="field min-w-0 flex-1 px-3 py-2 text-xs"
                />
                <button
                  type="button"
                  disabled={!value.trim() || mutation.isPending}
                  onClick={() => submit(key, value.trim())}
                  className="btn-secondary disabled:opacity-50"
                >
                  {pending && value ? 'Guardando...' : 'Guardar'}
                </button>
                {present && (
                  <button
                    type="button"
                    disabled={mutation.isPending}
                    onClick={() => submit(key, '')}
                    className="rounded-lg border border-bad/30 px-3 py-2 text-xs font-semibold text-bad disabled:opacity-50"
                  >
                    {pending && !value ? 'Borrando...' : 'Borrar'}
                  </button>
                )}
              </div>
            </div>
          )
        })}
      </div>
      {mutation.isError && <p className="text-[11px] text-bad">No se pudo actualizar la clave.</p>}
    </div>
  )
}

// Aviso de disponibilidad cuando el proveedor elegido usa la suscripcion (CLI local).
function SubscriptionStatus({ provider, settings }: { provider: string; settings: AppSettings }) {
  const available = settings.subscription_available ?? {}
  if (provider !== 'claude_code' && provider !== 'codex') return null
  const cli = provider === 'claude_code' ? 'Claude Code' : 'Codex'
  const ok = available[provider] === true
  return (
    <div
      className={`rounded-lg border px-3 py-2.5 text-[11px] leading-relaxed ${
        ok
          ? 'border-good/25 bg-good/[0.06] text-ink-soft'
          : 'border-warn/30 bg-warn/[0.06] text-ink-soft'
      }`}
    >
      {ok ? (
        <>
          Usa tu suscripcion via el CLI de <span className="font-semibold text-ink">{cli}</span> (no
          consume creditos de API). El CLI esta instalado. Debe tener sesion iniciada en la maquina
          que corre Snipra; si expira, el job degrada al fallback local.
        </>
      ) : (
        <>
          El CLI de <span className="font-semibold text-ink">{cli}</span> no se detecto en esta
          maquina. Instalalo e inicia sesion una vez para usar tu suscripcion, o elige otro proveedor.
        </>
      )}
    </div>
  )
}

const CAPABILITY_LABELS: Record<string, string> = {
  available: 'OK',
  missing: 'No disponible',
  unknown: 'Desconocido',
  deshake_fallback: 'Limitado (deshake)',
}

const CAPABILITY_CLASSES: Record<string, string> = {
  available: 'text-good',
  missing: 'text-bad',
  unknown: 'text-ink-mute',
  deshake_fallback: 'text-warn',
}

export function MediaDiagnosticsPanel() {
  const qc = useQueryClient()
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['mediaCapabilities'],
    queryFn: () => api.getMediaCapabilities(),
  })
  const refresh = useMutation({
    mutationFn: () => api.getMediaCapabilities(true),
    onSuccess: (caps: MediaCapabilities) => {
      qc.setQueryData(['mediaCapabilities'], caps)
    },
  })

  const rows: Array<{ label: string; status: string; detail?: string }> = []
  if (data) {
    rows.push(
      { label: 'FFmpeg', status: data.ffmpeg_version ? 'available' : 'unknown', detail: data.ffmpeg_version || undefined },
      { label: 'FFprobe', status: data.ffprobe_version ? 'available' : 'unknown', detail: data.ffprobe_version || undefined },
      { label: 'H.264', status: data.encoders.libx264 ?? 'unknown' },
      { label: 'AAC', status: data.encoders.aac ?? 'unknown' },
      { label: 'Subtítulos', status: data.features.subtitles ?? 'unknown' },
      { label: 'HDR → SDR', status: data.features.hdr_to_sdr ?? 'unknown' },
      { label: 'Estabilización', status: data.features.stabilization ?? 'unknown' },
      { label: 'Denoise', status: data.features.video_denoise ?? 'unknown' },
      { label: 'Audio Cleanup', status: data.features.audio_cleanup ?? 'unknown' },
    )
  }

  return (
    <section className="card space-y-4 p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-ink">Diagnóstico multimedia</h2>
          <p className="mt-1 text-xs text-ink-mute">
            Capabilities reales de tu instalación de FFmpeg: qué mejoras de medios
            puede ejecutar este equipo.
          </p>
        </div>
        <button
          type="button"
          onClick={() => refresh.mutate()}
          disabled={refresh.isPending || isLoading}
          className="shrink-0 rounded-lg border border-border px-3 py-2 text-xs text-ink-soft transition-colors hover:border-cyan/50 hover:text-cyan disabled:opacity-50"
        >
          {refresh.isPending ? 'Actualizando...' : 'Actualizar diagnóstico'}
        </button>
      </div>

      {isLoading && <p className="text-xs text-ink-mute">Diagnosticando FFmpeg...</p>}
      {isError && (
        <p className="text-xs text-warn">
          No se pudo diagnosticar la instalación multimedia.
          {error instanceof Error ? ` ${error.message}` : ''}
        </p>
      )}

      {refresh.isError && (
        <p role="alert" className="text-xs text-warn">
          No se pudo actualizar el diagnóstico multimedia. Puedes reintentarlo.
        </p>
      )}

      {data && (
        <>
          <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
            {rows.map((row) => (
              <div
                key={row.label}
                className="flex items-center justify-between gap-2 rounded-lg border border-border bg-base/40 px-3 py-2"
              >
                <span className="truncate text-xs text-ink-soft" title={row.detail}>
                  {row.label}
                  {row.detail ? <span className="ml-1 text-[10px] text-ink-mute">{row.detail}</span> : null}
                </span>
                <span
                  className={`shrink-0 text-[11px] font-semibold ${
                    CAPABILITY_CLASSES[row.status] ?? 'text-ink-mute'
                  }`}
                >
                  {CAPABILITY_LABELS[row.status] ?? row.status}
                </span>
              </div>
            ))}
          </div>
          {data.warnings.length > 0 && (
            <ul className="space-y-1 text-[11px] leading-4 text-ink-mute">
              {data.warnings.map((warning) => (
                <li key={warning}>· {warning}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  )
}
