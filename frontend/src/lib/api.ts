import type {
  AppSettings,
  Clip,
  ClipOperation,
  ClipTimeline,
  ClipWord,
  Compilation,
  CompilationTransition,
  ComposableClip,
  EnhancementProfileValue,
  EnhancementTask,
  Job,
  JobLog,
  JobOutputMode,
  MediaCapabilities,
  MediaOptions,
  ProcessingMode,
  SubtitleBlock,
  SubtitleStyleConfig,
  StorageStatus,
  TrackedChannel,
  EditorialOptions,
  EditorOperationInput,
  EditorEvent,
  EditorPreviewPlan,
  EditorTranscriptSegment,
  EditorProject,
  EditorProjectListItem,
  ShortBatchPlanResponse,
  GenerationTask,
  GenerationCapability,
  GenerationRequest,
} from './types'

const BASE = '/api'

export interface NvidiaCatalog {
  models: Array<{ id: string; text: boolean; vision: boolean; compatibility_source: string }>
  available: boolean
  stale: boolean
  warning: string | null
  fetched_at: number | null
  expires_at: number | null
}

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  const contentType = res.headers.get('content-type') ?? ''
  const readDetail = async () => {
    const body = await res.text()
    if (contentType.includes('application/json')) {
      try {
        const json = JSON.parse(body) as { detail?: string | { message?: string } }
        return typeof json.detail === 'string' ? json.detail : json.detail?.message
      } catch {
        return undefined
      }
    }
    return body.trim().startsWith('<') ? undefined : body.slice(0, 240)
  }
  if (!res.ok) {
    const detail = await readDetail()
    throw new Error(detail || `La API respondió ${res.status} ${res.statusText}`.trim())
  }
  if (!contentType.includes('application/json')) {
    await readDetail()
    throw new Error(`La API devolvió una respuesta no JSON para ${url}`)
  }
  return res.json() as Promise<T>
}

async function waitForClipOperation(operation: ClipOperation): Promise<ClipOperation> {
  const deadline = Date.now() + 60 * 60 * 1000
  let current = operation
  while (current.status === 'queued' || current.status === 'running') {
    if (Date.now() >= deadline) throw new Error('La operacion masiva excedio el tiempo limite')
    await new Promise((resolve) => window.setTimeout(resolve, 1500))
    current = await jsonFetch<ClipOperation>(`${BASE}/clip-operations/${operation.id}`)
  }
  if (current.status === 'failed') {
    throw new Error(current.error || 'La operacion masiva fallo')
  }
  return current
}

async function waitIfQueued(result: Clip | ClipOperation): Promise<Clip | ClipOperation> {
  return 'kind' in result ? waitForClipOperation(result) : result
}

export interface CreateJobOptions {
  num_clips?: number
  formats?: string[]
  processing_mode?: ProcessingMode
  caption_style?: string
  style_config?: Record<string, unknown>
  provider?: string
  model?: string
  output_mode?: JobOutputMode
  captions_enabled?: boolean
  download_quality?: string
  audio_format?: string
  sponsorblock_enabled?: boolean
  editorial_options?: Partial<EditorialOptions>
  media_options?: Partial<MediaOptions>
}

export interface ChannelInput {
  url?: string
  label?: string
  enabled?: boolean
  poll_interval_minutes?: number
  num_clips?: number
  formats?: string[]
  processing_mode?: ProcessingMode
  provider?: string
  model?: string
  caption_style?: string
  output_mode?: JobOutputMode
  captions_enabled?: boolean
  download_quality?: string
  audio_format?: string
  sponsorblock_enabled?: boolean
}

export interface SettingsUpdate {
  provider?: string
  model?: string
  whisper_model?: string
  default_num_clips?: number
  clip_duration_preset?: string
  min_clip_seconds?: number
  max_clip_seconds?: number
  enable_media_signals?: boolean
  enable_visual_rerank?: boolean
  enable_ai_upscale?: boolean
  vision_provider?: string
  vision_model?: string
  nvidia_vision_model?: string
  download_quality?: string
  audio_format?: string
  storage_retention_days?: number
  default_formats?: string[]
  sponsorblock_categories?: string[]
  reframe_mode?: string
  yt_dlp_cookies_from_browser?: string
  yt_dlp_cookies_path?: string
  caption_style?: string
  destination_profile?: string
  color_handling?: MediaOptions['color_handling']
  stabilization?: MediaOptions['stabilization']
  video_denoise?: MediaOptions['video_denoise']
  audio_cleanup?: MediaOptions['audio_cleanup']
}

export interface ApiKeysUpdate {
  claude?: string
  openai?: string
  nvidia?: string
}

export interface ClipUpdate {
  start?: number
  end?: number
  title?: string
  hook?: string
}

export interface CompilationInput {
  asset_ids: string[]
  title?: string
  transition?: CompilationTransition
  transition_seconds?: number
  bgm_path?: string
  bgm_volume?: number
  bgm_ducking?: boolean
  bgm_auto_level?: boolean
  bgm_duck_lu?: number
}

export const api = {
  getNvidiaModels: (refresh = false) => jsonFetch<NvidiaCatalog>(`${BASE}/settings/nvidia-models${refresh ? '?refresh=true' : ''}`),
  listJobs: () => jsonFetch<Job[]>(`${BASE}/jobs`),

  listEditorProjects: () => jsonFetch<EditorProjectListItem[]>(`${BASE}/editor/projects`),
  getEditorProject: (id: string) => jsonFetch<EditorProject>(`${BASE}/editor/projects/${id}`),
  createEditorProject: (name: string) => jsonFetch<EditorProject>(`${BASE}/editor/projects`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) }),
  renameEditorProject: (id: string, name: string) => jsonFetch<EditorProject>(`${BASE}/editor/projects/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) }),
  deleteEditorProject: (id: string) => jsonFetch<{ ok: true }>(`${BASE}/editor/projects/${id}`, { method: 'DELETE' }),
  openAutoResultInEditor: (clipId: string) => jsonFetch<EditorProject & { created: boolean }>(`${BASE}/editor/projects/from-auto/${clipId}`, { method: 'POST' }),
  uploadEditorAsset: (projectId: string, file: File) => { const form = new FormData(); form.append('file', file); return jsonFetch(`${BASE}/editor/projects/${projectId}/assets/upload`, { method: 'POST', body: form }) },
  removeEditorAsset: (projectId: string, assetId: string) => jsonFetch<{ ok: true }>(`${BASE}/editor/projects/${projectId}/assets/${assetId}`, { method: 'DELETE' }),
  editorPreviewPlan: (projectId: string) => jsonFetch<EditorPreviewPlan>(`${BASE}/editor/projects/${projectId}/preview-plan`),
  editorEvents: (projectId: string, intent?: string) => jsonFetch<{ events: EditorEvent[]; best_event_pool: EditorEvent[] }>(`${BASE}/editor/projects/${projectId}/events${intent ? `?intent=${intent}` : ''}`),
  createShortBatch: (projectId: string, body: { requested_count: number; intent: 'highlight' }) => jsonFetch<ShortBatchPlanResponse>(`${BASE}/editor/projects/${projectId}/shorts`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  editorTranscript: (projectId: string, startUs: number, endUs: number) => jsonFetch<{ segments: EditorTranscriptSegment[] }>(`${BASE}/editor/projects/${projectId}/transcript?start_us=${startUs}&end_us=${endUs}`),
  editorAgent: (projectId: string, body: { request: string; intent?: string; execute?: boolean; idempotency_key?: string }) => jsonFetch<{ status: string; state?: string; version?: number; run_id?: string; plan?: { objective: string; output_intent: string; operations: Array<{ type: string; reason?: string }> }; report?: { changes: number; duration_us: number; verification: string; undo_available: boolean } }>(`${BASE}/editor/projects/${projectId}/agent`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),

  editorAgentConfirm: (projectId: string, runId: string) => jsonFetch<{ status: string; state?: string; version?: number; run_id?: string; plan?: { objective: string; output_intent: string; operations: Array<{ type: string; reason?: string }> }; report?: { changes: number; duration_us: number; verification: string; undo_available: boolean } }>(`${BASE}/editor/projects/${projectId}/agent/${runId}/confirm`, { method: 'POST', headers: { 'Content-Type': 'application/json' } }),
  editorOperation: (projectId: string, body: EditorOperationInput) => jsonFetch<{ version: number; timeline: EditorProject['timeline'] }>(`${BASE}/editor/projects/${projectId}/operations`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  editorUndo: (projectId: string) => jsonFetch<{ version: number; timeline: EditorProject['timeline'] }>(`${BASE}/editor/projects/${projectId}/undo`, { method: 'POST' }),

  getJob: (id: string) => jsonFetch<Job>(`${BASE}/jobs/${id}`),

  deleteJob: async (id: string) => {
    const res = await fetch(`${BASE}/jobs/${id}`, { method: 'DELETE' })
    if (!res.ok) throw new Error('No se pudo borrar el trabajo')
  },

  retryJob: (id: string) =>
    jsonFetch<Job>(`${BASE}/jobs/${id}/retry`, {
      method: 'POST',
    }),

  cancelJob: (id: string) =>
    jsonFetch<Job>(`${BASE}/jobs/${id}/cancel`, {
      method: 'POST',
    }),

  getJobLogs: (jobId: string) => jsonFetch<JobLog[]>(`${BASE}/jobs/${jobId}/logs`),

  getExportManifest: (jobId: string) => jsonFetch<Record<string, unknown>>(`${BASE}/jobs/${jobId}/export-manifest`),

  downloadOriginalUrl: (jobId: string) => `${BASE}/jobs/${jobId}/download`,

  getClips: (jobId: string) => jsonFetch<Clip[]>(`${BASE}/jobs/${jobId}/clips`),

  getSettings: () => jsonFetch<AppSettings>(`${BASE}/settings`),

  getMediaCapabilities: (refresh = false) =>
    jsonFetch<MediaCapabilities>(`${BASE}/system/media-capabilities${refresh ? '?refresh=true' : ''}`),

  getViaAvailability: () => jsonFetch<{ available: boolean; code: string | null; reason: string | null }>(`${BASE}/settings/via-availability`),

  getStorage: () => jsonFetch<StorageStatus>(`${BASE}/storage`),

  cleanupStorage: () =>
    jsonFetch<{ removed_paths: number; removed_bytes: number; status: StorageStatus }>(`${BASE}/storage/cleanup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ remove_orphans: true }),
    }),

  updateSettings: (body: SettingsUpdate) =>
    jsonFetch<AppSettings>(`${BASE}/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  updateApiKeys: (body: ApiKeysUpdate) =>
    jsonFetch<AppSettings>(`${BASE}/settings/api-keys`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  // unico endpoint que devuelve el valor crudo del token, y solo una vez
  regenerateExtensionToken: () =>
    jsonFetch<AppSettings & { extension_token: string }>(
      `${BASE}/settings/extension-token/regenerate`,
      { method: 'POST' },
    ),

  // Emparejamiento de un clic: la extension crea el request_id (sin pasar por
  // aqui), esta llamada la hace la pestaña /pair tras el clic humano de "Aprobar".
  approveExtensionPairing: (requestId: string) =>
    jsonFetch<{ ok: true }>(`${BASE}/extension/pair/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_id: requestId }),
    }),

  updateClip: (clipId: string, body: ClipUpdate) =>
    jsonFetch<Clip>(`${BASE}/clips/${clipId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  rerenderClip: (clipId: string) =>
    jsonFetch<Clip>(`${BASE}/clips/${clipId}/render`, {
      method: 'POST',
    }),

  editClipStyle: async (clipId: string, body: { style?: string; style_config?: Partial<SubtitleStyleConfig>; apply_to_all?: boolean }) =>
    waitIfQueued(await jsonFetch<Clip | ClipOperation>(`${BASE}/clips/${clipId}/style`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })),

  editClipSubtitles: async (clipId: string, body: { subtitles: SubtitleBlock[]; style_config?: Partial<SubtitleStyleConfig>; apply_to_all?: boolean }) =>
    waitIfQueued(await jsonFetch<Clip | ClipOperation>(`${BASE}/clips/${clipId}/subtitles`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })),

  rewriteClipCaptions: async (clipId: string, body: { apply_to_all?: boolean } = {}) =>
    waitIfQueued(await jsonFetch<Clip | ClipOperation>(`${BASE}/clips/${clipId}/captions/rewrite`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })),

  translateClipCaptions: async (
    clipId: string,
    body: { target_language: string; apply_to_all?: boolean; provider?: string; model?: string },
  ) =>
    waitIfQueued(await jsonFetch<Clip | ClipOperation>(`${BASE}/clips/${clipId}/captions/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })),

  reeditClip: (
    clipId: string,
    body: {
      remove?: number[][]
      remove_fillers?: boolean
      remove_repeated_takes?: boolean
      remove_long_pauses?: boolean
    },
  ) =>
    jsonFetch<Clip>(`${BASE}/clips/${clipId}/reedit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  getClipTranscript: (clipId: string) =>
    jsonFetch<{ words: ClipWord[]; has_words: boolean }>(`${BASE}/clips/${clipId}/transcript`),

  getClipTimeline: (clipId: string) => jsonFetch<ClipTimeline>(`${BASE}/clips/${clipId}/timeline`),

  applyClipPrivacy: (clipId: string, suggestionIndices?: number[]) =>
    jsonFetch<Clip>(`${BASE}/clips/${clipId}/privacy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ suggestion_indices: suggestionIndices }),
    }),

  createFromUrl: (url: string, opts: CreateJobOptions = {}) =>
    jsonFetch<{ id: string; status: string }>(`${BASE}/jobs/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, ...opts }),
    }),

  createFromFile: (file: File, opts: CreateJobOptions = {}) => {
    const fd = new FormData()
    fd.append('file', file)
    if (opts.num_clips != null) fd.append('num_clips', String(opts.num_clips))
    if (opts.formats) fd.append('formats', opts.formats.join(','))
    if (opts.processing_mode) fd.append('processing_mode', opts.processing_mode)
    if (opts.caption_style) fd.append('caption_style', opts.caption_style)
    if (opts.style_config) fd.append('style_config', JSON.stringify(opts.style_config))
    if (opts.provider) fd.append('provider', opts.provider)
    if (opts.model) fd.append('model', opts.model)
    if (opts.output_mode) fd.append('output_mode', opts.output_mode)
    if (opts.captions_enabled != null) fd.append('captions_enabled', String(opts.captions_enabled))
    if (opts.download_quality) fd.append('download_quality', opts.download_quality)
    if (opts.audio_format) fd.append('audio_format', opts.audio_format)
    if (opts.sponsorblock_enabled != null) fd.append('sponsorblock_enabled', String(opts.sponsorblock_enabled))
    if (opts.editorial_options) fd.append('editorial_options', JSON.stringify(opts.editorial_options))
    if (opts.media_options) fd.append('media_options', JSON.stringify(opts.media_options))
    return jsonFetch<{ id: string; status: string }>(`${BASE}/jobs`, {
      method: 'POST',
      body: fd,
    })
  },

  listComposableClips: () => jsonFetch<ComposableClip[]>(`${BASE}/composable-clips`),

  deleteAsset: (assetId: string) =>
    jsonFetch<{ ok: true; clip_removed: boolean; format: string }>(`${BASE}/assets/${assetId}`, {
      method: 'DELETE',
    }),

  listCompositions: () => jsonFetch<Compilation[]>(`${BASE}/compositions`),

  createCompilation: (body: CompilationInput) =>
    jsonFetch<Compilation>(`${BASE}/compositions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  retryCompilation: (id: string) =>
    jsonFetch<Compilation>(`${BASE}/compositions/${id}/retry`, { method: 'POST' }),

  deleteCompilation: async (id: string) => {
    const res = await fetch(`${BASE}/compositions/${id}`, { method: 'DELETE' })
    if (!res.ok) throw new Error('No se pudo borrar la compilacion')
  },

  compilationVideoUrl: (id: string) => `${BASE}/compositions/${id}/video`,

  compilationThumbUrl: (id: string) => `${BASE}/compositions/${id}/thumb`,

  listChannels: () => jsonFetch<TrackedChannel[]>(`${BASE}/channels`),

  createChannel: (body: ChannelInput) =>
    jsonFetch<TrackedChannel>(`${BASE}/channels`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  updateChannel: (id: string, body: ChannelInput) =>
    jsonFetch<TrackedChannel>(`${BASE}/channels/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  deleteChannel: async (id: string) => {
    const res = await fetch(`${BASE}/channels/${id}`, { method: 'DELETE' })
    if (!res.ok) throw new Error('No se pudo borrar el canal')
  },

  checkChannelNow: (id: string) =>
    jsonFetch<TrackedChannel & { check_result: Record<string, unknown> }>(
      `${BASE}/channels/${id}/check`,
      { method: 'POST' },
    ),

  // Enhancement API (SNIPRA 1.6C Block 04)
  prepareAssetEnhancement: (
    projectId: string,
    assetId: string,
    body: {
      profile: EnhancementProfileValue
      target_width?: number
      target_height?: number
      upscale_factor?: number
      sharpen_strength?: number
      contrast?: number
      brightness?: number
      saturation?: number
      denoise_strength?: number
      preserve_audio?: boolean
    }
  ) =>
    jsonFetch<EnhancementTask>(`${BASE}/editor/projects/${projectId}/assets/${assetId}/enhancements`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  getEnhancementTask: (projectId: string, taskId: string) =>
    jsonFetch<EnhancementTask>(`${BASE}/editor/projects/${projectId}/enhancements/${taskId}`),

  listEnhancementTasks: (projectId: string) =>
    jsonFetch<EnhancementTask[]>(`${BASE}/editor/projects/${projectId}/enhancements`),

  approveEnhancementTask: (projectId: string, taskId: string, requestHash: string) =>
    jsonFetch<EnhancementTask>(`${BASE}/editor/projects/${projectId}/enhancements/${taskId}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_hash: requestHash }),
    }),

  rejectEnhancementTask: (projectId: string, taskId: string) =>
    jsonFetch<EnhancementTask>(`${BASE}/editor/projects/${projectId}/enhancements/${taskId}/reject`, {
      method: 'POST',
    }),

  cancelEnhancementTask: (projectId: string, taskId: string) =>
    jsonFetch<EnhancementTask>(`${BASE}/editor/projects/${projectId}/enhancements/${taskId}/cancel`, {
      method: 'POST',
    }),

  // Generation API (SNIPRA 1.6D Block 04)
  getGenerationCapability: (projectId: string) =>
    jsonFetch<GenerationCapability>(`${BASE}/editor/projects/${projectId}/generations/capability`),

  prepareGeneration: (projectId: string, body: GenerationRequest) =>
    jsonFetch<GenerationTask>(`${BASE}/editor/projects/${projectId}/generations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  getGenerationTask: (projectId: string, taskId: string) =>
    jsonFetch<GenerationTask>(`${BASE}/editor/projects/${projectId}/generations/${taskId}`),

  listGenerationTasks: (projectId: string) =>
    jsonFetch<GenerationTask[]>(`${BASE}/editor/projects/${projectId}/generations`),

  approveGenerationTask: (projectId: string, taskId: string, requestHash: string) =>
    jsonFetch<GenerationTask>(`${BASE}/editor/projects/${projectId}/generations/${taskId}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_hash: requestHash }),
    }),

  rejectGenerationTask: (projectId: string, taskId: string) =>
    jsonFetch<GenerationTask>(`${BASE}/editor/projects/${projectId}/generations/${taskId}/reject`, {
      method: 'POST',
    }),

  cancelGenerationTask: (projectId: string, taskId: string) =>
    jsonFetch<GenerationTask>(`${BASE}/editor/projects/${projectId}/generations/${taskId}/cancel`, {
      method: 'POST',
    }),

  // Abortable wait helper - ensures listener cleanup on both resolve and reject
  waitWithAbort: async (ms: number, signal?: AbortSignal): Promise<void> => {
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timer)
        signal?.removeEventListener('abort', onAbort)
      }

      const onAbort = () => {
        cleanup()
        reject(new Error('Polling abortado'))
      }

      if (signal?.aborted) {
        reject(new Error('Polling abortado'))
        return
      }

      const timer = setTimeout(() => {
        cleanup()
        resolve()
      }, ms)

      if (signal) {
        signal.addEventListener('abort', onAbort, { once: true })
      }
    })
  },

  // Polling helper for enhancement tasks
  waitForEnhancementTask: async (
    projectId: string,
    taskId: string,
    options?: { intervalMs?: number; timeoutMs?: number; onProgress?: (task: EnhancementTask) => void; signal?: AbortSignal }
  ): Promise<EnhancementTask> => {
    const interval = options?.intervalMs ?? 1200
    const timeout = options?.timeoutMs ?? 60 * 60 * 1000 // 1 hour
    const deadline = Date.now() + timeout

    // Check abort signal BEFORE first fetch - guarantee NO FETCH AFTER ABORT
    if (options?.signal?.aborted) {
      throw new Error('Polling abortado')
    }

    let current = await jsonFetch<EnhancementTask>(`${BASE}/editor/projects/${projectId}/enhancements/${taskId}`, { signal: options?.signal })
    if (options?.onProgress) options.onProgress(current)
    while (current.status === 'QUEUED' || current.status === 'RUNNING') {
      if (Date.now() >= deadline) throw new Error('La tarea de mejora excedió el tiempo límite')
      await api.waitWithAbort(interval, options?.signal)
      current = await jsonFetch<EnhancementTask>(`${BASE}/editor/projects/${projectId}/enhancements/${taskId}`, { signal: options?.signal })
      if (options?.onProgress) options.onProgress(current)
    }
    if (current.status === 'FAILED') {
      throw new Error(current.error || 'La tarea de mejora falló')
    }
    return current
  },
}
