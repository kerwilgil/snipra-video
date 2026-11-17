export type JobStatus =
  | 'queued'
  | 'ingesting'
  | 'transcribing'
  | 'detecting'
  | 'rendering'
  | 'done'
  | 'failed'
  | 'canceled'

export type JobOutputMode = 'clips' | 'download' | 'audio'

export interface EditorialOptions {
  compact_editing: boolean
  semantic_cards: boolean
  privacy_assist: boolean
  protected_zones: boolean
  overlay_qa: boolean
  transcription_quality: 'default' | 'fast' | 'balanced' | 'precise'
}

export interface Job {
  id: string
  created_at: string
  updated_at: string
  stage_started_at: string | null
  source_type: 'upload' | 'url'
  source_name: string
  title: string
  status: JobStatus
  stage: string
  progress: number
  error: string | null
  duration: number | null
  source_width: number
  source_height: number
  num_clips: number
  formats: string[]
  processing_mode: ProcessingMode
  caption_style: string
  output_mode: JobOutputMode
  captions_enabled: boolean
  download_quality: string
  audio_format: string
  provider: string
  model: string
  detection_source: string
  sponsorblock_enabled: boolean
  sponsorblock_segments: Array<{ start: number; end: number; category: string }>
  channel_label?: string | null
  editorial_options: EditorialOptions
  media_options?: MediaOptions
}

export type ProcessingMode =
  | 'auto_smart'
  | 'podcast_viral'
  | 'vlog'
  | 'gaming'
  | 'gaming_auto'
  | 'gaming_hook'
  | 'gaming_highlights'
  | 'tutorial_screen'
  | 'keep_original'
  | 'music'

export interface ProcessingModeOption {
  value: ProcessingMode
  label: string
  description: string
}

export interface ClipAsset {
  id: string
  format: string
  width: number
  height: number
  duration: number
  size_bytes: number
  video_url: string
  thumb_url: string | null
}

export interface ClipWord {
  word: string
  start: number
  end: number
}

export interface SubtitleBlock {
  start: number
  end: number
  text: string
  highlight_words?: string[]
  words?: ClipWord[]
}

export interface SubtitleStyleConfig {
  style: string
  size: number
  position: string
  text_color: string
  highlight_color: string
  secondary_highlight_color: string
  stroke_color: string
  stroke_width: number
  shadow_enabled: boolean
  background_box: boolean
  uppercase: boolean
  emoji_enabled: boolean
  highlight_max_words: number
}

export interface Clip {
  decision_results?: DecisionResult[]
  id: string
  index: number
  start: number
  end: number
  duration: number
  source_ranges: Array<{
    start: number
    end: number
    kind: string
    score: number
  }>
  title: string
  hook: string
  score: number
  reason: string
  detection_source: string
  hashtags: string[]
  description: string
  editorial_cards: Array<{
    kind: 'contrast' | 'consequence' | 'derived'
    at: number
    duration: number
    kicker: string
    text: string
    secondary: string
    evidence: string
  }>
  privacy_suggestions: Array<{
    text: string
    confidence: number
    format: string
    x: number
    y: number
    width: number
    height: number
    start: number
    end: number
  }>
  privacy_masks: Array<Record<string, unknown>>
  protected_zones: Array<Record<string, unknown>>
  editorial_qa: { passed?: boolean; issues?: string[]; checked_overlays?: number }
  assets: ClipAsset[]
}

export interface ClipOperation {
  id: string
  job_id: string
  clip_id: string
  kind: 'style' | 'subtitles' | 'rewrite_captions' | 'translate_captions'
  status: 'queued' | 'running' | 'done' | 'failed'
  progress: number
  error: string | null
  created_at: string
  updated_at: string
}

export interface ClipTimelineFrame {
  time: number
  filename: string
  url: string
}

export interface ClipTimelineFormat {
  format: string
  mode: string
  width: number
  height: number
  duration: number
  size_bytes: number
  video: string
  thumbnail: string
  quality: {
    passed: boolean
    attempts: number
    issues: string[]
    checks: Record<string, boolean | number | string>
  }
  delivery_qc?: Partial<DeliveryQCResult>
  destination_profile?: string
  frames: ClipTimelineFrame[]
  audio_energy?: Array<{ time: number; rms: number }>
}

export interface ClipTimeline {
  version: number
  clip: {
    title: string
    hook: string
    score: number
    reason: string
    source_start: number
    source_end: number
    duration: number
    trim_offset: number
  }
  processing: {
    mode: string
    requested_reframe: string
    caption_style: string
  }
  formats: ClipTimelineFormat[]
  subtitles: Array<{ start: number; end: number; text: string; word_count: number; words?: ClipWord[] }>
  words: ClipWord[]
  editorial?: {
    cards: Clip['editorial_cards']
    protected_zones: Array<Record<string, unknown>>
    privacy_suggestions: Clip['privacy_suggestions']
    privacy_masks: Array<Record<string, unknown>>
    qa: Clip['editorial_qa']
  }
}

export interface DurationPreset {
  value: string
  label: string
  description: string
  min: number
  max: number
}

export interface MediaOptions {
  destination_profile: string
  color_handling: 'auto' | 'preserve' | 'sdr'
  stabilization: 'off' | 'auto' | 'light' | 'medium' | 'strong'
  video_denoise: 'off' | 'light' | 'medium' | 'strong'
  audio_cleanup: 'off' | 'voice_light' | 'voice' | 'voice_strong'
}

export interface DeliveryQCCheck {
  id: string
  status: 'pass' | 'warn' | 'fail'
  actual: string | number | boolean | null
  expected: string | number | boolean | null
  message: string
}

export interface DeliveryQCResult {
  passed: boolean
  profile: string
  source_hdr_format: string | null
  output_hdr_format: string | null
  checks: DeliveryQCCheck[]
}

export type CapabilityStatus = 'available' | 'missing' | 'unknown' | 'deshake_fallback'

export interface MediaCapabilities {
  ffmpeg_version: string
  ffprobe_version: string
  overall_status: 'ok' | 'degraded' | 'unknown'
  encoders: Record<string, CapabilityStatus>
  filters: Record<string, CapabilityStatus>
  features: Record<string, CapabilityStatus>
  backends: Record<string, string>
  warnings: string[]
}

export interface DestinationProfileItem {
  id: string
  display_name: string
  aspect: string
  frame: string
  max_seconds: number | null
}

export interface AppSettings {
  provider: string
  model: string
  providers: string[]
  default_models: Record<string, string>
  whisper_model: string
  default_num_clips: number
  clip_duration_preset: string
  duration_presets: DurationPreset[]
  min_clip_seconds: number
  max_clip_seconds: number
  enable_media_signals: boolean
  enable_visual_rerank: boolean
  enable_ai_upscale: boolean
  ai_upscale_available: boolean
  vision_provider: string
  vision_model: string
  vision_providers?: string[]
  vision_models?: string[]
  vision_default_models?: Record<string, string>
  vision_models_by_provider?: Record<string, string[]>
  vision_effective?: { provider: string; model: string }
  nvidia_vision_model: string
  nvidia_vision_models?: Array<{ value: string; label: string }>
  download_quality: string
  audio_format: string
  available_audio_formats: string[]
  storage_retention_days: number
  default_formats: string[]
  sponsorblock_categories: string[]
  sponsorblock_categories_available: string[]
  reframe_mode: string
  caption_style: string
  caption_styles: Array<{ value: string; label: string; description: string }>
  highlight_colors: Record<string, string>
  default_style_config: SubtitleStyleConfig
  podcast_viral_config: Record<string, unknown>
  default_for_9_16: Record<string, unknown>
  presets: Record<string, unknown>
  available_formats: string[]
  processing_modes: ProcessingModeOption[]
  yt_dlp_cookies_from_browser: string
  yt_dlp_cookies_path: string
  yt_dlp_cookies_configured: boolean
  extension_token_configured: boolean
  keys_present: Record<string, boolean>
  subscription_available: Record<string, boolean>
  binaries: Record<string, { available: boolean; version: string; path: string }>
  media_defaults?: MediaOptions
  destination_profiles?: DestinationProfileItem[]
}

export interface StorageStatus {
  work_bytes: number
  job_count: number
  orphan_count: number
  work_dir: string
  compositions_bytes: number
  compilation_count: number
  orphan_composition_count: number
  compositions_dir: string
}

export interface JobLog {
  id: string
  created_at: string
  level: 'info' | 'error' | string
  message: string
}

export interface EditorTimelineClip { id: string; asset_id: string; source_in_us: number; source_out_us: number; timeline_start_us: number; timeline_end_us?: number; playback_rate: number; kind?: 'caption' | 'text'; text?: string; position?: { x: number; y: number }; style?: { font_size: number; color: string }; transform?: { x: number; y: number; scale: number; crop?: { x: number; y: number; width: number; height: number } }; audio?: { volume: number; muted: boolean; fade_in_us?: number; fade_out_us?: number }; provenance?: Record<string, unknown> }
export interface EditorTimelineTrack { id: string; type: string; locked: boolean; muted: boolean; solo: boolean; clips: EditorTimelineClip[] }
export interface EditorAsset {
  id: string
  filename: string
  kind: string
  duration_us: number
  width: number
  height: number
  has_audio: boolean
  fingerprint: string
  metadata?: {
    asset_provenance?: AssetProvenanceMetadata
    [key: string]: unknown
  }
}
export interface EditorProject { id: string; name: string; source_job_id: string | null; source_clip_id?: string | null; current_version: number; created_at: string; updated_at: string; timeline: { duration_us: number; tracks: EditorTimelineTrack[] }; assets: EditorAsset[] }
export type EditorProjectListItem = Omit<EditorProject, 'timeline'> & {
  timeline: EditorProject['timeline'] | null
  unavailable_reason?: string
}

export type DecisionTask = 'clip_quality' | 'content_type' | 'hook_detection' |
  'engagement_candidate' | 'recommended_action'

export interface DecisionResult {
  schema_version: '1'
  engine: string
  model: string | null
  task: DecisionTask
  classification: string | null
  probabilities: Record<string, number>
  mutually_exclusive: boolean
  confidence: number | null
  recommended_action: 'keep' | 'trim' | 'merge' | 'discard' | 'review' | null
  evidence_refs: string[]
  duration_ms: number
  status: 'success' | 'disabled' | 'unavailable' | 'timeout' | 'error'
}
export interface EditorPreviewSegment { track_id: string; clip_id: string; asset_id: string; timeline_start_us: number; timeline_end_us: number; source_in_us: number; source_out_us: number; playback_rate: number; transform: { x: number; y: number; scale: number; crop?: { x: number; y: number; width: number; height: number } }; audio: { volume: number; muted: boolean; fade_in_us?: number; fade_out_us?: number }; video_url: string }
export interface EditorPreviewPlan { version: number; duration_us: number; segments: EditorPreviewSegment[]; content?: EditorTimelineClip[]; output_canvas?: { width: number; height: number } }
export interface EditorOperationInput { base_version: number; idempotency_key: string; type: string; parameters: Record<string, unknown>; generated_by?: string }
export interface EditorEvent { id: string; start_us: number; end_us: number; title: string; reason: string; legacy_score: number; source_ranges: Array<Record<string, unknown>>; uncertain: boolean; coherent?: boolean; output_suitable?: boolean; quality_gate?: string }
export interface EditorTranscriptSegment { start_us: number; end_us: number; evidence: { kind: string; content: string } }

export type AssetOrigin = 'SOURCE' | 'ENHANCED' | 'GENERATED'
export type AssetSemanticRole = 'EVIDENTIARY' | 'ILLUSTRATIVE'

export interface AssetProvenanceMetadata {
  origin: AssetOrigin
  semantic_role: AssetSemanticRole
  parent_asset_id: string | null
  root_source_asset_id: string | null
  operation_type: string | null
  operation_version: string | null
  provider: string | null
  model_or_engine: string | null
  request_hash: string | null
  output_hash: string | null
  output_size_bytes: number | null
  created_at: string
  parameters: Record<string, unknown> | null
  source_fingerprint: string | null
  duration_us: number | null
  width: number | null
  height: number | null
  fps: number | null
}

export type EnhancementProfileValue = 'DEFAULT' | 'UPSCALE' | 'CLEANUP' | 'CUSTOM'

export interface EnhancementEstimate {
  cost_class: 'UNKNOWN' | 'LOCAL' | 'FREE_TIER' | 'SUBSCRIPTION' | 'PAID_PER_USE'
  monetary_cost_known: boolean
  monetary_cost_usd: number | null
  estimated_seconds: number | null
  notes: string
}

export type AssetTaskStatus = 'ESTIMATING' | 'AWAITING_APPROVAL' | 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED'
export type AssetTaskApprovalState = 'PENDING' | 'APPROVED' | 'REJECTED'

export interface EnhancementTask {
  id: string
  project_id: string
  kind: string
  status: AssetTaskStatus
  provider: string
  model: string
  request: Record<string, unknown>
  request_hash: string
  cost_estimate: EnhancementEstimate
  approval_state: AssetTaskApprovalState
  approved_request_hash: string | null
  progress: number
  output_asset_id: string | null
  error: string | null
  created_at: string
  updated_at: string
  output_asset?: EditorAsset
}

export interface GenerationProviderCapabilities {
  provider: string
  model: string
  supports_text_to_video: boolean
  supports_image_to_video: boolean
  supports_async: boolean
  supports_polling: boolean
  supports_local_cancel: boolean
  supports_remote_cancel: boolean
  local: boolean
  paid: boolean
  billable: boolean
  requires_api_key: boolean
  enabled_by_default: boolean
  max_duration_seconds: number
  supported_resolutions: string[]
  supported_aspect_ratios: string[]
}

export interface GenerationCapability {
  local_ltx: {
    provider: string
    model: string
    hardware_supported: boolean
    hardware_message: string
    hardware_reason: string
    runtime_ready: boolean
    runtime_message: string
    physical_validation: string
    paid: boolean
    billable: boolean
    requires_api_key: boolean
    enabled_by_default: boolean
  }
  cloud_providers: Array<{
    provider: string
    model: string
    supported: boolean
    available: boolean
    ready: boolean
    physical_validation: string
    message: string
    paid: boolean
    billable: boolean
    requires_api_key: boolean
    enabled_by_default: boolean
  }>
}

export type GenerationMode = 'text_to_video' | 'image_to_video'
export type GenerationProvider = 'local_ltx' | 'runway' | 'luma'
export type GenerationModel = 'ltx-video-2b-distilled-fp8' | 'gen4.5' | 'ray-3.2'
export type GenerationAspectRatio = '16:9' | '9:16' | '1:1' | '4:3' | '3:4'
export type GenerationResolution = 'SD' | 'HD'

export interface GenerationRequest {
  mode: GenerationMode
  provider: GenerationProvider
  model: GenerationModel
  prompt_intent: string
  visual_direction?: string
  camera_direction?: string
  avoid_constraints?: string
  duration_seconds: number
  aspect_ratio: GenerationAspectRatio
  resolution: GenerationResolution
  reference_asset_ids: string[]
  grounding_asset_ids: string[]
  seed?: number
  generation_parameters: Record<string, unknown>
}

export interface GenerationEstimate {
  provider: GenerationProvider
  model: GenerationModel
  request_hash: string
  api_cost_usd: number
  currency: string
  pricing_basis?: string
  pricing_version?: string
  pricing_checked_at?: string
  estimated_compute_class?: string
  estimated_vram_class?: string
  estimated_duration_seconds?: number
  billable: boolean
  requires_monetary_approval: boolean
  estimate_version: string
  is_exact: boolean
}

export type GenerationTaskStatus = AssetTaskStatus
export type GenerationTaskApprovalState = AssetTaskApprovalState

export interface GenerationTask {
  id: string
  project_id: string
  kind: string
  status: GenerationTaskStatus
  provider: string
  model: string
  request: GenerationRequest
  request_hash: string
  cost_estimate: GenerationEstimate
  approval_state: GenerationTaskApprovalState
  approved_request_hash: string | null
  progress: number
  output_asset_id: string | null
  error: string | null
  created_at: string
  updated_at: string
  output_asset?: EditorAsset
  external_task_id?: string
  external_task_state?: string
  external_task_metadata?: Record<string, unknown>
  cancel_requested?: boolean
  next_poll_at?: string
  submit_count?: number
}

export interface HighlightScoreComponent { name: string; raw_value: number; weight: number; contribution: number; evidence_refs: string[]; note: string }
export interface HighlightTopicContext { topic_id: string; topic_start: number; topic_end: number; overlap_ratio: number; confidence: number; keywords: string[]; representative_text: string; evidence_refs: string[] }
export interface HighlightEvent {
  event_id: string; rank: number; title: string; hook: string; reason: string; overall_score: number;
  original_start: number; original_end: number; refined_start: number; refined_end: number;
  refinement_status: string; refinement_confidence: number; peak_at: number | null; payoff_at: number | null;
  payoff_window_start: number | null; payoff_window_end: number | null; reaction_at: number | null;
  reaction_window_start: number | null; reaction_window_end: number | null; primary_topic_id: string | null;
  topic_contexts: HighlightTopicContext[]; score_components: HighlightScoreComponent[]; evidence_refs: string[];
}
export interface HighlightClusterRelation { event_id: string; relation: string; similarity?: number; reason?: string }
export interface HighlightCluster { cluster_id: string; representative_event_id: string; member_event_ids: string[]; relations: HighlightClusterRelation[]; topic_ids: string[]; refined_start: number; refined_end: number }
export interface EnhancedHighlightEvidence { contract_version: string; asset_id: string; source_event_count: number; events: HighlightEvent[]; clusters: HighlightCluster[]; diverse_order_event_ids: string[] }
export interface ShortPlanResponse { short_id: string; event_id: string; source_asset_id: string; original_start: number; original_end: number; refined_start: number; refined_end: number; title: string; hook: string; reason: string; refinement_status: string; refinement_confidence: number; peak_at: number | null; payoff_at: number | null; payoff_window_start: number | null; payoff_window_end: number | null; reaction_at: number | null; reaction_window_start: number | null; reaction_window_end: number | null; primary_topic_id: string | null; topic_contexts: HighlightTopicContext[]; score_components: HighlightScoreComponent[]; evidence_refs: string[]; director_reason: string; editorial_intent: string }
export interface ShortBatchPlanResponse { batch_id: string; project_id: string; asset_id: string | null; requested_count: number; selected_count: number; status: 'planned' | 'partial' | 'failed' | 'needs_user'; shorts: ShortPlanResponse[]; enhanced_evidence: EnhancedHighlightEvidence; director_decision: Record<string, unknown>; created_at: string; created_by?: string }

export type CompilationStatus = 'queued' | 'rendering' | 'done' | 'failed'

export type CompilationTransition = 'none' | 'fade'

/** Un asset ya renderizado, disponible para componer. La unidad componible es el
 *  asset (no el clip): un mismo clip puede existir en 9x16 y 16x9. */
export interface ComposableClip {
  asset_id: string
  clip_id: string
  job_id: string
  job_title: string
  clip_index: number
  title: string
  hook: string
  score: number
  format: string
  duration: number
  width: number
  height: number
  size_bytes: number
  video_url: string
  thumb_url: string | null
}

export interface Compilation {
  id: string
  created_at: string
  updated_at: string
  title: string
  format: string
  asset_ids: string[]
  clip_count: number
  transition: CompilationTransition
  transition_seconds: number
  bgm_path: string
  bgm_volume: number
  bgm_ducking: boolean
  bgm_auto_level: boolean
  bgm_duck_lu: number
  audio_report: Record<string, string | number>
  status: CompilationStatus
  progress: number
  error: string | null
  duration: number
  width: number
  height: number
  size_bytes: number
}

export interface TrackedChannel {
  id: string
  created_at: string
  url: string
  label: string
  enabled: boolean
  poll_interval_minutes: number
  last_checked_at: string | null
  last_check_status: 'healthy' | 'warning' | 'error' | 'pending' | string
  last_check_error: string | null
  last_success_at: string | null
  consecutive_failures: number
  videos_seen_last_check: number
  new_videos_last_check: number
  canonical_url?: string | null
  channel_title?: string | null
  num_clips: number
  formats: string[]
  processing_mode: ProcessingMode
  caption_style: string
  output_mode: JobOutputMode
  captions_enabled: boolean
  download_quality: string
  provider: string
  model: string
  sponsorblock_enabled: boolean
  jobs_created: number
}
