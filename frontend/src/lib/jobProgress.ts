import type { Job, JobStatus } from './types'

const ACTIVE_STATUS = new Set<JobStatus>([
  'queued',
  'ingesting',
  'transcribing',
  'detecting',
  'rendering',
])

const STATUS_STAGE: Record<JobStatus, string> = {
  queued: 'En cola',
  ingesting: 'Descargando y analizando fuente',
  transcribing: 'Transcribiendo audio',
  detecting: 'Detectando mejores momentos',
  rendering: 'Generando clips',
  done: 'Listo',
  failed: 'Error',
  canceled: 'Cancelado',
}

const STATUS_PROGRESS_FLOOR: Record<JobStatus, number> = {
  queued: 0.02,
  ingesting: 0.05,
  transcribing: 0.1,
  detecting: 0.55,
  rendering: 0.65,
  done: 1,
  failed: 0,
  canceled: 0,
}

export function displayJobStage(job: Job): string {
  return job.stage?.trim() || STATUS_STAGE[job.status]
}

export function displayJobProgress(job: Job): number {
  if (!ACTIVE_STATUS.has(job.status)) return job.progress
  return Math.max(job.progress || 0, STATUS_PROGRESS_FLOOR[job.status])
}
