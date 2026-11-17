import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ConfirmDialog } from '../components/ConfirmDialog'
import { IconClips, IconPlus, IconTrash } from '../components/icons'
import { TopbarActions, TopbarSearch } from '../components/topbar'
import { Select } from '../components/Select'
import { ProgressBar, StatusBadge } from '../components/ui'
import { api } from '../lib/api'
import { processingModeLabel } from '../lib/gaming'
import { formatDateTime, formatDuration, formatResolution } from '../lib/format'
import { displayJobProgress, displayJobStage } from '../lib/jobProgress'
import type { Job, JobStatus } from '../lib/types'

const ACTIVE = new Set<JobStatus>(['queued', 'ingesting', 'transcribing', 'detecting', 'rendering'])
const SORT_OPTIONS = [
  { value: 'newest', label: 'Más recientes' },
  { value: 'oldest', label: 'Más antiguos' },
] as const
const FILTERS = [
  { id: 'all', label: 'Todos' },
  { id: 'active', label: 'En proceso' },
  { id: 'done', label: 'Completados' },
  { id: 'issues', label: 'Con errores' },
] as const
type Filter = (typeof FILTERS)[number]['id']
type SortOrder = (typeof SORT_OPTIONS)[number]['value']

export function JobsPage() {
  const qc = useQueryClient()
  const { data: jobs, isLoading, isError, isFetching, refetch } = useQuery({
    queryKey: ['jobs'],
    queryFn: api.listJobs,
    refetchInterval: 4000,
  })
  const del = useMutation({
    mutationFn: api.deleteJob,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['jobs'] }),
  })

  const [pendingDelete, setPendingDelete] = useState<{ id: string; name: string } | null>(null)
  const [filter, setFilter] = useState<Filter>('all')
  const [sortOrder, setSortOrder] = useState<SortOrder>('newest')
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const filteredJobs = useMemo(() => {
    const query = search.trim().toLocaleLowerCase()
    const visible = (jobs ?? []).filter((job) => {
      const matchesFilter = filter === 'all'
        || (filter === 'active' && ACTIVE.has(job.status))
        || (filter === 'done' && job.status === 'done')
        || (filter === 'issues' && (job.status === 'failed' || job.status === 'canceled'))
      const matchesSearch = !query || `${job.title} ${job.source_name}`.toLocaleLowerCase().includes(query)
      return matchesFilter && matchesSearch
    })
    const direction = sortOrder === 'newest' ? -1 : 1
    return visible.sort((left, right) => direction * (Date.parse(left.created_at) - Date.parse(right.created_at)))
  }, [filter, jobs, search, sortOrder])

  const selectedJob = filteredJobs.find((job) => job.id === selectedId) ?? filteredJobs[0] ?? null
  const { data: clips } = useQuery({
    queryKey: ['clips', selectedJob?.id],
    queryFn: () => api.getClips(selectedJob!.id),
    enabled: selectedJob?.status === 'done',
  })
  const preview = clips?.flatMap((clip) => clip.assets).find((asset) => asset.thumb_url) ?? null

  return (
    <div className="projects-page">
      <TopbarSearch>
        <label className="project-search">
          <SearchGlyph />
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Buscar proyectos y videos…" aria-label="Buscar proyectos y videos" />
        </label>
      </TopbarSearch>
      <TopbarActions>
        <Link to="/" className="btn-primary project-create"><IconPlus /> Nuevo trabajo</Link>
      </TopbarActions>

      <div className="projects-workspace">
        <section className="projects-list-area" aria-labelledby="projects-heading">
          <header className="projects-heading-row">
            <div>
              <h1 id="projects-heading">Proyectos</h1>
              <p>Gestiona tus trabajos y clips en un solo lugar.</p>
            </div>
          </header>

          <div className="projects-toolbar">
            <div className="project-filter-tabs" role="tablist" aria-label="Filtrar proyectos">
              {FILTERS.map((item) => (
                <button key={item.id} type="button" role="tab" aria-selected={filter === item.id} className={filter === item.id ? 'is-active' : ''} onClick={() => setFilter(item.id)}>
                  {item.label}
                </button>
              ))}
            </div>
            <label className="project-sort-label">
              <span className="sr-only">Ordenar proyectos</span>
              <Select ariaLabel="Ordenar proyectos" value={sortOrder} onChange={(value) => setSortOrder(value as SortOrder)} options={[...SORT_OPTIONS]} />
            </label>
          </div>

          {del.isError && <div role="alert" className="project-error">No se pudo borrar el trabajo. Inténtalo de nuevo.</div>}
          {isLoading && <p className="project-empty">Cargando proyectos…</p>}
          {isError && <div role="alert" className="page-error"><span>No se pudieron cargar los trabajos. Comprueba que Snipra esté en ejecución e inténtalo de nuevo.</span><button type="button" className="btn-secondary" disabled={isFetching} onClick={() => void refetch()}>{isFetching ? 'Reintentando...' : 'Reintentar'}</button></div>}
          {!isLoading && !isError && filteredJobs.length === 0 && (
            <div className="project-empty-state">
              <div className="project-empty-icon"><IconClips /></div>
              <strong>{jobs?.length ? 'No hay resultados' : 'Todavía no hay trabajos'}</strong>
              <span>{jobs?.length ? 'Prueba con otro filtro o término de búsqueda.' : 'Crea un trabajo para empezar a generar clips.'}</span>
              {!jobs?.length && <Link to="/" className="btn-primary">Crear trabajo</Link>}
            </div>
          )}

          <div className="project-cards">
            {filteredJobs.map((job) => (
              <ProjectCard key={job.id} job={job} selected={selectedJob?.id === job.id} previewUrl={selectedJob?.id === job.id ? preview?.thumb_url ?? null : null} onSelect={() => setSelectedId(job.id)} onDelete={() => setPendingDelete({ id: job.id, name: job.title || job.source_name })} deleting={del.isPending} />
            ))}
          </div>
        </section>

        <aside className="project-inspector" aria-label="Detalle del proyecto seleccionado">
          {selectedJob ? <ProjectInspector job={selectedJob} previewUrl={preview?.thumb_url ?? null} /> : (
            <div className="inspector-placeholder"><IconClips /><span>{isError ? 'Los detalles aparecerán cuando se carguen los trabajos.' : 'Selecciona un proyecto para ver sus detalles.'}</span></div>
          )}
        </aside>
      </div>

      <ConfirmDialog
        open={pendingDelete !== null}
        title="Borrar trabajo"
        message={`¿Borrar «${pendingDelete?.name}»? También se eliminarán sus clips.`}
        confirmLabel="Borrar"
        onConfirm={() => { if (pendingDelete) del.mutate(pendingDelete.id); setPendingDelete(null) }}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  )
}

function ProjectCard({ job, selected, previewUrl, onSelect, onDelete, deleting }: { job: Job; selected: boolean; previewUrl: string | null; onSelect: () => void; onDelete: () => void; deleting: boolean }) {
  const active = ACTIVE.has(job.status)
  const progress = displayJobProgress(job)
  const pct = Math.round(Math.min(1, Math.max(0, progress)) * 100)
  return (
    <article className={`project-card ${selected ? 'is-selected' : ''}`}>
      <button type="button" className="project-card-main" onClick={onSelect} aria-pressed={selected}>
        {previewUrl ? <img className="project-thumbnail" src={previewUrl} alt="" /> : <span className="project-thumbnail project-thumbnail-empty"><IconClips />{job.duration ? <span>{formatDuration(job.duration)}</span> : null}</span>}
        <span className="project-card-info">
          <span className="project-card-title">{job.title || job.source_name}</span>
          <span className="project-meta-line"><span>{job.channel_label || (job.source_type === 'url' ? 'Enlace' : 'Archivo local')}</span><i />{formatDateTime(job.created_at)}<i />{formatDuration(job.duration)}</span>
          {job.status === 'failed' ? <span className="project-card-description project-card-error">{job.error || 'El trabajo encontró un error.'}</span> : <span className="project-card-description">{job.output_mode === 'clips' ? `${job.num_clips} clips · ${processingModeLabel(job.processing_mode)}` : job.output_mode === 'audio' ? `Extracción de audio · ${(job.audio_format || 'mp3').toUpperCase()}` : `Descarga original · ${job.download_quality === 'auto' ? 'Mejor disponible' : `${job.download_quality}p`}`}</span>}
          <span className="project-card-facts">
            {job.output_mode === 'clips' && <span><IconClips /> {job.num_clips} clips</span>}
            {job.source_height > 0 && <span>{formatResolution(job.source_height)}</span>}
            {job.formats.length > 0 && job.output_mode === 'clips' && <span>{job.formats.join(' / ')}</span>}
          </span>
        </span>
        <span className="project-card-status">
          <StatusBadge status={job.status} />
          {active && <span className="project-progress-value">{pct}%</span>}
        </span>
      </button>
      {active && <div className="project-progress"><ProgressBar value={progress} /><span>{displayJobStage(job)}</span></div>}
      <button type="button" className="project-delete" onClick={onDelete} disabled={deleting} aria-label={`Borrar ${job.title || job.source_name}`} title="Borrar trabajo"><IconTrash /></button>
    </article>
  )
}

function ProjectInspector({ job, previewUrl }: { job: Job; previewUrl: string | null }) {
  return (
    <>
      <div className="inspector-preview">
        {previewUrl ? <img src={previewUrl} alt="Miniatura del clip del trabajo" /> : <div className="inspector-preview-empty"><IconClips /><span>{job.status === 'done' ? 'Vista previa no disponible' : 'El trabajo aún no tiene vista previa'}</span></div>}
        {job.duration ? <span className="inspector-duration">{formatDuration(job.duration)}</span> : null}
      </div>
      <div className="inspector-content">
        <h2>{job.title || job.source_name}</h2>
        <p className="inspector-meta">{job.channel_label || (job.source_type === 'url' ? 'Enlace' : 'Archivo local')} <i /> {formatDateTime(job.created_at)} <i /> {formatDuration(job.duration)}</p>
        <div className="inspector-summary">
          <div className="inspector-summary-heading"><span>Estado del trabajo</span><StatusBadge status={job.status} /></div>
          <div className="inspector-detail-row"><span>Clips generados</span><strong>{job.output_mode === 'clips' ? job.num_clips : '—'}</strong></div>
          <div className="inspector-detail-row"><span>Resolución fuente</span><strong>{formatResolution(job.source_height)}</strong></div>
          <div className="inspector-detail-row"><span>Formato</span><strong>{job.output_mode === 'clips' ? job.formats.join(' / ') || '—' : job.output_mode === 'audio' ? job.audio_format.toUpperCase() : 'Original'}</strong></div>
        </div>
        <Link to={`/jobs/${job.id}`} className="btn-primary inspector-open"><IconClips /> Abrir trabajo</Link>
      </div>
    </>
  )
}

function SearchGlyph() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.8" cy="10.8" r="6.8" /><path d="m16 16 4.4 4.4" /></svg>
}
