import { useMemo, useState } from 'react'
import type { HighlightCluster, HighlightEvent, ShortBatchPlanResponse } from '../lib/types'

const time = (value: number | null | undefined) => value == null ? '—' : `${value.toFixed(2)} s`
const windowTime = (start: number | null, end: number | null) => start == null || end == null ? '—' : `${time(start)} → ${time(end)}`

export function HighlightInspector({
  plan, loading, error, requestedCount, onRequestedCountChange, onAnalyze, onSeek,
}: {
  plan: ShortBatchPlanResponse | null; loading: boolean; error: string | null; requestedCount: number
  onRequestedCountChange: (value: number) => void; onAnalyze: () => void; onSeek: (seconds: number) => void
}) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const events = useMemo(() => [...(plan?.enhanced_evidence.events ?? [])].sort((a, b) => a.rank - b.rank), [plan])
  const selected = useMemo(() => new Map((plan?.shorts ?? []).map((short) => [short.event_id, short])), [plan])
  const clusterByEvent = useMemo(() => clusterMap(plan?.enhanced_evidence.clusters ?? []), [plan])
  return <section className="highlight-inspector" aria-label="Highlight Inspector">
    <header><div><b>Highlights</b><span>Inspección read-only de evidencia analítica</span></div></header>
    <label className="highlight-count">Cantidad
      <input aria-label="Cantidad de highlights" type="number" min="1" max="10" value={requestedCount} disabled={loading} onChange={(event) => onRequestedCountChange(Math.min(10, Math.max(1, Number(event.target.value) || 1)))} />
    </label>
    <button className="highlight-analyze" disabled={loading} onClick={onAnalyze}>{loading ? 'Analizando highlights…' : 'Analizar highlights'}</button>
    {loading && <p className="highlight-progress" role="status" aria-live="polite"><span aria-hidden="true" />Snipra está evaluando candidatos, refinando momentos y preparando una propuesta editorial.</p>}
    {error && <p className="editor-error" role="alert">{error}</p>}
    {plan && <p className={`highlight-result ${plan.status}`}>{resultLabel(plan)}</p>}
    {plan && !events.length && <p className="editor-empty">No se encontraron eventos inspeccionables.</p>}
    <div className="highlight-events">
      {events.map((event) => {
        const short = selected.get(event.event_id)
        const cluster = clusterByEvent.get(event.event_id)
        const open = expanded === event.event_id
        return <article key={event.event_id} className={`highlight-card ${short ? 'director-selected' : ''}`}>
          <div className="highlight-card-main">
            <span className="highlight-rank">Ranking analítico #{event.rank}</span>
            {short ? <span className="director-badge">Seleccionado por Director</span> : <span className="not-selected">No seleccionado</span>}
            <strong>{event.title}</strong>
            <small>Score analítico {event.overall_score.toFixed(3)} · {time(event.refined_start)} → {time(event.refined_end)} · {time(event.refined_end - event.refined_start)}</small>
            <small>Tema: {event.primary_topic_id ?? '—'} · Refinamiento: {event.refinement_status} ({event.refinement_confidence.toFixed(2)})</small>
            {short && <p className="director-reason">{short.director_reason}</p>}
            {cluster && <ClusterLabel cluster={cluster.cluster} eventId={event.event_id} />}
          </div>
          <div className="highlight-card-actions"><button onClick={() => onSeek(event.refined_start)}>Ver momento</button><button aria-expanded={open} onClick={() => setExpanded(open ? null : event.event_id)}>{open ? 'Ocultar detalle' : 'Ver detalle'}</button></div>
          {open && <EventDetails event={event} cluster={cluster?.cluster} />}
        </article>
      })}
    </div>
  </section>
}

function resultLabel(plan: ShortBatchPlanResponse) {
  if (plan.status === 'partial') return `${plan.selected_count} de ${plan.requested_count} propuestos — resultado parcial`
  if (plan.status === 'planned') return `${plan.selected_count} de ${plan.requested_count} propuestos`
  if (plan.status === 'failed') return 'No fue posible preparar una propuesta'
  return 'La propuesta requiere revisión'
}
function clusterMap(clusters: HighlightCluster[]) {
  const map = new Map<string, { cluster: HighlightCluster }>()
  clusters.forEach((cluster) => cluster.member_event_ids.forEach((id) => map.set(id, { cluster })))
  return map
}
function ClusterLabel({ cluster, eventId }: { cluster: HighlightCluster; eventId: string }) {
  const relation = cluster.relations.find((item) => item.event_id === eventId)
  return <small className="cluster-label">Agrupación analítica · Cluster relacionado {cluster.cluster_id} · {cluster.representative_event_id === eventId ? 'Representative' : 'Near duplicate relation'}{relation?.relation ? `: ${relation.relation}` : ''}</small>
}
function EventDetails({ event, cluster }: { event: HighlightEvent; cluster?: HighlightCluster }) {
  return <div className="highlight-details">
    <section><b>Boundaries</b><p>Original: {time(event.original_start)} → {time(event.original_end)}</p><p>Refinado: {time(event.refined_start)} → {time(event.refined_end)}</p><p>Status: {event.refinement_status} · Confidence: {event.refinement_confidence.toFixed(2)}</p></section>
    <section><b>Signals</b><p>Peak: {time(event.peak_at)} · Payoff: {time(event.payoff_at)} ({windowTime(event.payoff_window_start, event.payoff_window_end)})</p><p>Reaction: {time(event.reaction_at)} ({windowTime(event.reaction_window_start, event.reaction_window_end)})</p></section>
    <section><b>Topics</b><p>Primary: {event.primary_topic_id ?? '—'}</p>{event.topic_contexts.map((topic) => <p key={topic.topic_id}>{topic.topic_id} · overlap {topic.overlap_ratio.toFixed(2)} · confidence {topic.confidence.toFixed(2)}<br />{topic.keywords.join(', ') || '—'} · {topic.representative_text || '—'}</p>)}</section>
    <section><b>Score explanation</b><p>overall_score: {event.overall_score.toFixed(3)} (dato analítico; no es decisión editorial)</p>{event.score_components.map((part) => <p key={part.name}>{part.name}: raw {part.raw_value} · weight {part.weight} · contribution {part.contribution} · {part.note}</p>)}</section>
    <section><b>Evidence</b><p>{event.evidence_refs.join(', ') || '—'}</p>{cluster && <p>Agrupación analítica: {cluster.member_event_ids.join(', ')}. El representative no establece prioridad editorial.</p>}</section>
  </div>
}
