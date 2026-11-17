import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { ConfirmDialog } from '../components/ConfirmDialog'
import { IconDownload, IconFilm, IconRefresh, IconTrash } from '../components/icons'
import { Select } from '../components/Select'
import { ToggleField } from '../components/ToggleSwitch'
import { TopbarActions, TopbarTitle } from '../components/topbar'
import { ProgressBar, Spinner } from '../components/ui'
import { api } from '../lib/api'
import { BGM_PATH_PLACEHOLDER, formatBytes, formatDuration } from '../lib/format'
import type { Compilation, CompilationTransition, ComposableClip } from '../lib/types'

const TRANSITION_OPTIONS = [
  { value: 'none', label: 'Corte directo' },
  { value: 'fade', label: 'Crossfade' },
]

const STATUS_LABELS: Record<Compilation['status'], string> = {
  queued: 'En cola',
  rendering: 'Renderizando',
  done: 'Listo',
  failed: 'Fallo',
}

const MAX_CLIPS = 30

export function ComposePage() {
  const qc = useQueryClient()

  const { data: clips, isLoading, isError: clipsError, isFetching: clipsFetching, refetch: refetchClips } = useQuery({
    queryKey: ['composable-clips'],
    queryFn: api.listComposableClips,
  })

  const { data: compilations, isError: compilationsError, isFetching: compilationsFetching, refetch: refetchCompilations } = useQuery({
    queryKey: ['compositions'],
    queryFn: api.listCompositions,
    // Mientras haya algo en cola o renderizando hay que refrescar para ver el avance.
    refetchInterval: (q) =>
      (q.state.data ?? []).some((item) => item.status === 'queued' || item.status === 'rendering')
        ? 2500
        : false,
  })

  // Orden de selección = orden final en el video, por eso es un array y no un Set.
  const [selected, setSelected] = useState<string[]>([])
  const [title, setTitle] = useState('')
  const [transition, setTransition] = useState<CompilationTransition>('fade')
  const [transitionSeconds, setTransitionSeconds] = useState(0.4)
  const [bgmPath, setBgmPath] = useState('')
  const [bgmVolume, setBgmVolume] = useState(0.18)
  const [bgmDucking, setBgmDucking] = useState(true)
  const [bgmAutoLevel, setBgmAutoLevel] = useState(true)
  const [bgmDuckLu, setBgmDuckLu] = useState(12)
  const [error, setError] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<Compilation | null>(null)
  const [pendingClipDelete, setPendingClipDelete] = useState<ComposableClip | null>(null)
  // La lista de compilaciones vivia debajo de la biblioteca, que es el elemento
  // mas alto de la pagina: al componer, el resultado quedaba fuera de pantalla y
  // parecia que el boton no hacia nada. Con vistas separadas el feedback es
  // inmediato porque al crear se salta sola a "Compilaciones".
  const [view, setView] = useState<'compose' | 'library'>('compose')

  const byId = useMemo(
    () => new Map((clips ?? []).map((clip) => [clip.asset_id, clip])),
    [clips],
  )
  const selectedClips = useMemo(
    () => selected.map((id) => byId.get(id)).filter((clip): clip is ComposableClip => !!clip),
    [selected, byId],
  )
  // Solo se pueden unir clips del mismo formato: el primero elegido fija el formato
  // y el resto se deshabilita, así el error se evita en la UI y no en el servidor.
  const lockedFormat = selectedClips[0]?.format ?? null
  const totalSeconds = selectedClips.reduce((sum, clip) => sum + clip.duration, 0)
  const overlapSeconds =
    transition === 'fade' ? Math.max(0, selectedClips.length - 1) * transitionSeconds : 0
  const estimatedSeconds = Math.max(0, totalSeconds - overlapSeconds)
  const activeCount = (compilations ?? []).filter(
    (item) => item.status === 'queued' || item.status === 'rendering',
  ).length

  const create = useMutation({
    mutationFn: () =>
      api.createCompilation({
        asset_ids: selected,
        title: title.trim(),
        transition,
        transition_seconds: transitionSeconds,
        bgm_path: bgmPath.trim(),
        bgm_volume: bgmVolume,
        bgm_ducking: bgmDucking,
        bgm_auto_level: bgmAutoLevel,
        bgm_duck_lu: bgmDuckLu,
      }),
    onSuccess: () => {
      setSelected([])
      setTitle('')
      setError(null)
      setView('library')  // el usuario ve al instante su compilacion en cola
      qc.invalidateQueries({ queryKey: ['compositions'] })
    },
    onError: (e) => setError(String(e instanceof Error ? e.message : e)),
  })

  const retry = useMutation({
    mutationFn: api.retryCompilation,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['compositions'] }),
  })

  const remove = useMutation({
    mutationFn: api.deleteCompilation,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['compositions'] }),
  })

  const removeClip = useMutation({
    mutationFn: api.deleteAsset,
    onSuccess: (_res, assetId) => {
      // Si estaba elegido para la compilacion en curso, hay que sacarlo tambien
      // de la seleccion o quedaria un id fantasma que el backend rechazaria.
      setSelected((prev) => prev.filter((id) => id !== assetId))
      setError(null)
      qc.invalidateQueries({ queryKey: ['composable-clips'] })
    },
    onError: (e) => setError(String(e instanceof Error ? e.message : e)),
  })

  function toggle(clip: ComposableClip) {
    setError(null)
    setSelected((prev) => {
      if (prev.includes(clip.asset_id)) return prev.filter((id) => id !== clip.asset_id)
      if (prev.length >= MAX_CLIPS) return prev
      return [...prev, clip.asset_id]
    })
  }

  function move(assetId: string, direction: -1 | 1) {
    setSelected((prev) => {
      const index = prev.indexOf(assetId)
      const target = index + direction
      if (index < 0 || target < 0 || target >= prev.length) return prev
      const next = [...prev]
      ;[next[index], next[target]] = [next[target], next[index]]
      return next
    })
  }

  return (
    <div className="page-container section-stack">
      <TopbarTitle>
        <span className="topbar-context">Compositor</span>
        <strong>Componer video</strong>
      </TopbarTitle>
      {view === 'compose' && (
        <TopbarActions>
          <button
            type="button"
            onClick={() => {
              setError(null)
              if (selected.length < 2) return setError('Elige al menos 2 clips para componer.')
              create.mutate()
            }}
            disabled={create.isPending}
            className="btn-primary disabled:cursor-not-allowed"
          >
            {create.isPending ? <Spinner /> : <IconFilm className="h-[18px] w-[18px]" />}
            {create.isPending ? 'Encolando...' : 'Componer video'}
          </button>
        </TopbarActions>
      )}
      <header className="page-header">
        <h1>Componer video</h1>
        <p className="page-description">Une clips que ya generaste en un video nuevo: recopilados, mejores momentos, resúmenes. Todo local, sobre tu propio material.</p>
      </header>

      <div className="flex gap-1 border-b border-border" role="tablist" aria-label="Vistas del compositor">
        <TabButton active={view === 'compose'} onClick={() => setView('compose')}>
          Componer
        </TabButton>
        <TabButton active={view === 'library'} onClick={() => setView('library')}>
          Compilaciones
          {(compilations ?? []).length > 0 && (
            // Cuando hay algo renderizando el badge se pone cian y late; antes
            // llevaba un "⋯" que en esta fuente se veia como un guion suelto.
            <span
              className={`num ml-2 inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[10px] font-bold ${
                activeCount > 0 ? 'animate-pulse bg-cyan text-black' : 'bg-border text-ink-soft'
              }`}
            >
              {activeCount > 0 && (
                <span className="h-1.5 w-1.5 rounded-full bg-black/70" aria-hidden="true" />
              )}
              {activeCount > 0 ? activeCount : (compilations ?? []).length}
            </span>
          )}
        </TabButton>
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-bad/30 bg-bad/[0.06] px-4 py-3 text-sm text-bad">
          {error}
        </div>
      )}
      {view === 'compose' && clipsError && <div role="alert" className="page-error"><span>No se pudo cargar la biblioteca de clips.</span><button type="button" className="btn-secondary" disabled={clipsFetching} onClick={() => void refetchClips()}>{clipsFetching ? 'Reintentando...' : 'Reintentar'}</button></div>}
      {view === 'library' && compilationsError && <div role="alert" className="page-error"><span>No se pudieron cargar las compilaciones.</span><button type="button" className="btn-secondary" disabled={compilationsFetching} onClick={() => void refetchCompilations()}>{compilationsFetching ? 'Reintentando...' : 'Reintentar'}</button></div>}

      {view === 'compose' && (
      <section className="grid gap-4 lg:grid-cols-[1fr_320px]">
        <div className="section-card min-w-0">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-sm font-semibold text-ink">Biblioteca de clips</div>
              <div className="mt-1 text-xs text-ink-mute">
                {lockedFormat
                  ? `Formato fijado a ${lockedFormat} por el primer clip elegido`
                  : 'El primer clip que elijas fija el formato de la compilación'}
              </div>
            </div>
            {selected.length > 0 && (
              <button
                type="button"
                onClick={() => setSelected([])}
                className="text-xs text-ink-mute underline-offset-2 hover:text-cyan hover:underline"
              >
                Limpiar selección
              </button>
            )}
          </div>

          {isLoading && <p className="text-sm text-ink-mute">Cargando biblioteca...</p>}

          {!isLoading && !clipsError && (clips ?? []).length === 0 && (
            <p className="text-sm text-ink-mute">
              Todavía no hay clips renderizados. Generá clips primero desde "Subir video".
            </p>
          )}

          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {(clips ?? []).map((clip) => {
              const order = selected.indexOf(clip.asset_id)
              const isSelected = order >= 0
              const blocked = !isSelected && lockedFormat !== null && clip.format !== lockedFormat
              return (
                // El boton de borrar va como HERMANO del de seleccion, no dentro:
                // un <button> anidado en otro <button> es HTML invalido.
                <div key={clip.asset_id} className="relative">
                <button
                  type="button"
                  onClick={() => toggle(clip)}
                  disabled={blocked}
                  aria-pressed={isSelected}
                  title={blocked ? `No se puede mezclar ${clip.format} con ${lockedFormat}` : clip.title}
                  className={`relative flex w-full cursor-pointer flex-col overflow-hidden rounded-lg border text-left transition-colors ${
                    isSelected
                      ? 'border-cyan/60 bg-cyan/[0.07]'
                      : blocked
                        ? 'cursor-not-allowed border-border bg-base/20 opacity-40'
                        : 'border-border bg-base/40 hover:border-border-strong'
                  }`}
                >
                  {/* La imagen va absolute: con h-full en flujo normal, el alto
                      porcentual depende de una caja cuyo alto sale del aspect-ratio,
                      el navegador cae al tamaño intrinseco y estira la tarjeta.
                      aspect-square (no video) porque la biblioteca mezcla 9x16, 1x1
                      y 16x9: un recorte cuadrado muestra bastante de los verticales
                      sin desbalancear la grilla. */}
                  <div className="relative aspect-square w-full overflow-hidden bg-black">
                    {clip.thumb_url ? (
                      <img
                        src={clip.thumb_url}
                        alt=""
                        className="absolute inset-0 h-full w-full object-cover"
                        loading="lazy"
                      />
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center text-[11px] text-ink-mute">
                        sin miniatura
                      </div>
                    )}
                    {isSelected && (
                      <span className="num absolute left-1.5 top-1.5 flex h-6 w-6 items-center justify-center rounded-full bg-cyan text-[11px] font-bold text-black">
                        {order + 1}
                      </span>
                    )}
                    <span className="num absolute right-1.5 top-1.5 rounded bg-black/75 px-1.5 py-0.5 text-[10px] font-semibold text-white">
                      {clip.format}
                    </span>
                  </div>
                  <div className="min-w-0 p-2.5 pr-10">
                    <div className="truncate text-xs font-semibold text-ink">
                      {clip.title || `Clip ${clip.clip_index + 1}`}
                    </div>
                    <div className="num mt-1 truncate text-[10px] text-ink-mute">
                      {formatDuration(clip.duration)} · {clip.job_title}
                    </div>
                  </div>
                </button>
                <button
                  type="button"
                  onClick={() => setPendingClipDelete(clip)}
                  disabled={removeClip.isPending}
                  title={`Borrar el ${clip.format} de este clip del disco`}
                  aria-label={`Borrar ${clip.title || 'clip'} en ${clip.format}`}
                  className="absolute bottom-2 right-2 flex h-7 w-7 cursor-pointer items-center justify-center rounded-md border border-border bg-panel/90 text-ink-mute transition-colors hover:border-bad/50 hover:text-bad disabled:opacity-40"
                >
                  <IconTrash className="h-3.5 w-3.5" />
                </button>
                </div>
              )
            })}
          </div>
        </div>

        <div className="flex min-w-0 flex-col gap-4">
          <div className="card p-5">
            <div className="mb-3 text-sm font-semibold text-ink">
              Orden final ({selectedClips.length})
            </div>
            {selectedClips.length === 0 ? (
              <p className="text-xs text-ink-mute">
                Tocá clips de la biblioteca para agregarlos. El número indica su posición.
              </p>
            ) : (
              <>
                <ol className="space-y-1.5">
                  {selectedClips.map((clip, index) => (
                    <li
                      key={clip.asset_id}
                      className="flex items-center gap-2 rounded-lg border border-border bg-base/40 px-2 py-1.5"
                    >
                      <span className="num w-5 shrink-0 text-[11px] font-semibold text-cyan">
                        {index + 1}
                      </span>
                      <span className="min-w-0 flex-1 truncate text-[11px] text-ink-soft">
                        {clip.title || `Clip ${clip.clip_index + 1}`}
                      </span>
                      <span className="num shrink-0 text-[10px] text-ink-mute">
                        {formatDuration(clip.duration)}
                      </span>
                      <span className="flex shrink-0 gap-0.5">
                        <button
                          type="button"
                          onClick={() => move(clip.asset_id, -1)}
                          disabled={index === 0}
                          aria-label={`Subir ${clip.title || 'clip'}`}
                          className="rounded border border-border px-1.5 text-[11px] text-ink-mute hover:border-cyan/50 hover:text-cyan disabled:opacity-30"
                        >
                          ↑
                        </button>
                        <button
                          type="button"
                          onClick={() => move(clip.asset_id, 1)}
                          disabled={index === selectedClips.length - 1}
                          aria-label={`Bajar ${clip.title || 'clip'}`}
                          className="rounded border border-border px-1.5 text-[11px] text-ink-mute hover:border-cyan/50 hover:text-cyan disabled:opacity-30"
                        >
                          ↓
                        </button>
                      </span>
                    </li>
                  ))}
                </ol>
                <p className="num mt-3 text-[11px] text-ink-mute">
                  Duración estimada: {formatDuration(estimatedSeconds)}
                  {overlapSeconds > 0 && ` (${overlapSeconds.toFixed(1)}s de cruce)`}
                </p>
              </>
            )}
          </div>

          <div className="card space-y-3 p-5">
            <div className="text-sm font-semibold text-ink">Ajustes</div>
            <label className="block">
              <span className="mb-1.5 block text-xs text-ink-mute">Título</span>
              <input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Mejores momentos"
                className="field w-full px-3 py-2 text-sm"
              />
            </label>
            <label className="block">
              <span className="mb-1.5 block text-xs text-ink-mute">Transición</span>
              <Select
                ariaLabel="Transición entre clips"
                value={transition}
                onChange={(v) => setTransition(v as CompilationTransition)}
                options={TRANSITION_OPTIONS}
              />
            </label>
            {transition === 'fade' && (
              <label className="block">
                <span className="mb-1.5 flex items-center justify-between text-xs text-ink-mute">
                  Duración del cruce
                  <span className="num text-cyan">{transitionSeconds.toFixed(1)}s</span>
                </span>
                <input
                  type="range"
                  min={0.2}
                  max={1.5}
                  step={0.1}
                  value={transitionSeconds}
                  onChange={(e) => setTransitionSeconds(Number(e.target.value))}
                  className="w-full cursor-pointer"
                />
              </label>
            )}
          </div>

          <div className="card space-y-3 p-5">
            <div className="text-sm font-semibold text-ink">Música de fondo</div>
            <label className="block">
              <span className="mb-1.5 block text-xs text-ink-mute">Ruta del archivo (opcional)</span>
              <input
                value={bgmPath}
                onChange={(e) => setBgmPath(e.target.value)}
                placeholder={BGM_PATH_PLACEHOLDER}
                className="field w-full px-3 py-2 text-xs"
              />
            </label>
            {bgmPath.trim() && (
              <>
                <ToggleField
                  label="Nivel automático por LUFS"
                  checked={bgmAutoLevel}
                  onChange={setBgmAutoLevel}
                />
                {bgmAutoLevel ? (
                  <label className="block">
                    <span className="mb-1.5 flex items-center justify-between text-xs text-ink-mute">
                      Música por debajo de la voz
                      <span className="num text-cyan">-{bgmDuckLu} LU</span>
                    </span>
                    <input
                      type="range"
                      min={6}
                      max={24}
                      step={1}
                      value={bgmDuckLu}
                      onChange={(e) => setBgmDuckLu(Number(e.target.value))}
                      className="w-full cursor-pointer"
                    />
                  </label>
                ) : <label className="block">
                  <span className="mb-1.5 flex items-center justify-between text-xs text-ink-mute">
                    Volumen
                    <span className="num text-cyan">{Math.round(bgmVolume * 100)}%</span>
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.02}
                    value={bgmVolume}
                    onChange={(e) => setBgmVolume(Number(e.target.value))}
                    className="w-full cursor-pointer"
                  />
                </label>}
                <ToggleField
                  label="Bajar música cuando hay voz"
                  checked={bgmDucking}
                  onChange={setBgmDucking}
                />
                <p className="text-[11px] leading-relaxed text-ink-soft">
                  Snipra mide voz y música con EBU R128. El ducking baja además la música
                  mientras alguien habla.
                </p>
              </>
            )}
          </div>
        </div>
      </section>
      )}

      {view === 'library' && (
        <section className="space-y-3">
          {(compilations ?? []).length === 0 ? (
            <div className="card flex min-h-[220px] flex-col items-center justify-center gap-3 p-10 text-center">
              <IconFilm className="h-8 w-8 text-ink-mute" />
              <p className="text-sm text-ink-soft">Todavía no compusiste ningún video.</p>
              <button
                type="button"
                onClick={() => setView('compose')}
                className="text-sm font-medium text-cyan hover:underline"
              >
                Componer el primero
              </button>
            </div>
          ) : (
            <div className="grid gap-3">
              {(compilations ?? []).map((item) => (
                <CompilationRow
                  key={item.id}
                  compilation={item}
                  retrying={retry.isPending}
                  onRetry={() => retry.mutate(item.id)}
                  onDelete={() => setPendingDelete(item)}
                />
              ))}
            </div>
          )}
        </section>
      )}

      <ConfirmDialog
        open={pendingDelete !== null}
        title="Borrar compilación"
        message={`Borrar "${pendingDelete?.title || 'compilación'}"? Los clips originales no se tocan.`}
        confirmLabel="Borrar"
        onConfirm={() => {
          if (pendingDelete) remove.mutate(pendingDelete.id)
          setPendingDelete(null)
        }}
        onCancel={() => setPendingDelete(null)}
      />

      <ConfirmDialog
        open={pendingClipDelete !== null}
        title="Borrar clip de la biblioteca"
        message={
          pendingClipDelete
            ? `Borrar del disco el ${pendingClipDelete.format} de "${
                pendingClipDelete.title || `Clip ${pendingClipDelete.clip_index + 1}`
              }"? Es permanente y tambien desaparece de Mis trabajos. Los otros formatos del mismo clip, si los hay, se conservan.`
            : ''
        }
        confirmLabel="Borrar del disco"
        onConfirm={() => {
          if (pendingClipDelete) removeClip.mutate(pendingClipDelete.asset_id)
          setPendingClipDelete(null)
        }}
        onCancel={() => setPendingClipDelete(null)}
      />
    </div>
  )
}

function CompilationRow({
  compilation,
  retrying,
  onRetry,
  onDelete,
}: {
  compilation: Compilation
  retrying: boolean
  onRetry: () => void
  onDelete: () => void
}) {
  const active = compilation.status === 'queued' || compilation.status === 'rendering'
  const done = compilation.status === 'done'

  return (
    <div className="card p-4 transition-colors hover:border-border-strong">
      <div className="grid gap-4 lg:grid-cols-[120px_1fr_auto] lg:items-center">
        <div className="relative aspect-square overflow-hidden rounded-lg border border-border bg-black">
          {done ? (
            <img
              src={api.compilationThumbUrl(compilation.id)}
              alt=""
              className="absolute inset-0 h-full w-full object-cover"
              loading="lazy"
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <IconFilm className="h-6 w-6 text-ink-mute" />
            </div>
          )}
        </div>

        <div className="min-w-0 space-y-2">
          <div className="truncate text-sm font-semibold text-ink">
            {compilation.title || 'Sin título'}
          </div>
          <div className="flex flex-wrap gap-1.5">
            <MetaChip>{STATUS_LABELS[compilation.status]}</MetaChip>
            <MetaChip>{compilation.clip_count} clips</MetaChip>
            <MetaChip>{compilation.format}</MetaChip>
            <MetaChip>{compilation.transition === 'fade' ? 'Crossfade' : 'Corte directo'}</MetaChip>
            {compilation.bgm_path && <MetaChip>Con música</MetaChip>}
            {compilation.bgm_path && compilation.bgm_auto_level && <MetaChip>LUFS auto</MetaChip>}
            {done && <MetaChip>{formatDuration(compilation.duration)}</MetaChip>}
            {done && <MetaChip>{formatBytes(compilation.size_bytes)}</MetaChip>}
          </div>
          {active && <ProgressBar value={compilation.progress} />}
          {compilation.status === 'failed' && compilation.error && (
            <p className="break-words text-xs leading-5 text-bad">{compilation.error}</p>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-2">
          {done && (
            <a
              href={api.compilationVideoUrl(compilation.id)}
              download
              className="btn-primary"
            >
              <IconDownload className="h-4 w-4" />
              Descargar
            </a>
          )}
          {compilation.status === 'failed' && (
            <button
              type="button"
              onClick={onRetry}
              disabled={retrying}
              className="rounded-lg border border-border px-3 py-2 text-xs text-ink-soft hover:border-cyan/50 hover:text-cyan disabled:opacity-50"
            >
              <IconRefresh className="h-4 w-4" />
            </button>
          )}
          <button
            type="button"
            onClick={onDelete}
            disabled={active}
            title={active ? 'No se puede borrar mientras se renderiza' : 'Borrar compilación'}
            aria-label="Borrar compilación"
            className="flex h-9 w-9 shrink-0 cursor-pointer items-center justify-center rounded-lg border border-border text-ink-mute transition-colors hover:border-bad/50 hover:text-bad disabled:opacity-40"
          >
            <IconTrash className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  )
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      // -mb-px para que el borde inferior activo tape el borde del contenedor
      className={`-mb-px cursor-pointer border-b-2 px-4 py-2.5 text-sm font-medium transition-colors ${
        active
          ? 'border-cyan text-ink'
          : 'border-transparent text-ink-mute hover:text-ink-soft'
      }`}
    >
      {children}
    </button>
  )
}

function MetaChip({ children }: { children: React.ReactNode }) {
  return (
    <span className="num rounded-md border border-border bg-base/40 px-2 py-1 text-[11px] text-ink-mute">
      {children}
    </span>
  )
}
