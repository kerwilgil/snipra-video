import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../lib/api'
import { Select } from './Select'

export function NvidiaModelField({ value, onChange, vision = false, label = 'Modelo', compact = false }: {
  value: string
  onChange: (value: string) => void
  vision?: boolean
  label?: string
  compact?: boolean
}) {
  const client = useQueryClient()
  const { data, isPending, isError, isFetching } = useQuery({
    queryKey: ['nvidia-models'], queryFn: () => api.getNvidiaModels(),
    staleTime: 60_000, refetchInterval: 300_000,
  })
  const refresh = useMutation({
    mutationFn: () => api.getNvidiaModels(true),
    onSuccess: (catalog) => client.setQueryData(['nvidia-models'], catalog),
  })
  const models = data?.models.filter((m) => vision ? m.vision : m.text) ?? []
  const selected = value || 'auto'
  const missing = selected !== 'auto' && !models.some((m) => m.id === selected)
  const unavailable = missing && data?.available && !data.stale
  return (
    <div className={`config-field ${compact ? 'nvidia-model-compact' : ''}`}>
      <span className={compact ? 'sr-only' : 'mb-2 block text-xs text-ink-mute'}>{label}</span>
      <Select className={compact ? 'agent-model-select' : undefined} ariaLabel={label} value={selected} onChange={onChange} options={[
        { value: 'auto', label: 'Automático (recomendado)' },
        ...models.map((m) => ({ value: m.id, label: m.id })),
        ...(missing ? [{ value: selected, label: `${selected} (${unavailable
          ? 'modelo guardado / no disponible actualmente' : 'modelo guardado / disponibilidad sin confirmar'})` }] : []),
      ]} />
      <p className={compact ? 'sr-only' : 'config-description text-xs text-ink-mute'}>
        {selected === 'auto'
          ? 'Snipra usará automáticamente el modelo recomendado actual.'
          : 'Seleccionaste manualmente un modelo específico.'}
      </p>
      {unavailable && <p role="alert" className="text-xs text-warn">El modelo guardado ya no está disponible. Elige otro o Automático; Snipra no lo sustituirá.</p>}
      {isPending && <p role="status" className="text-xs text-ink-mute">Cargando catálogo NVIDIA…</p>}
      {(isError || refresh.isError || data?.warning) && <p role="status" className="text-xs text-warn">
        {data?.warning ?? 'No se pudo consultar el catálogo NVIDIA. La selección guardada se conserva.'}
      </p>}
      {!compact && models.some((m) => m.compatibility_source === 'inferred') && <p className="text-xs text-ink-mute">
        NVIDIA no declara todas las modalidades; algunas se infieren del nombre.
      </p>}
      {!compact && <button type="button" className="text-left text-xs text-cyan disabled:opacity-50"
        disabled={isFetching || refresh.isPending} onClick={() => refresh.mutate()}>Actualizar catálogo</button>
      }
    </div>
  )
}
