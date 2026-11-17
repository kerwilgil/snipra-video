import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, type NvidiaCatalog } from '../lib/api'
import { NvidiaModelField } from './NvidiaModelField'

const catalog: NvidiaCatalog = {
  models: [
    { id: 'vendor/new-chat', text: true, vision: false, compatibility_source: 'metadata' },
    { id: 'vendor/new-vision', text: true, vision: true, compatibility_source: 'metadata' },
  ], available: true, stale: false, warning: null, fetched_at: 100, expires_at: 400,
}

afterEach(() => { cleanup(); vi.restoreAllMocks() })

function mount(value = 'auto', vision = false, data = catalog) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  vi.spyOn(api, 'getNvidiaModels').mockResolvedValue(data)
  const change = vi.fn()
  render(<QueryClientProvider client={client}><NvidiaModelField value={value} vision={vision} onChange={change} /></QueryClientProvider>)
  return change
}

describe('NVIDIA dynamic selection', () => {
  it('discovers new IDs, lists auto first and emits auto or the exact manual ID', async () => {
    const change = mount()
    await screen.findByText('Actualizar catálogo')
    await screen.findByRole('button', { name: 'Modelo' })
    // Wait for the async catalog before opening the dropdown.
    await screen.findByText(/Snipra usará automáticamente/)
    fireEvent.click(screen.getByRole('button', { name: 'Modelo' }))
    expect((await screen.findAllByRole('option'))[0]).toHaveTextContent('Automático (recomendado)')
    fireEvent.click(await screen.findByRole('option', { name: 'vendor/new-chat' }))
    expect(change).toHaveBeenCalledWith('vendor/new-chat')
    fireEvent.click(screen.getByRole('button', { name: 'Modelo' }))
    fireEvent.click(screen.getByRole('option', { name: 'Automático (recomendado)' }))
    expect(change).toHaveBeenLastCalledWith('auto')
  })

  it('preserves a retired manual model and explains its availability', async () => {
    const change = mount('retired/custom')
    expect(await screen.findByRole('alert')).toHaveTextContent('ya no está disponible')
    expect(screen.getByRole('button', { name: 'Modelo' })).toHaveTextContent('modelo guardado / no disponible actualmente')
    expect(screen.getByText(/Seleccionaste manualmente/)).toBeInTheDocument()
    expect(change).not.toHaveBeenCalled()
  })

  it('filters vision models and keeps automatic vision selection', async () => {
    mount('auto', true)
    fireEvent.click(screen.getByRole('button', { name: 'Modelo' }))
    expect(await screen.findByRole('option', { name: 'vendor/new-vision' })).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: 'vendor/new-chat' })).not.toBeInTheDocument()
  })

  it('warns about stale catalog without claiming the manual ID is retired', async () => {
    mount('retired/custom', false, { ...catalog, stale: true, warning: 'El catálogo puede estar desactualizado.' })
    expect(await screen.findByText('El catálogo puede estar desactualizado.')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Modelo' })).toHaveTextContent('disponibilidad sin confirmar')
  })
})
