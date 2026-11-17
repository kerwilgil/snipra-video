import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MediaDiagnosticsPanel } from './SettingsPage'
import { api } from '../lib/api'
import type { MediaCapabilities } from '../lib/types'

const FULL_CAPS: MediaCapabilities = {
  ffmpeg_version: '9.0.1',
  ffprobe_version: '9.0.1',
  overall_status: 'ok',
  encoders: { libx264: 'available', aac: 'available' },
  filters: {},
  features: {
    hdr_to_sdr: 'available',
    stabilization: 'available',
    video_denoise: 'available',
    audio_cleanup: 'available',
    subtitles: 'available',
    social_export: 'available',
  },
  backends: {},
  warnings: [],
}

const LIMITED_CAPS: MediaCapabilities = {
  ...FULL_CAPS,
  overall_status: 'degraded',
  features: {
    hdr_to_sdr: 'missing',
    stabilization: 'deshake_fallback',
    video_denoise: 'missing',
    audio_cleanup: 'missing',
  },
  warnings: ['HDR→SDR no disponible: faltan los filtros zscale y/o tonemap.'],
}

const clients: QueryClient[] = []

function withClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  clients.push(client)
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

describe('Diagnóstico multimedia', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getMediaCapabilities').mockResolvedValue(FULL_CAPS)
  })

  afterEach(() => {
    cleanup()
    clients.splice(0).forEach((client) => client.clear())
    vi.restoreAllMocks()
  })

  it('muestra OK para todas las capabilities cuando están disponibles', async () => {
    withClient(<MediaDiagnosticsPanel />)
    await waitFor(() => expect(screen.getByText('FFmpeg')).toBeInTheDocument())
    for (const label of ['FFmpeg', 'FFprobe', 'H.264', 'AAC', 'Subtítulos', 'HDR → SDR', 'Estabilización', 'Denoise', 'Audio Cleanup']) {
      expect(screen.getByText(label)).toBeInTheDocument()
    }
    expect(screen.getAllByText('OK')).toHaveLength(9)
    expect(screen.queryByText('No disponible')).not.toBeInTheDocument()
    expect(api.getMediaCapabilities).toHaveBeenCalledWith()
  })

  it('distingue No disponible, Limitado (deshake) y Desconocido', async () => {
    vi.mocked(api.getMediaCapabilities).mockResolvedValue(LIMITED_CAPS)
    withClient(<MediaDiagnosticsPanel />)
    await waitFor(() => expect(screen.getByText('HDR → SDR')).toBeInTheDocument())
    expect(screen.getByText(/faltan los filtros zscale/)).toBeInTheDocument()
    expect(screen.getByText('Limitado (deshake)')).toBeInTheDocument()
    expect(screen.getAllByText('No disponible')).toHaveLength(3)
    const subtitles = screen.getByText('Subtítulos').closest('div')!
    expect(within(subtitles).getByText('Desconocido')).toBeInTheDocument()
    expect(within(subtitles).queryByText('No disponible')).not.toBeInTheDocument()
  })

  it('permite forzar un re-diagnóstico con el botón de actualizar', async () => {
    withClient(<MediaDiagnosticsPanel />)
    await screen.findByText('HDR → SDR')
    vi.mocked(api.getMediaCapabilities).mockResolvedValue(LIMITED_CAPS)
    fireEvent.click(screen.getByText('Actualizar diagnóstico'))
    await screen.findByText('Limitado (deshake)')
    expect(api.getMediaCapabilities).toHaveBeenLastCalledWith(true)
    const hdr = screen.getByText('HDR → SDR').closest('div')!
    expect(within(hdr).getByText('No disponible')).toBeInTheDocument()
  })

  it('tolera error del endpoint y permite recuperar con refresh', async () => {
    vi.mocked(api.getMediaCapabilities).mockRejectedValueOnce(new Error('HTTP 503'))
    withClient(<MediaDiagnosticsPanel />)
    await screen.findByText(/No se pudo diagnosticar/)
    fireEvent.click(screen.getByRole('button', { name: 'Actualizar diagnóstico' }))
    await screen.findByText('HDR → SDR')
    expect(api.getMediaCapabilities).toHaveBeenLastCalledWith(true)
    expect(screen.queryByText(/No se pudo diagnosticar/)).not.toBeInTheDocument()
  })

  it('mantiene el último diagnóstico ante error de refresh y permite reintentar', async () => {
    withClient(<MediaDiagnosticsPanel />)
    await screen.findByText('HDR → SDR')
    vi.mocked(api.getMediaCapabilities).mockRejectedValueOnce(new Error('HTTP 503'))
    fireEvent.click(screen.getByRole('button', { name: 'Actualizar diagnóstico' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('No se pudo actualizar')
    expect(screen.getAllByText('OK')).toHaveLength(9)
    fireEvent.click(screen.getByRole('button', { name: 'Actualizar diagnóstico' }))
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument())
  })
})
