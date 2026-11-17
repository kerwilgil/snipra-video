import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../lib/api'
import type { AppSettings, MediaOptions } from '../lib/types'
import { UploadPage } from './UploadPage'

vi.mock('../components/topbar', () => ({
  TopbarTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TopbarActions: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

const OFF: MediaOptions = {
  destination_profile: 'auto', color_handling: 'auto', stabilization: 'off',
  video_denoise: 'off', audio_cleanup: 'off',
}
const ADVANCED: MediaOptions = {
  destination_profile: 'tiktok', color_handling: 'sdr', stabilization: 'medium',
  video_denoise: 'strong', audio_cleanup: 'voice',
}
const SETTINGS = {
  media_defaults: ADVANCED,
  destination_profiles: [{ id: 'tiktok', display_name: 'TikTok', aspect: '9:16', frame: '1080x1920', max_seconds: 600 }],
} as AppSettings
let client: QueryClient

function mount() {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}><MemoryRouter><UploadPage /></MemoryRouter></QueryClientProvider>)
}

function choose(label: string, option: string) {
  fireEvent.click(screen.getByRole('button', { name: label }))
  fireEvent.click(screen.getByRole('option', { name: option }))
}

function submitUrl() {
  fireEvent.change(screen.getByPlaceholderText('https://youtube.com/watch?v=...'), { target: { value: 'https://youtube.com/watch?v=test' } })
  fireEvent.click(screen.getByRole('button', { name: 'Generar clips' }))
}

describe('Opciones Media Foundation en UploadPage', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getSettings').mockResolvedValue({} as AppSettings)
    vi.spyOn(api, 'createFromUrl').mockResolvedValue({ id: 'job', status: 'queued' })
    vi.spyOn(api, 'createFromFile').mockResolvedValue({ id: 'job', status: 'queued' })
  })
  afterEach(() => { cleanup(); client.clear(); vi.restoreAllMocks() })

  it('mantiene defaults auto/off con settings legacy y formatos existentes', async () => {
    mount()
    await waitFor(() => expect(client.getQueryData(['settings'])).toEqual({}))
    submitUrl()
    await waitFor(() => expect(api.createFromUrl).toHaveBeenCalledWith(
      'https://youtube.com/watch?v=test', expect.objectContaining({ formats: ['9x16'], media_options: OFF }),
    ))
  })

  it('aplica defaults recibidos después del primer render', async () => {
    let resolve!: (settings: AppSettings) => void
    vi.mocked(api.getSettings).mockReturnValue(new Promise((done) => { resolve = done }))
    mount()
    expect(screen.getByRole('button', { name: 'Estabilización de video' })).toHaveTextContent('Desactivada')
    await act(async () => { resolve(SETTINGS) })
    await waitFor(() => expect(screen.getByRole('button', { name: 'Destino de publicación' })).toHaveTextContent('TikTok'))
    submitUrl()
    await waitFor(() => expect(api.createFromUrl).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({ media_options: ADVANCED })))
  })

  it('conserva selección explícita aunque settings se cargue o refresque', async () => {
    let resolve!: (settings: AppSettings) => void
    vi.mocked(api.getSettings).mockReturnValue(new Promise((done) => { resolve = done }))
    mount()
    choose('Estabilización de video', 'Estabilización · Suave')
    await act(async () => { resolve(SETTINGS) })
    await waitFor(() => expect(screen.getByRole('button', { name: 'Destino de publicación' })).toHaveTextContent('TikTok'))
    act(() => { client.setQueryData(['settings'], { ...SETTINGS, media_defaults: { ...ADVANCED, stabilization: 'strong' } }) })
    expect(screen.getByRole('button', { name: 'Estabilización de video' })).toHaveTextContent('Suave')
    submitUrl()
    await waitFor(() => expect(api.createFromUrl).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({ media_options: { ...ADVANCED, stabilization: 'light' } })))
  })

  it('envía los cinco valores elegidos al crear desde archivo', async () => {
    vi.mocked(api.getSettings).mockResolvedValue(SETTINGS)
    const { container } = mount()
    await waitFor(() => expect(screen.getByRole('button', { name: 'Destino de publicación' })).toHaveTextContent('TikTok'))
    choose('Manejo de color HDR', 'Color HDR · Intentar preservar')
    choose('Estabilización de video', 'Estabilización · Fuerte')
    choose('Reducción de ruido de video', 'Ruido de video · Medio')
    choose('Limpieza de audio', 'Limpieza de audio · Voz suave')
    const file = new File(['video'], 'video.mp4', { type: 'video/mp4' })
    fireEvent.change(container.querySelector('input[type="file"]')!, { target: { files: [file] } })
    await screen.findByRole('button', { name: 'Quitar archivo' })
    fireEvent.click(screen.getByRole('button', { name: 'Generar clips' }))
    await waitFor(() => expect(api.createFromFile).toHaveBeenCalledWith(file, expect.objectContaining({
      media_options: { destination_profile: 'tiktok', color_handling: 'preserve', stabilization: 'strong', video_denoise: 'medium', audio_cleanup: 'voice_light' },
    })))
  })

  it('mantiene las mismas opciones en todos los jobs de un batch', async () => {
    vi.mocked(api.getSettings).mockResolvedValue(SETTINGS)
    mount()
    await waitFor(() => expect(client.getQueryData(['settings'])).toEqual(SETTINGS))
    fireEvent.click(screen.getByRole('button', { name: 'Pegar varias URLs' }))
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'https://youtube.com/watch?v=one\nhttps://youtube.com/watch?v=two' } })
    fireEvent.click(screen.getByRole('button', { name: 'Crear jobs' }))
    await waitFor(() => expect(api.createFromUrl).toHaveBeenCalledTimes(2))
    for (const [, options] of vi.mocked(api.createFromUrl).mock.calls) {
      expect(options?.media_options).toEqual(ADVANCED)
    }
  })
})
