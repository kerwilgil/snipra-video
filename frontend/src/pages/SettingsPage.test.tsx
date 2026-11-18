import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { TopbarSlotsContext } from '../components/topbar'
import { api } from '../lib/api'
import type { AppSettings } from '../lib/types'
import { DownloadCookiesFields, SettingsPage } from './SettingsPage'

describe('Cookies para sitios con sesión', () => {
  it('permite elegir los siete modos sin mostrar valores de cookies', () => {
    const onBrowserChange = vi.fn()
    render(<DownloadCookiesFields browser="auto" path="" onBrowserChange={onBrowserChange} onPathChange={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Cookies para sitios con sesión' }))
    expect(screen.getAllByRole('option')).toHaveLength(7)
    fireEvent.click(screen.getByRole('option', { name: 'Firefox' }))
    expect(onBrowserChange).toHaveBeenCalledWith('firefox')
    expect(screen.getByText(/Permite utilizar una sesión ya iniciada/)).toBeInTheDocument()
  })

  it('registra solo una ruta local en modo archivo', () => {
    const onPathChange = vi.fn()
    render(<DownloadCookiesFields browser="cookies_file" path="" onBrowserChange={vi.fn()} onPathChange={onPathChange} />)
    fireEvent.change(screen.getByLabelText('Ruta cookies.txt'), { target: { value: 'C:\\local\\cookies.txt' } })
    expect(onPathChange).toHaveBeenCalledWith('C:\\local\\cookies.txt')
    expect(screen.getByText(/Se guarda solo la ruta/)).toBeInTheDocument()
  })

  it('ninguno oculta el archivo y conserva la compatibilidad con perfiles existentes', () => {
    const { rerender } = render(<DownloadCookiesFields browser="none" path="C:\\local\\cookies.txt" onBrowserChange={vi.fn()} onPathChange={vi.fn()} />)
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument()
    rerender(<DownloadCookiesFields browser="brave:Default" path="" onBrowserChange={vi.fn()} onPathChange={vi.fn()} />)
    expect(screen.getByText('Perfil de navegador configurado')).toBeInTheDocument()
  })
})

describe('Media Foundation en SettingsPage', () => {
  const settings = {
    provider: 'naive', model: '', whisper_model: 'base', default_num_clips: 5,
    min_clip_seconds: 15, max_clip_seconds: 60, default_formats: ['9x16'],
    available_formats: ['9x16', '1x1', '16x9'], caption_style: 'viral_bold',
    caption_styles: [], providers: ['naive'], sponsorblock_categories: [],
    sponsorblock_categories_available: [], binaries: {}, keys_present: {},
    reframe_mode: 'track', yt_dlp_cookies_from_browser: 'none', yt_dlp_cookies_path: '',
    media_defaults: { destination_profile: 'tiktok', color_handling: 'sdr',
      stabilization: 'medium', video_denoise: 'light', audio_cleanup: 'voice' },
    destination_profiles: [{ id: 'tiktok', display_name: 'TikTok', frame: '1080x1920', aspect: '9:16', max_seconds: 600 }],
  } as AppSettings
  let client: QueryClient
  let actions: HTMLDivElement

  function mount(value = settings) {
    client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } })
    client.setQueryData(['settings'], value)
    client.setQueryDefaults(['storage'], { enabled: false })
    client.setQueryDefaults(['via-availability'], { enabled: false })
    vi.spyOn(api, 'getSettings').mockResolvedValue(value)
    vi.spyOn(api, 'getMediaCapabilities').mockRejectedValue(new Error('HTTP 503'))
    vi.spyOn(api, 'updateSettings').mockResolvedValue(value)
    actions = document.createElement('div')
    document.body.appendChild(actions)
    render(<QueryClientProvider client={client}><MemoryRouter>
      <TopbarSlotsContext.Provider value={{ actionsSlot: actions, titleSlot: null, searchSlot: null }}>
        <SettingsPage />
      </TopbarSlotsContext.Provider>
    </MemoryRouter></QueryClientProvider>)
  }

  afterEach(() => { cleanup(); actions.remove(); client.clear(); vi.restoreAllMocks() })

  it('carga defaults, guarda overrides válidos y sigue operativo si diagnóstico falla', async () => {
    mount()
    expect(screen.getByRole('spinbutton', { name: 'Retencion de archivos (dias; 0 desactiva)' })).toHaveValue(0)
    await screen.findByText(/No se pudo diagnosticar/)
    expect(screen.getByRole('button', { name: 'Destino por defecto' })).toHaveTextContent('TikTok')
    expect(screen.getByRole('button', { name: 'Color HDR' })).toHaveTextContent('Siempre SDR')
    expect(screen.getByRole('button', { name: 'Estabilización' })).toHaveTextContent('Media')
    expect(screen.getByRole('button', { name: 'Reducción de ruido de video' })).toHaveTextContent('Suave')
    expect(screen.getByRole('button', { name: 'Limpieza de audio' })).toHaveTextContent('Voz')
    for (const [label, option] of [
      ['Destino por defecto', 'Auto (según formato)'], ['Color HDR', 'Intentar preservar color'],
      ['Estabilización', 'Fuerte'], ['Reducción de ruido de video', 'Media'], ['Limpieza de audio', 'Voz suave'],
    ]) {
      fireEvent.click(screen.getByRole('button', { name: label }))
      fireEvent.click(screen.getByRole('option', { name: option }))
    }
    fireEvent.click(screen.getByRole('button', { name: 'Guardar ajustes' }))
    await waitFor(() => expect(api.updateSettings).toHaveBeenCalledWith(expect.objectContaining({
      destination_profile: 'auto', color_handling: 'preserve', stabilization: 'strong',
      video_denoise: 'medium', audio_cleanup: 'voice_light',
    }), expect.anything()))
  })

  it('usa auto/off cuando el backend legacy no entrega media_defaults', async () => {
    mount({ ...settings, media_defaults: undefined, destination_profiles: undefined })
    fireEvent.click(screen.getByRole('button', { name: 'Guardar ajustes' }))
    await waitFor(() => expect(api.updateSettings).toHaveBeenCalledWith(expect.objectContaining({
      destination_profile: 'auto', color_handling: 'auto', stabilization: 'off', video_denoise: 'off', audio_cleanup: 'off',
    }), expect.anything()))
  })

  it('guarda NVIDIA automático como auto y conserva un modelo retirado manual', async () => {
    vi.spyOn(api, 'getNvidiaModels').mockResolvedValue({
      models: [{ id: 'new/model-chat', text: true, vision: false, compatibility_source: 'metadata' }],
      available: true, stale: false, warning: null, fetched_at: 1, expires_at: 301,
    })
    mount({ ...settings, provider: 'nvidia', model: 'retired/manual', providers: ['naive', 'nvidia'] })
    expect(await screen.findByRole('alert')).toHaveTextContent('ya no está disponible')
    fireEvent.click(screen.getByRole('button', { name: 'Guardar ajustes' }))
    await waitFor(() => expect(api.updateSettings).toHaveBeenCalledWith(expect.objectContaining({
      provider: 'nvidia', model: 'retired/manual',
    }), expect.anything()))
    await screen.findByText('Ajustes guardados.')
    fireEvent.click(screen.getByRole('button', { name: 'Modelo', exact: true }))
    fireEvent.click(screen.getByRole('option', { name: 'Automático (recomendado)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Guardar ajustes' }))
    await waitFor(() => expect(api.updateSettings).toHaveBeenLastCalledWith(expect.objectContaining({
      provider: 'nvidia', model: 'auto',
    }), expect.anything()))
  })
})
