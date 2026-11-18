import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../lib/api'
import type { AppSettings, TrackedChannel } from '../lib/types'
import { ChannelsPage } from './ChannelsPage'

vi.mock('../components/topbar', () => ({
  TopbarTitle: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}))

let client: QueryClient

function mount() {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <ChannelsPage />
    </QueryClientProvider>,
  )
}

describe('Canales trackeados: geometría consistente', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getSettings').mockResolvedValue({
      providers: ['naive'], processing_modes: [],
    } as AppSettings)
    vi.spyOn(api, 'listChannels').mockResolvedValue([])
  })

  afterEach(() => {
    cleanup()
    client?.clear()
    vi.restoreAllMocks()
  })

  it('mantiene el estado vacío, resumen y lista bajo los mismos contenedores', async () => {
    const { container } = mount()
    await screen.findByText('Aun no trackeas ningun canal.')

    const page = container.querySelector('.channels-page')
    expect(page).toBeInTheDocument()
    expect(page?.querySelector(':scope > .channels-header')).toBeInTheDocument()
    expect(page?.querySelector(':scope > .channels-overview')).toBeInTheDocument()
    expect(page?.querySelector(':scope > .section-card')).toContainElement(
      screen.getByText('Aun no trackeas ningun canal.'),
    )
    expect(page?.querySelector(':scope > .channels-list')).toHaveAttribute(
      'aria-label', 'Lista de canales',
    )
  })

  it('conserva la misma lista contenedora cuando existen canales', async () => {
    vi.mocked(api.listChannels).mockResolvedValue([{
      id: 'canal-demo',
      label: 'Canal de prueba',
      url: 'https://www.youtube.com/@canal-demo',
      formats: ['9x16'],
      num_clips: 5,
      poll_interval_minutes: 60,
      processing_mode: 'auto_smart',
      enabled: true,
      consecutive_failures: 0,
      last_check_status: 'pending',
    } as TrackedChannel])

    const { container } = mount()
    await screen.findByText('Canal de prueba')

    const page = container.querySelector('.channels-page')
    expect(page?.querySelector(':scope > .channels-list .channel-card')).toBeInTheDocument()
    expect(page?.querySelector(':scope > .channels-overview')).toBeInTheDocument()
  })

  it('sujeta vacío y lista al mismo límite CSS de 1200 px que el resumen', () => {
    const stylesheet = readFileSync(
      resolve(dirname(fileURLToPath(import.meta.url)), '../styles/page-layout.css'), 'utf8',
    )
    const sharedRule = stylesheet.match(
      /\.channels-page > \.channels-header,[\s\S]*?\{ width: 100%; max-width: 1200px; \}/,
    )?.[0]

    expect(sharedRule).toContain('.channels-page > .channels-overview,')
    expect(sharedRule).toContain('.channels-page > .section-card,')
    expect(sharedRule).toContain('.channels-page > .channels-list,')
  })
})
