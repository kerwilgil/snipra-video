import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../lib/api'
import type { Clip, ClipTimeline, DeliveryQCResult } from '../lib/types'
import { ClipCard } from './ClipCard'

const CLIP = {
  id: 'clip', index: 0, start: 0, end: 10, duration: 10, title: 'Clip', hook: '',
  score: 0.8, reason: '', detection_source: '', hashtags: [], description: '',
  source_ranges: [], editorial_cards: [], privacy_suggestions: [], privacy_masks: [],
  protected_zones: [], editorial_qa: {}, assets: [],
} satisfies Clip
let client: QueryClient

async function mount(qc?: Partial<DeliveryQCResult>) {
  const timeline: ClipTimeline = {
    version: 1,
    clip: { title: 'Clip', hook: '', score: 0.8, reason: '', source_start: 0, source_end: 10, duration: 10, trim_offset: 0 },
    processing: { mode: 'auto_smart', requested_reframe: 'track', caption_style: 'viral_bold' },
    formats: [{ format: '9x16', mode: 'track', width: 1080, height: 1920, duration: 10,
      size_bytes: 1000, video: 'video.mp4', thumbnail: '', frames: [],
      quality: { passed: true, attempts: 1, issues: [], checks: {} }, delivery_qc: qc }],
    subtitles: [], words: [],
  }
  vi.spyOn(api, 'getClipTimeline').mockResolvedValue(timeline)
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<QueryClientProvider client={client}><ClipCard clip={CLIP} jobId="job" /></QueryClientProvider>)
  fireEvent.click(screen.getByRole('button', { name: 'Timeline QA' }))
  await screen.findByText('Intentos QA')
  return screen.queryByText('Delivery Check')?.parentElement?.parentElement
}

function result(status: 'pass' | 'warn' | 'fail', passed = status !== 'fail'): DeliveryQCResult {
  return { passed, profile: 'generic_vertical', source_hdr_format: null, output_hdr_format: 'sdr',
    checks: [{ id: 'fps', status, actual: 30, expected: 30, message: 'FPS de salida' }] }
}

describe('Delivery QC en ClipCard/Timeline', () => {
  afterEach(() => { cleanup(); client.clear(); vi.restoreAllMocks() })

  it('tolera timeline legacy sin delivery_qc', async () => {
    expect(await mount()).toBeUndefined()
    expect(screen.getByText('Timeline QA', { selector: 'div' })).toBeInTheDocument()
  })

  it.each([{}, { passed: true }, { passed: true, checks: [] }])('tolera QC parcial sin afirmar PASS: %j', async (qc) => {
    const panel = (await mount(qc))!
    expect(within(panel).getByText(/Desconocido ·/)).toBeInTheDocument()
    expect(within(panel).queryByText(/PASS/)).not.toBeInTheDocument()
  })

  it.each(['pass', 'warn', 'fail'] as const)('representa %s con su severidad en resumen y detalle', async (status) => {
    const panel = (await mount(result(status)))!
    expect(within(panel).getByText(`${status.toUpperCase()} · generic_vertical`)).toBeInTheDocument()
    expect(within(panel).getByText(status.toUpperCase(), { exact: true })).toHaveClass(
      status === 'pass' ? 'text-good' : status === 'warn' ? 'text-warn' : 'text-bad',
    )
  })

  it('no oculta un check FAIL aunque passed sea true', async () => {
    const panel = (await mount(result('fail', true)))!
    expect(within(panel).getByText('FAIL · generic_vertical')).toHaveClass('text-bad')
  })
})
