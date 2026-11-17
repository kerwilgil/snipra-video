import type { ProcessingMode } from './types'

export type GamingOutput = 'auto' | 'clips' | 'hook' | 'highlights'

export const GAMING_OUTPUT_OPTIONS: Array<{
  value: GamingOutput
  label: string
  description: string
}> = [
  {
    value: 'auto',
    label: 'Automatico',
    description: 'Snipra decide entre clips, hook o highlights segun los eventos detectados.',
  },
  {
    value: 'clips',
    label: 'Clips independientes',
    description: 'Genera varios videos separados con momentos completos.',
  },
  {
    value: 'hook',
    label: 'Hook',
    description: 'Genera un solo edit de 8-25 s con accion, desenlace y posible replay.',
  },
  {
    value: 'highlights',
    label: 'Highlights',
    description: 'Genera un montaje de 30-45 s con 4-8 microjugadas.',
  },
]

const GAMING_MODE_BY_OUTPUT: Record<GamingOutput, ProcessingMode> = {
  auto: 'gaming_auto',
  clips: 'gaming',
  hook: 'gaming_hook',
  highlights: 'gaming_highlights',
}

export function resolveGamingProcessingMode(
  contentMode: ProcessingMode,
  output: GamingOutput,
): ProcessingMode {
  return contentMode === 'gaming' ? GAMING_MODE_BY_OUTPUT[output] : contentMode
}

export function isInternalGamingMode(mode: ProcessingMode): boolean {
  return mode === 'gaming_auto' || mode === 'gaming_hook' || mode === 'gaming_highlights'
}

export function processingModeLabel(mode: ProcessingMode): string {
  const labels: Partial<Record<ProcessingMode, string>> = {
    gaming: 'Gaming / Clips',
    gaming_auto: 'Gaming / Automatico',
    gaming_hook: 'Gaming / Hook',
    gaming_highlights: 'Gaming / Highlights',
  }
  return labels[mode] ?? mode
}
