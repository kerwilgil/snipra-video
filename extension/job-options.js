// Construcción y validación del payload enviado a /api/jobs/url.
// Se mantiene libre de APIs de Chrome para poder probarlo también con Node.

(() => {
  'use strict'

  const OUTPUT_MODES = new Set(['clips', 'download', 'audio'])
  const GAMING_MODE_BY_OUTPUT = {
    auto: 'gaming_auto',
    clips: 'gaming',
    hook: 'gaming_hook',
    highlights: 'gaming_highlights',
  }

  function resolveProcessingMode(processingMode, gamingOutput) {
    if (processingMode !== 'gaming') return processingMode || 'auto_smart'
    return GAMING_MODE_BY_OUTPUT[gamingOutput] || GAMING_MODE_BY_OUTPUT.auto
  }

  function buildJobPayload(options) {
    const url = String(options.url || '').trim()
    const outputMode = String(options.outputMode || 'clips').trim().toLowerCase()
    if (!/^https?:\/\//i.test(url)) throw new Error('La pestaña activa no tiene una URL válida.')
    if (!OUTPUT_MODES.has(outputMode)) throw new Error('Modo de salida no soportado.')

    const payload = { url, output_mode: outputMode }

    if (outputMode === 'audio') {
      payload.audio_format = options.audioFormat || 'mp3'
      return payload
    }

    payload.download_quality = options.downloadQuality || 'auto'
    if (outputMode === 'download') return payload

    const formats = Array.isArray(options.formats)
      ? options.formats.filter((value) => ['9x16', '1x1', '16x9'].includes(value))
      : []
    if (formats.length === 0) throw new Error('Elige al menos un formato de salida.')

    const processingMode = resolveProcessingMode(options.processingMode, options.gamingOutput)
    const requestedClips = Number.parseInt(options.numClips, 10)
    const singleEdit = processingMode === 'gaming_hook' || processingMode === 'gaming_highlights'

    payload.processing_mode = processingMode
    payload.num_clips = singleEdit
      ? 1
      : Math.min(12, Math.max(1, Number.isFinite(requestedClips) ? requestedClips : 5))
    payload.formats = formats
    payload.captions_enabled = options.captionsEnabled !== false
    payload.editorial_options = {
      compact_editing: options.compactEditing === true,
      semantic_cards: options.semanticCards === true,
      privacy_assist: options.privacyAssist === true,
      protected_zones: true,
      overlay_qa: true,
      transcription_quality: ['default', 'fast', 'balanced', 'precise'].includes(options.transcriptionQuality)
        ? options.transcriptionQuality
        : 'default',
    }
    return payload
  }

  globalThis.SnipraJobOptions = Object.freeze({ buildJobPayload, resolveProcessingMode })
})()
