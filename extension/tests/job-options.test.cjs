const assert = require('node:assert/strict')
const test = require('node:test')

require('../job-options.js')

const { buildJobPayload, resolveProcessingMode } = globalThis.SnipraJobOptions

test('resuelve las cuatro variantes gaming', () => {
  assert.equal(resolveProcessingMode('gaming', 'auto'), 'gaming_auto')
  assert.equal(resolveProcessingMode('gaming', 'clips'), 'gaming')
  assert.equal(resolveProcessingMode('gaming', 'hook'), 'gaming_hook')
  assert.equal(resolveProcessingMode('gaming', 'highlights'), 'gaming_highlights')
  assert.equal(resolveProcessingMode('vlog', 'hook'), 'vlog')
})

test('clips conserva calidad, perfil, formatos y subtítulos', () => {
  assert.deepEqual(
    buildJobPayload({
      url: 'https://example.com/video',
      outputMode: 'clips',
      downloadQuality: '1440',
      processingMode: 'podcast_viral',
      gamingOutput: 'auto',
      numClips: '7',
      formats: ['9x16', '1x1'],
      captionsEnabled: false,
      compactEditing: true,
      semanticCards: true,
      privacyAssist: true,
      transcriptionQuality: 'precise',
    }),
    {
      url: 'https://example.com/video',
      output_mode: 'clips',
      download_quality: '1440',
      processing_mode: 'podcast_viral',
      num_clips: 7,
      formats: ['9x16', '1x1'],
      captions_enabled: false,
      editorial_options: {
        compact_editing: true,
        semantic_cards: true,
        privacy_assist: true,
        protected_zones: true,
        overlay_qa: true,
        transcription_quality: 'precise',
      },
    },
  )
})

test('hook y highlights fuerzan un único resultado', () => {
  for (const gamingOutput of ['hook', 'highlights']) {
    const payload = buildJobPayload({
      url: 'https://example.com/game',
      outputMode: 'clips',
      processingMode: 'gaming',
      gamingOutput,
      numClips: 9,
      formats: ['9x16'],
    })
    assert.equal(payload.num_clips, 1)
  }
})

test('descarga y audio solo envían las opciones que les corresponden', () => {
  assert.deepEqual(
    buildJobPayload({
      url: 'https://example.com/video',
      outputMode: 'download',
      downloadQuality: '2160',
    }),
    {
      url: 'https://example.com/video',
      output_mode: 'download',
      download_quality: '2160',
    },
  )
  assert.deepEqual(
    buildJobPayload({
      url: 'https://example.com/video',
      outputMode: 'audio',
      audioFormat: 'flac',
    }),
    {
      url: 'https://example.com/video',
      output_mode: 'audio',
      audio_format: 'flac',
    },
  )
})

test('rechaza una URL o formatos inválidos', () => {
  assert.throws(() => buildJobPayload({ url: 'chrome://settings', outputMode: 'clips' }))
  assert.throws(() =>
    buildJobPayload({
      url: 'https://example.com/video',
      outputMode: 'clips',
      formats: [],
    }),
  )
})
