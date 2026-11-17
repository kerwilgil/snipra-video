'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { JSDOM } = require('jsdom')

const root = path.resolve(__dirname, '..')
const detectionScript = fs.readFileSync(path.join(root, 'video-detection.js'), 'utf8')
const popupScript = fs.readFileSync(path.join(root, 'popup.js'), 'utf8')
const optionsScript = fs.readFileSync(path.join(root, 'job-options.js'), 'utf8')
const popupHtml = fs.readFileSync(path.join(root, 'popup.html'), 'utf8')
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.json'), 'utf8'))
const reelUrl = 'https://www.instagram.com/reel/TEST_REEL/'
const postUrl = 'https://www.instagram.com/p/TEST_POST/'
const storyUrl = 'https://www.instagram.com/stories/example/123456789/'
const videoHtml = '<video src="blob:https://www.instagram.com/fixture"></video>'
const jobId = 'a'.repeat(32)

test('Los toggles de opciones tienen el nombre accesible visible correcto', () => {
  const dom = new JSDOM(popupHtml)
  const names = {
    captionsToggle: 'Subtítulos quemados', compactEditingToggle: 'Edición compacta',
    semanticCardsToggle: 'Tarjetas verificadas', privacyAssistToggle: 'Privacidad por OCR',
  }
  for (const [id, name] of Object.entries(names)) {
    const input = dom.window.document.getElementById(id)
    const label = dom.window.document.getElementById(input.getAttribute('aria-labelledby'))
    assert.equal(label?.textContent.trim(), name)
  }
  dom.window.close()
})

function detect(url, html = '') {
  const dom = new JSDOM(html, { url, runScripts: 'outside-only' })
  dom.window.eval(detectionScript)
  const result = dom.window.SnipraVideoDetection.runDetection()
  dom.window.close()
  return result
}

for (const host of ['instagram.com', 'www.instagram.com', 'm.instagram.com']) {
  for (const route of ['reel', 'reels']) {
    test(`Instagram ${host}/${route} reconoce Reel con carga diferida`, () => {
      const result = detect(`https://${host}/${route}/TEST_REEL/`)
      assert.equal(result.context.platform, 'Instagram')
      assert.equal(result.context.contentLabel, 'Reel')
      assert.equal(result.context.videoDetected, true)
      assert.equal(result.context.downloadable, true)
    })
  }
}

test('Instagram Post con video blob usa el contexto; el blob continúa unusable', () => {
  const result = detect(postUrl, videoHtml)
  assert.equal(result.context.contentLabel, 'Publicación con video')
  assert.equal(result.context.downloadable, true)
  assert.equal(result.candidates[0].usable, false)
  assert.equal(result.primaryCandidate, null)
})

for (const html of ['<video></video>', '<meta property="og:video" content="https://cdn.example/video.mp4">', '<script type="application/ld+json">{"@type":"VideoObject","contentUrl":"https://cdn.example/video.mp4"}</script>']) {
  test(`Post reconoce evidencia ${html.startsWith('<video') ? 'HTML5' : html.startsWith('<meta') ? 'OpenGraph' : 'JSON-LD'}`, () => {
    assert.equal(detect(postUrl, html).context.downloadable, true)
  })
}

test('Instagram Post sin video no ofrece descarga', () => {
  const result = detect(postUrl, '<img src="https://cdn.example/photo.jpg">')
  assert.equal(result.context.contentLabel, 'Publicación')
  assert.equal(result.context.videoDetected, false)
  assert.equal(result.context.downloadable, false)
})

test('Story reconoce contenido y necesidad potencial de sesión sin prometer acceso', () => {
  const result = detect(storyUrl, videoHtml)
  assert.equal(result.context.contentLabel, 'Story')
  assert.equal(result.context.authenticationMayBeRequired, true)
  assert.equal(result.context.videoDetected, true)
})

test('Story sin contenido detectado no activa descarga', () => {
  assert.equal(detect(storyUrl).context.downloadable, false)
})

test('Página de login de Instagram muestra autenticación requerida', () => {
  const result = detect(reelUrl, '<form action="/accounts/login"><input name="username"></form>')
  assert.equal(result.context.authRequired, true)
  assert.equal(result.context.downloadable, false)
})

for (const route of ['example/', 'explore/', 'explore/search/', '']) {
  test(`Instagram genérico /${route} neutral incluso con video recomendado`, () => {
    const result = detect(`https://www.instagram.com/${route}`, videoHtml)
    assert.equal(result.context.platform, 'Instagram')
    assert.equal(result.context.videoDetected, false)
    assert.equal(result.context.downloadable, false)
  })
}

test('Dominio que imita Instagram no se clasifica como Instagram', () => {
  assert.notEqual(detect('https://instagram.com.example/reel/TEST/').context.platform, 'Instagram')
})

for (const [platform, url, html] of [
  ['YouTube', 'https://www.youtube.com/watch?v=fixture', ''],
  ['YouTube', 'https://youtu.be/fixture', ''],
  ['Vimeo', 'https://vimeo.com/123456', ''],
  ['Página web', 'https://example.com/watch', '<video src="https://cdn.example/video.mp4"></video>'],
  ['Página web', 'https://www.tiktok.com/@example/video/123456', ''],
]) {
  test(`Regresión ${url} conserva la URL de página y la acción`, () => {
    const result = detect(url, html)
    assert.equal(result.context.platform, platform)
    assert.equal(result.context.downloadable, true)
    assert.equal(result.pageUrl, url)
  })
}

test('Reinyección puntual funciona al volver a abrir popup', () => {
  const dom = new JSDOM('', { url: reelUrl, runScripts: 'outside-only' })
  dom.window.eval(detectionScript)
  dom.window.eval(detectionScript)
  assert.equal(dom.window.SnipraVideoDetection.runDetection().context.contentType, 'reel')
  dom.window.close()
})

async function waitFor(predicate) {
  for (let i = 0; i < 100; i++) {
    if (predicate()) return
    await new Promise(resolve => setTimeout(resolve, 5))
  }
  assert.fail('El popup no alcanzó el estado esperado')
}

async function popup(t, { url = reelUrl, html = '', offline = false, responseStatus = 201,
  errorDetail = '', jobStatus = 'done', jobError = '', preferences = {}, storedJob = false,
  sourceName = url, injectionFails = false } = {}) {
  const dom = new JSDOM(popupHtml, { url: 'https://extension.example/popup.html', runScripts: 'outside-only', pretendToBeVisual: true })
  t.after(() => dom.window.close())
  const values = { extensionToken: 'NON_SECRET_TEST_FIXTURE', ...preferences }
  if (storedJob) values.activeJob = { id: jobId, base: 'http://localhost:8000/api', outputMode: 'download' }
  const calls = []
  const injections = []
  const storage = {
    get: async () => ({ ...values }),
    set: async data => Object.assign(values, data),
    remove: async key => { delete values[key] },
  }
  dom.window.chrome = {
    runtime: { getManifest: () => manifest, sendMessage: async () => ({ ok: true }) },
    storage: { local: storage, onChanged: { addListener() {} } },
    tabs: { query: async () => [{ id: 1, url }], create: async () => {} },
    scripting: { executeScript: async input => {
      injections.push(input)
      if (injectionFails) throw new Error('No se permite inspeccionar')
      return input.files ? [] : [{ result: detect(url, html) }]
    } },
  }
  dom.window.fetch = async (target, options = {}) => {
    if (offline) throw new Error('Failed to fetch')
    if (target.endsWith('/health')) return { ok: true, json: async () => ({ app: 'Snipra', status: 'ok', version: '1.6.0' }) }
    if (options.method === 'POST') {
      calls.push(JSON.parse(options.body))
      return { ok: responseStatus === 201, status: responseStatus,
        json: async () => responseStatus === 201 ? { id: jobId } : { detail: errorDetail } }
    }
    return { ok: true, json: async () => ({ id: jobId, source_name: sourceName, status: jobStatus, progress: 0.4, error: jobError }) }
  }
  dom.window.eval(optionsScript)
  dom.window.eval(popupScript)
  await waitFor(() => dom.window.document.getElementById('healthText').textContent !== 'Conectando...')
  return {
    window: dom.window, document: dom.window.document, calls, injections, values,
    submit: () => dom.window.document.getElementById('jobForm').dispatchEvent(new dom.window.Event('submit', { cancelable: true })),
    text: () => dom.window.document.body.textContent,
  }
}

test('Popup Reel muestra Instagram, CTA Descargar, conexión y versión 1.0.0', async t => {
  const p = await popup(t)
  assert.equal(p.document.getElementById('detectionStatus').textContent, 'Instagram · Reel · Video detectado')
  assert.equal(p.document.getElementById('submitButton').textContent, 'Descargar video')
  assert.equal(p.document.getElementById('submitButton').disabled, false)
  assert.equal(p.document.getElementById('healthText').textContent, 'Snipra conectado')
  assert.equal(p.document.getElementById('extensionVersion').textContent, 'Extensión 1.0.0 · Solo local')
  assert.equal(p.injections[0].files[0], 'video-detection.js')
})

test('CTA usa activeTabUrl exacta, nunca canonical/primaryCandidate/OG/CDN/manifest', async t => {
  const url = `${postUrl}?igsh=fixture`
  const p = await popup(t, { url, html: '<link rel="canonical" href="https://example.com/other"><video src="https://cdn.example/source.m3u8?token=NON_SECRET_TEST_FIXTURE"></video><meta property="og:video" content="https://cdn.example/video.mp4">' })
  p.submit()
  await waitFor(() => p.calls.length === 1 && !p.document.getElementById('downloadFile').hidden)
  assert.equal(p.calls[0].url, url)
  assert.equal(p.calls[0].output_mode, 'download')
  assert.equal(p.text().includes('cdn.example'), false)
  assert.equal(p.text().includes('NON_SECRET_TEST_FIXTURE'), false)
  assert.equal(p.document.getElementById('downloadFile').href, `http://localhost:8000/api/jobs/${jobId}/download`)
  assert.match(p.document.getElementById('status').textContent, /Descarga completa/)
})

test('Doble submit y pulsaciones durante progreso crean solamente un job', async t => {
  const p = await popup(t, { jobStatus: 'ingesting' })
  p.submit()
  p.submit()
  await waitFor(() => /40%/.test(p.document.getElementById('status').textContent))
  p.submit()
  await new Promise(resolve => setTimeout(resolve, 10))
  assert.equal(p.calls.length, 1)
  assert.equal(p.document.getElementById('submitButton').disabled, true)
  assert.equal(p.document.getElementById('downloadFile').hidden, true)
})

test('Publicación sin video desactiva CTA y rechaza submit programático', async t => {
  const p = await popup(t, { url: postUrl })
  assert.equal(p.document.getElementById('submitButton').disabled, true)
  assert.match(p.document.getElementById('detectionDetails').textContent, /No se detectó un video descargable/)
  p.submit()
  assert.equal(p.calls.length, 0)
})

test('Story con video muestra advertencia de sesión sin afirmar autenticación válida', async t => {
  const p = await popup(t, { url: storyUrl, html: videoHtml })
  assert.match(p.document.getElementById('detectionStatus').textContent, /Instagram · Story · Video detectado/)
  assert.match(p.document.getElementById('detectionDetails').textContent, /necesita una sesión con acceso/)
})

test('Backend offline se distingue del error de Instagram y desactiva CTA', async t => {
  const p = await popup(t, { offline: true })
  assert.match(p.document.getElementById('status').textContent, /No se pudo conectar con Snipra/)
  assert.equal(p.document.getElementById('submitButton').disabled, true)
  assert.match(p.document.getElementById('detectionStatus').textContent, /Instagram · Reel/)
})

test('AUTH REQUIRED sanitiza cookies, rutas de perfiles y stderr', async t => {
  const p = await popup(t, { jobStatus: 'failed', jobError: 'Could not copy Chrome cookie database C:\\Users\\PRIVATE_FIXTURE\\Profile; sessionid=NON_SECRET_TEST_FIXTURE' })
  p.submit()
  await waitFor(() => /necesita una sesión del navegador/.test(p.document.getElementById('status').textContent))
  assert.equal(p.text().includes('NON_SECRET_TEST_FIXTURE'), false)
  assert.equal(p.text().includes('PRIVATE_FIXTURE'), false)
  assert.equal(p.text().includes('cookie database'), false)
})

test('Error HTTP malicioso no expone Authorization, tokens ni URLs firmadas', async t => {
  const p = await popup(t, { responseStatus: 400, errorDetail: 'Authorization: Bearer NON_SECRET_TEST_FIXTURE https://cdn.example/media?access_token=NON_SECRET_TEST_FIXTURE' })
  p.submit()
  await waitFor(() => p.document.getElementById('status').className === 'error')
  assert.match(p.document.getElementById('status').textContent, /No se pudo completar/)
  assert.equal(p.text().includes('NON_SECRET_TEST_FIXTURE'), false)
  assert.equal(p.text().includes('cdn.example'), false)
})

test('URL de pestaña oculta query y fragmento sensibles sin cambiar ingest', async t => {
  const url = `${reelUrl}?access_token=NON_SECRET_TEST_FIXTURE#sessionid=NON_SECRET_TEST_FIXTURE`
  const p = await popup(t, { url })
  assert.equal(p.document.getElementById('url').textContent, reelUrl)
  assert.equal(p.document.getElementById('url').title, reelUrl)
  p.submit()
  await waitFor(() => p.calls.length === 1)
  assert.equal(p.calls[0].url, url)
  assert.equal(p.text().includes('NON_SECRET_TEST_FIXTURE'), false)
})

test('Popup retoma job de esta página después de cerrarse sin nuevo submit', async t => {
  const p = await popup(t, { storedJob: true })
  await waitFor(() => !p.document.getElementById('downloadFile').hidden)
  assert.equal(p.calls.length, 0)
  assert.match(p.document.getElementById('status').textContent, /Descarga completa/)
})

test('Job guardado de otra página no bloquea CTA ni muestra su archivo', async t => {
  const p = await popup(t, { storedJob: true, sourceName: postUrl })
  await waitFor(() => !p.document.getElementById('submitButton').disabled)
  assert.equal(p.document.getElementById('downloadFile').hidden, true)
})

test('Cancelación usa el estado canceled real del backend', async t => {
  const p = await popup(t, { storedJob: true, jobStatus: 'canceled' })
  await waitFor(() => /Trabajo cancelado/.test(p.document.getElementById('status').textContent))
  assert.equal(p.document.getElementById('submitButton').disabled, false)
})

test('Inspección bloqueada no inventa un video descargable', async t => {
  const p = await popup(t, { injectionFails: true })
  assert.equal(p.document.getElementById('submitButton').disabled, true)
  assert.match(p.document.getElementById('detectionStatus').textContent, /No se pudo inspeccionar/)
})

test('Backend guardado ajeno a localhost nunca recibe token o solicitud', async t => {
  const p = await popup(t, { preferences: { backendBase: 'https://external.example/api' } })
  assert.equal(p.values.backendBase, 'http://localhost:8000/api')
})

test('Manifest 1.0.0 describe relación con Snipra y conserva mínimo privilegio', () => {
  assert.equal(manifest.version, '1.0.0')
  assert.match(manifest.description, /Detecta videos.*envía su URL a Snipra/)
  assert.deepEqual(manifest.permissions, ['activeTab', 'storage', 'scripting'])
  assert.equal(manifest.content_scripts, undefined)
  assert.ok(manifest.host_permissions.every(value => /^http:\/\/(localhost|127\.0\.0\.1):800[01]\/\*$/.test(value)))
  for (const size of [16, 48, 128]) {
    assert.equal(manifest.icons[size], `icons/icon${size}.png`)
    assert.equal(manifest.action.default_icon[size], manifest.icons[size])
    assert.ok(fs.existsSync(path.join(root, manifest.icons[size])))
  }
})
