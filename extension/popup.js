// Snipra 1.0.0 - popup MV3 sin content scripts.
// Lee la pestaña activa y crea un trabajo en la API local con los ajustes elegidos.

'use strict'

const BACKEND_BASES = [
  'http://localhost:8000/api',
  'http://127.0.0.1:8000/api',
  'http://localhost:8001/api',
  'http://127.0.0.1:8001/api',
]
const FRONTEND_URL = 'http://localhost:5173'
const STORAGE = {
  token: 'extensionToken',
  backendBase: 'backendBase',
  outputMode: 'outputMode',
  downloadQuality: 'downloadQuality',
  audioFormat: 'audioFormat',
  processingMode: 'processingMode',
  gamingOutput: 'gamingOutput',
  numClips: 'numClips',
  formats: 'formats',
  captionsEnabled: 'captionsEnabled',
  compactEditing: 'compactEditing',
  semanticCards: 'semanticCards',
  privacyAssist: 'privacyAssist',
  transcriptionQuality: 'transcriptionQuality',
  activeJob: 'activeJob',
}

const LAUNCHER_NAME = /mac/i.test(navigator.userAgent) ? 'Snipra.app' : 'Snipra.exe'
const GAMING_HINTS = {
  auto: 'Snipra elegirá entre clips, hook o highlights según los eventos detectados.',
  clips: 'Generará varios videos separados con momentos completos.',
  hook: 'Generará un único edit de 8–25 segundos con acción y desenlace.',
  highlights: 'Generará un montaje de 30–45 segundos con varias jugadas breves.',
}

const healthState = document.getElementById('healthState')
const healthText = document.getElementById('healthText')
const urlEl = document.getElementById('url')
const statusEl = document.getElementById('status')
const jobForm = document.getElementById('jobForm')
const submitButton = document.getElementById('submitButton')
const outputTabs = [...document.querySelectorAll('[data-output]')]
const qualityPanel = document.getElementById('qualityPanel')
const qualitySelect = document.getElementById('qualitySelect')
const audioPanel = document.getElementById('audioPanel')
const audioFormatSelect = document.getElementById('audioFormatSelect')
const clipsPanel = document.getElementById('clipsPanel')
const processingModeSelect = document.getElementById('processingModeSelect')
const gamingPanel = document.getElementById('gamingPanel')
const gamingOutputSelect = document.getElementById('gamingOutputSelect')
const gamingHint = document.getElementById('gamingHint')
const numClipsInput = document.getElementById('numClipsInput')
const formatInputs = [...document.querySelectorAll('input[name="format"]')]
const captionsToggle = document.getElementById('captionsToggle')
const compactEditingToggle = document.getElementById('compactEditingToggle')
const semanticCardsToggle = document.getElementById('semanticCardsToggle')
const privacyAssistToggle = document.getElementById('privacyAssistToggle')
const transcriptionQualitySelect = document.getElementById('transcriptionQualitySelect')
const pairSection = document.getElementById('pairSection')
const btnPair = document.getElementById('btnPair')
const pairStatus = document.getElementById('pairStatus')
const connectedNote = document.getElementById('connectedNote')
const tokenSection = document.getElementById('tokenSection')
const tokenForm = document.getElementById('tokenForm')
const tokenInput = document.getElementById('tokenInput')
const openSnipraEl = document.getElementById('openSnipra')
const extensionVersionEl = document.getElementById('extensionVersion')

const detectionSection = document.getElementById('detectionSection')
const detectionStatus = document.getElementById('detectionStatus')
const detectionCandidates = document.getElementById('detectionCandidates')
const detectionDetails = document.getElementById('detectionDetails')
const downloadFile = document.getElementById('downloadFile')

let activeTabUrl = ''
let activeBackendBase = ''
let outputMode = 'clips'
let preferredNumClips = 5
let createdJobId = ''
let detectionContext = null
let actionBusy = false
let backendConnected = false
let monitoringJob = false
let pollTimer = null
let defaultInstagramDownload = false

function setStatus(kind, message) {
  statusEl.className = kind
  statusEl.textContent = message
}

function clearStatus() {
  statusEl.className = ''
  statusEl.textContent = ''
}

function renderVersions() {
  const extension = `Extensión ${chrome.runtime.getManifest().version}`
  extensionVersionEl.textContent = `${extension} · Solo local`
}

function setHealth(kind, message) {
  healthState.className = `health ${kind}`.trim()
  healthText.textContent = message
  healthState.title = message
}

function setBusy(busy) {
  actionBusy = busy
  updateActionState()
}

function updateActionState() {
  submitButton.disabled = actionBusy || monitoringJob || !activeTabUrl || !backendConnected || !detectionContext?.downloadable
  outputTabs.forEach((button) => { button.disabled = actionBusy || monitoringJob })
}

// Display page identity only. Keep the exact active URL exclusively for ingest;
// signed candidate URLs, query tokens and browser profile paths never enter UI.
function displayPageUrl(value) {
  try {
    const url = new URL(value)
    const videoId = url.hostname.endsWith('youtube.com') ? url.searchParams.get('v') : null
    return `${url.origin}${url.pathname}${videoId ? `?v=${encodeURIComponent(videoId)}` : ''}`
  } catch { return 'Página actual' }
}

function safeErrorMessage(value) {
  const message = typeof value === 'string' ? value : value?.message || ''
  if (/sesión|autentic|login|log in|cookie|dpapi/i.test(message)) {
    const platform = detectionContext?.platform === 'Instagram' ? 'Instagram' : 'Este sitio'
    return `${platform} necesita una sesión del navegador para acceder a este contenido. Revisa las cookies para sitios con sesión en Ajustes de Snipra.`
  }
  if (/no se pudo conectar|no está corriendo|fetch|network|timeout|abort/i.test(message)) {
    return `No se pudo conectar con Snipra. Abre ${LAUNCHER_NAME} y vuelve a intentarlo.`
  }
  return 'No se pudo completar la solicitud. Revisa el trabajo en Snipra y vuelve a intentarlo.'
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 1800) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, { ...options, signal: controller.signal, cache: 'no-store' })
  } finally {
    clearTimeout(timer)
  }
}

async function findBackend(force = false) {
  if (activeBackendBase && !force) return activeBackendBase

  const stored = await chrome.storage.local.get(STORAGE.backendBase)
  const preferred = stored[STORAGE.backendBase]
  const candidates = [...new Set([preferred, ...BACKEND_BASES].filter(base => BACKEND_BASES.includes(base)))]

  for (const base of candidates) {
    try {
      const response = await fetchWithTimeout(`${base}/health`, {}, 1400)
      if (!response.ok) continue
      const data = await response.json()
      if (data.app !== 'Snipra' || data.status !== 'ok') continue
      activeBackendBase = base
      backendConnected = true
      await chrome.storage.local.set({ [STORAGE.backendBase]: base })
      renderVersions()
      setHealth('ok', 'Snipra conectado')
      updateActionState()
      return base
    } catch {
      // Prueba el siguiente host/puerto local.
    }
  }

  activeBackendBase = ''
  backendConnected = false
  renderVersions()
  setHealth('error', 'Sin conexión')
  updateActionState()
  throw new Error('No se pudo conectar con Snipra.')
}

async function getStoredToken() {
  const stored = await chrome.storage.local.get(STORAGE.token)
  return stored[STORAGE.token] || ''
}

function renderConnectionState(token) {
  const connected = Boolean(token)
  pairSection.hidden = connected
  connectedNote.hidden = !connected
  tokenInput.value = ''
  tokenInput.placeholder = connected ? 'Token guardado — pega otro para reemplazarlo' : 'Token de extensión'
}

async function disconnectStaleToken() {
  await chrome.storage.local.remove(STORAGE.token)
  renderConnectionState('')
  tokenSection.open = true
}

function selectedFormats() {
  return formatInputs.filter((input) => input.checked).map((input) => input.value)
}

function setOutputMode(nextMode, persist = true) {
  outputMode = ['clips', 'download', 'audio'].includes(nextMode) ? nextMode : 'clips'
  outputTabs.forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.output === outputMode))
  })
  clipsPanel.hidden = outputMode !== 'clips'
  qualityPanel.hidden = outputMode === 'audio'
  audioPanel.hidden = outputMode !== 'audio'
  submitButton.textContent = {
    clips: 'Crear clips',
    download: 'Descargar video',
    audio: 'Extraer audio',
  }[outputMode]
  clearStatus()
  updateActionState()
  if (persist) chrome.storage.local.set({ [STORAGE.outputMode]: outputMode })
}

function renderGamingOptions() {
  const isGaming = processingModeSelect.value === 'gaming'
  const gamingOutput = gamingOutputSelect.value
  const singleEdit = isGaming && ['hook', 'highlights'].includes(gamingOutput)
  gamingPanel.hidden = !isGaming
  gamingHint.textContent = GAMING_HINTS[gamingOutput] || GAMING_HINTS.auto
  if (singleEdit) {
    numClipsInput.value = '1'
    numClipsInput.disabled = true
    numClipsInput.title = 'Hook y Highlights generan un solo video.'
  } else {
    numClipsInput.disabled = false
    numClipsInput.value = String(preferredNumClips)
    numClipsInput.title = ''
  }
}

function restorePreferences(values) {
  qualitySelect.value = values[STORAGE.downloadQuality] || 'auto'
  audioFormatSelect.value = values[STORAGE.audioFormat] || 'mp3'
  processingModeSelect.value = values[STORAGE.processingMode] || 'auto_smart'
  gamingOutputSelect.value = values[STORAGE.gamingOutput] || 'auto'
  preferredNumClips = Math.min(12, Math.max(1, Number(values[STORAGE.numClips]) || 5))
  numClipsInput.value = String(preferredNumClips)
  captionsToggle.checked = values[STORAGE.captionsEnabled] !== false
  compactEditingToggle.checked = values[STORAGE.compactEditing] === true
  semanticCardsToggle.checked = values[STORAGE.semanticCards] === true
  privacyAssistToggle.checked = values[STORAGE.privacyAssist] === true
  transcriptionQualitySelect.value = values[STORAGE.transcriptionQuality] || 'default'

  const formats = Array.isArray(values[STORAGE.formats]) && values[STORAGE.formats].length
    ? values[STORAGE.formats]
    : ['9x16']
  formatInputs.forEach((input) => { input.checked = formats.includes(input.value) })

  setOutputMode(values[STORAGE.outputMode] || 'clips', false)
  renderGamingOptions()
}

async function initActiveTab() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
    const tabUrl = tab?.url || ''
    if (!/^https?:\/\//i.test(tabUrl) || new URL(tabUrl).username || new URL(tabUrl).password) {
      urlEl.textContent = 'La pestaña activa no tiene una URL http(s) válida.'
      submitButton.disabled = true
      showDetectionError('URL de pestaña no válida')
      return
    }
    activeTabUrl = tabUrl
    urlEl.textContent = displayPageUrl(tabUrl)
    urlEl.title = displayPageUrl(tabUrl)

    if (detectionSection) detectionSection.hidden = false
    if (detectionStatus) detectionStatus.textContent = 'Inspeccionando página...'
    if (detectionCandidates) detectionCandidates.innerHTML = ''

    try {
      await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['video-detection.js'] })
      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => globalThis.SnipraVideoDetection.runDetection(),
      })
      const detection = results[0]?.result
      if (detection) {
        renderDetection(detection)
      } else {
        showDetectionFallback(tabUrl)
      }
    } catch {
      showDetectionFallback(tabUrl)
    }
  } catch {
    urlEl.textContent = 'No se pudo leer la pestaña activa.'
    submitButton.disabled = true
    showDetectionError('No se pudo acceder a la pestaña')
  }
}

function renderDetection(detection) {
  detectionContext = detection.context || { downloadable: false }
  const context = detectionContext
  detectionSection.hidden = false
  detectionCandidates.replaceChildren()
  if (context.authRequired) {
    detectionStatus.className = 'detection-status warn'
    detectionStatus.textContent = `${context.platform} · ${context.contentLabel} · Sesión requerida`
    detectionDetails.textContent = 'Instagram necesita una sesión del navegador para acceder a este contenido. Configura las cookies en Ajustes de Snipra.'
  } else if (context.videoDetected) {
    detectionStatus.className = 'detection-status ok'
    detectionStatus.textContent = `${context.platform} · ${context.contentLabel} · Video detectado`
    detectionDetails.textContent = context.authenticationMayBeRequired
      ? 'Story detectada. Snipra necesita una sesión con acceso; utiliza las cookies configuradas en Ajustes.'
      : 'La extensión envía la URL de esta pestaña. Snipra descarga y procesa el video.'
    if (context.platform === 'Instagram' && !defaultInstagramDownload) {
      setOutputMode('download', false)
      defaultInstagramDownload = true
    }
  } else {
    detectionStatus.className = 'detection-status neutral'
    detectionStatus.textContent = `${context.platform} · ${context.contentLabel}`
    detectionDetails.textContent = 'No se detectó un video descargable en esta página.'
  }
  updateActionState()
}

function showDetectionFallback() {
  detectionContext = { downloadable: false }
  detectionSection.hidden = false
  detectionStatus.textContent = 'No se pudo inspeccionar esta página'
  detectionStatus.className = 'detection-status warn'
  detectionDetails.textContent = 'Abre la página del video y vuelve a abrir la extensión.'
  detectionCandidates.replaceChildren()
  updateActionState()
}

function showDetectionError(message) {
  detectionContext = { downloadable: false }
  detectionSection.hidden = false
  detectionStatus.textContent = 'Página no disponible'
  detectionStatus.className = 'detection-status error'
  detectionDetails.textContent = message
  detectionCandidates.replaceChildren()
  updateActionState()
}

async function monitorJob(job) {
  if (!job || !/^[a-f0-9]{32}$/.test(job.id) || !BACKEND_BASES.includes(job.base)) return
  clearTimeout(pollTimer)
  monitoringJob = true
  updateActionState()
  try {
    const response = await fetchWithTimeout(`${job.base}/jobs/${job.id}`, {}, 4000)
    if (!response.ok) throw new Error('Respuesta no disponible')
    const data = await response.json()
    if (data.source_name !== activeTabUrl) {
      monitoringJob = false
      updateActionState()
      return
    }
    createdJobId = job.id
    openSnipraEl.textContent = 'Ver trabajo ↗'
    if (data.status === 'done') {
      monitoringJob = false
      setStatus('success', job.outputMode === 'clips' ? 'Clips completos. Puedes verlos en Snipra.' : 'Descarga completa. Tu archivo está listo.')
      if (['download', 'audio'].includes(job.outputMode)) {
        downloadFile.href = `${job.base}/jobs/${job.id}/download`
        downloadFile.hidden = false
        downloadFile.setAttribute('download', '')
      }
    } else if (['failed', 'canceled'].includes(data.status)) {
      monitoringJob = false
      setStatus('error', data.status === 'canceled' ? 'Trabajo cancelado en Snipra.' : safeErrorMessage(data.error))
    } else {
      const progress = Math.round(Math.min(1, Math.max(0, Number(data.progress) || 0)) * 100)
      setStatus('info', data.status === 'queued' ? 'Trabajo aceptado. Esperando a Snipra…' : `Snipra está ${job.outputMode === 'clips' ? 'procesando' : 'descargando'}… ${progress}%`)
      pollTimer = setTimeout(() => monitorJob(job), 1000)
    }
  } catch {
    monitoringJob = false
    setStatus('error', 'No se pudo conectar con Snipra. El trabajo puede seguir procesándose; vuelve a abrir la extensión para consultar su estado.')
  }
  updateActionState()
}

window.addEventListener('unload', () => clearTimeout(pollTimer))

async function sendJob() {
  if (!activeTabUrl || actionBusy || monitoringJob || !detectionContext?.downloadable) return
  setBusy(true)
  clearStatus()
  downloadFile.hidden = true

  const token = await getStoredToken()
  if (!token) {
    setStatus('error', 'Conecta la extensión con Snipra antes de crear un trabajo.')
    pairSection.hidden = false
    setBusy(false)
    return
  }

  let payload
  try {
    payload = globalThis.SnipraJobOptions.buildJobPayload({
      url: activeTabUrl,
      outputMode,
      downloadQuality: qualitySelect.value,
      audioFormat: audioFormatSelect.value,
      processingMode: processingModeSelect.value,
      gamingOutput: gamingOutputSelect.value,
      numClips: numClipsInput.value,
      formats: selectedFormats(),
      captionsEnabled: captionsToggle.checked,
      compactEditing: compactEditingToggle.checked,
      semanticCards: semanticCardsToggle.checked,
      privacyAssist: privacyAssistToggle.checked,
      transcriptionQuality: transcriptionQualitySelect.value,
    })
  } catch (error) {
    setStatus('error', safeErrorMessage(error.message))
    setBusy(false)
    return
  }

  setStatus('info', 'Enviando a Snipra...')

  try {
    // Revalida el backend antes del POST. El POST no se reintenta automáticamente:
    // si la conexión se corta después de que Snipra lo recibe, repetirlo podría
    // crear el mismo trabajo dos veces.
    const base = await findBackend(true)
    const response = await fetchWithTimeout(`${base}/jobs/url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Snipra-Extension-Token': token,
      },
      body: JSON.stringify(payload),
    }, 10000)

    if (response.status === 403) {
      await disconnectStaleToken()
      setStatus('error', 'La conexión venció o el token cambió. Vuelve a conectar la extensión.')
      return
    }

    if (!response.ok) {
      let detail = response.statusText
      try {
        const body = await response.json()
        detail = body.detail || detail
      } catch {
        // El cuerpo no era JSON.
      }
      setStatus('error', safeErrorMessage(detail))
      return
    }

    const data = await response.json()
    createdJobId = data.id
    if (!/^[a-f0-9]{32}$/.test(createdJobId)) throw new Error('Respuesta inválida')
    await chrome.storage.local.set({ [STORAGE.activeJob]: { id: createdJobId, base, outputMode } })
    setStatus('success', {
      clips: 'Trabajo creado. Snipra está detectando los mejores momentos.',
      download: 'Descarga de video encolada en Snipra.',
      audio: 'Extracción de audio encolada en Snipra.',
    }[outputMode])
    openSnipraEl.textContent = 'Ver trabajo ↗'
    await monitorJob({ id: createdJobId, base, outputMode })
  } catch (error) {
    setStatus('error', safeErrorMessage(error.message))
  } finally {
    setBusy(false)
  }
}

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'local' || !(STORAGE.token in changes)) return
  const token = changes[STORAGE.token].newValue || ''
  renderConnectionState(token)
  if (token) {
    pairStatus.textContent = 'Conectado ✓'
    setStatus('success', 'Extensión conectada con Snipra.')
  }
})

btnPair.addEventListener('click', async () => {
  btnPair.disabled = true
  pairStatus.textContent = 'Abriendo Snipra para aprobar la conexión...'
  try {
    const response = await chrome.runtime.sendMessage({ type: 'start-pairing' })
    if (!response?.ok) throw new Error(response?.error || 'No se pudo iniciar la conexión.')
    pairStatus.textContent = 'Aprueba la conexión en la pestaña que se abrió.'
  } catch (error) {
    pairStatus.textContent = safeErrorMessage(error.message)
  } finally {
    btnPair.disabled = false
  }
})

outputTabs.forEach((button) => {
  button.addEventListener('click', () => setOutputMode(button.dataset.output))
})

qualitySelect.addEventListener('change', () => chrome.storage.local.set({ [STORAGE.downloadQuality]: qualitySelect.value }))
audioFormatSelect.addEventListener('change', () => chrome.storage.local.set({ [STORAGE.audioFormat]: audioFormatSelect.value }))
processingModeSelect.addEventListener('change', () => {
  chrome.storage.local.set({ [STORAGE.processingMode]: processingModeSelect.value })
  renderGamingOptions()
})
gamingOutputSelect.addEventListener('change', () => {
  chrome.storage.local.set({ [STORAGE.gamingOutput]: gamingOutputSelect.value })
  renderGamingOptions()
})
numClipsInput.addEventListener('change', () => {
  preferredNumClips = Math.min(12, Math.max(1, Number(numClipsInput.value) || 5))
  numClipsInput.value = String(preferredNumClips)
  chrome.storage.local.set({ [STORAGE.numClips]: preferredNumClips })
})
formatInputs.forEach((input) => {
  input.addEventListener('change', () => chrome.storage.local.set({ [STORAGE.formats]: selectedFormats() }))
})
captionsToggle.addEventListener('change', () => chrome.storage.local.set({ [STORAGE.captionsEnabled]: captionsToggle.checked }))
compactEditingToggle.addEventListener('change', () => chrome.storage.local.set({ [STORAGE.compactEditing]: compactEditingToggle.checked }))
semanticCardsToggle.addEventListener('change', () => chrome.storage.local.set({ [STORAGE.semanticCards]: semanticCardsToggle.checked }))
privacyAssistToggle.addEventListener('change', () => chrome.storage.local.set({ [STORAGE.privacyAssist]: privacyAssistToggle.checked }))
transcriptionQualitySelect.addEventListener('change', () => chrome.storage.local.set({ [STORAGE.transcriptionQuality]: transcriptionQualitySelect.value }))

jobForm.addEventListener('submit', async (event) => {
  event.preventDefault()
  await sendJob()
})

tokenForm.addEventListener('submit', async (event) => {
  event.preventDefault()
  const value = tokenInput.value.trim()
  if (!value) {
    setStatus('error', 'Pega un token válido o usa “Conectar con Snipra”.')
    return
  }
  await chrome.storage.local.set({ [STORAGE.token]: value })
  tokenSection.open = false
  tokenInput.value = ''
  setStatus('success', 'Token guardado en la extensión.')
})

openSnipraEl.addEventListener('click', (event) => {
  event.preventDefault()
  const url = createdJobId ? `${FRONTEND_URL}/jobs/${createdJobId}` : FRONTEND_URL
  chrome.tabs.create({ url })
})

async function init() {
  renderVersions()
  const keys = Object.values(STORAGE)
  const values = await chrome.storage.local.get(keys)
  activeBackendBase = values[STORAGE.backendBase] || ''
  restorePreferences(values)
  renderConnectionState(values[STORAGE.token] || '')
  await initActiveTab()
  try {
    await findBackend(true)
  } catch {
    setStatus('error', `No se pudo conectar con Snipra. Abre ${LAUNCHER_NAME} y vuelve a abrir la extensión.`)
  }
  if (backendConnected && values[STORAGE.activeJob]) await monitorJob(values[STORAGE.activeJob])
}

init()
