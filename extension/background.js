// Snipra 1.0.0 - service worker de emparejamiento.
// Sobrevive al cierre del popup mientras el usuario aprueba la conexión local.

'use strict'

const BACKEND_BASES = [
  'http://localhost:8000/api',
  'http://127.0.0.1:8000/api',
  'http://localhost:8001/api',
  'http://127.0.0.1:8001/api',
]
const FRONTEND_URL = 'http://localhost:5173'
const TOKEN_STORAGE_KEY = 'extensionToken'
const BACKEND_STORAGE_KEY = 'backendBase'
const POLL_INTERVAL_MS = 1500
const MAX_POLL_MS = 5 * 60 * 1000

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== 'start-pairing') return undefined
  startPairing()
    .then((result) => sendResponse({ ok: true, ...result }))
    .catch((error) => sendResponse({ ok: false, error: String(error?.message || error) }))
  return true
})

async function backendCandidates() {
  const stored = await chrome.storage.local.get(BACKEND_STORAGE_KEY)
  return [...new Set([stored[BACKEND_STORAGE_KEY], ...BACKEND_BASES].filter(Boolean))]
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

async function createPairingRequest() {
  for (const base of await backendCandidates()) {
    try {
      // Verifica que el puerto pertenece a Snipra antes de enviar una mutación.
      // En equipos con varios servicios locales, 8000 puede estar ocupado por otra app.
      const healthResponse = await fetchWithTimeout(`${base}/health`, {}, 1400)
      if (!healthResponse.ok) continue
      const health = await healthResponse.json()
      if (health.app !== 'Snipra' || health.status !== 'ok') continue

      const response = await fetchWithTimeout(
        `${base}/extension/pair/request`,
        { method: 'POST' },
        5000,
      )
      if (!response.ok) continue
      const data = await response.json()
      if (!data.request_id) continue
      await chrome.storage.local.set({ [BACKEND_STORAGE_KEY]: base })
      return { base, requestId: data.request_id }
    } catch {
      // Prueba el siguiente host/puerto local.
    }
  }
  throw new Error('Snipra no está corriendo o rechazó la solicitud.')
}

async function startPairing() {
  const { base, requestId } = await createPairingRequest()
  await chrome.tabs.create({ url: `${FRONTEND_URL}/pair?request_id=${encodeURIComponent(requestId)}` })
  pollForToken(base, requestId)
  return { requestId }
}

async function pollForToken(base, requestId) {
  const deadline = Date.now() + MAX_POLL_MS
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS))
    try {
      const response = await fetchWithTimeout(
        `${base}/extension/pair/claim?request_id=${encodeURIComponent(requestId)}`,
        {},
        4000,
      )
      if (response.status === 404) return
      if (!response.ok) continue
      const data = await response.json()
      if (!data.pending && data.token) {
        await chrome.storage.local.set({ [TOKEN_STORAGE_KEY]: data.token })
        return
      }
    } catch {
      // Si Snipra se reinicia durante la aprobación, reintenta hasta el límite.
    }
  }
}
