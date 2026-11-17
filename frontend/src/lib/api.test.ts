import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { api } from "./api"
import type { EnhancementTask, MediaOptions } from "./types"

describe('Contratos API Media Foundation', () => {
  const media: MediaOptions = {
    destination_profile: 'instagram_reels', color_handling: 'auto', stabilization: 'light',
    video_denoise: 'medium', audio_cleanup: 'voice_strong',
  }
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true, headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => ({ id: 'job', status: 'queued' }),
    }))
  })
  afterEach(() => { vi.unstubAllGlobals() })

  it('usa endpoint de diagnóstico y refresh=true explícito', async () => {
    await api.getMediaCapabilities()
    expect(fetch).toHaveBeenLastCalledWith('/api/system/media-capabilities', undefined)
    await api.getMediaCapabilities(true)
    expect(fetch).toHaveBeenLastCalledWith('/api/system/media-capabilities?refresh=true', undefined)
  })

  it('serializa opciones snake_case en JSON para URL', async () => {
    await api.createFromUrl('https://youtube.com/watch?v=test', { media_options: media })
    const [url, init] = vi.mocked(fetch).mock.calls[0]
    expect(url).toBe('/api/jobs/url')
    expect(JSON.parse(init?.body as string)).toEqual({ url: 'https://youtube.com/watch?v=test', media_options: media })
  })

  it('serializa media_options como JSON en FormData para archivo', async () => {
    await api.createFromFile(new File(['video'], 'video.mp4'), { media_options: media })
    const [url, init] = vi.mocked(fetch).mock.calls[0]
    expect(url).toBe('/api/jobs')
    expect(init?.body).toBeInstanceOf(FormData)
    expect(JSON.parse((init?.body as FormData).get('media_options') as string)).toEqual(media)
  })

  it('conserva compatibilidad de payloads legacy sin media_options', async () => {
    await api.createFromUrl('https://youtube.com/watch?v=test')
    expect(JSON.parse(vi.mocked(fetch).mock.calls[0][1]?.body as string)).not.toHaveProperty('media_options')
    await api.createFromFile(new File(['video'], 'video.mp4'))
    expect((vi.mocked(fetch).mock.calls[1][1]?.body as FormData).has('media_options')).toBe(false)
  })
})

describe("waitWithAbort", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("resolves after the specified timeout", async () => {
    const setTimeoutSpy = vi.spyOn(global, "setTimeout")
    const promise = api.waitWithAbort(1000)
    expect(setTimeoutSpy).toHaveBeenCalledWith(expect.any(Function), 1000)

    await vi.advanceTimersByTimeAsync(1000)

    await expect(promise).resolves.toBeUndefined()
    setTimeoutSpy.mockRestore()
  })

  it("rejects when AbortSignal is triggered before timeout", async () => {
    const controller = new AbortController()
    const promise = api.waitWithAbort(1000, controller.signal)

    controller.abort()

    await expect(promise).rejects.toThrow("Polling abortado")
  })

  it("rejects immediately if signal is already aborted", async () => {
    const controller = new AbortController()
    controller.abort()

    const promise = api.waitWithAbort(1000, controller.signal)

    await expect(promise).rejects.toThrow("Polling abortado")
  })

  it("cleans up listener on normal resolution", async () => {
    const controller = new AbortController()
    const promise = api.waitWithAbort(1000, controller.signal)

    await vi.advanceTimersByTimeAsync(1000)

    await expect(promise).resolves.toBeUndefined()
    // The abort listener should have been removed - no error on subsequent abort
    controller.abort()
  })

  it("cleans up listener on abort", async () => {
    const controller = new AbortController()
    const promise = api.waitWithAbort(1000, controller.signal)

    controller.abort()

    await expect(promise).rejects.toThrow("Polling abortado")
    // Timer should have been cleared - advancing timers shouldn't cause issues
    await vi.advanceTimersByTimeAsync(2000)
  })

  it("adds and removes abort listener correctly", async () => {
    const controller = new AbortController()
    const addListenerSpy = vi.spyOn(controller.signal, "addEventListener")
    const removeListenerSpy = vi.spyOn(controller.signal, "removeEventListener")

    const promise = api.waitWithAbort(1000, controller.signal)

    // Listener should be added
    expect(addListenerSpy).toHaveBeenCalledWith("abort", expect.any(Function), { once: true })

    await vi.advanceTimersByTimeAsync(1000)
    await promise

    // Listener should be removed on normal resolution
    expect(removeListenerSpy).toHaveBeenCalledWith("abort", expect.any(Function))

    addListenerSpy.mockRestore()
    removeListenerSpy.mockRestore()
  })

  it("removes abort listener on abort", async () => {
    const controller = new AbortController()
    const addListenerSpy = vi.spyOn(controller.signal, "addEventListener")
    const removeListenerSpy = vi.spyOn(controller.signal, "removeEventListener")

    const promise = api.waitWithAbort(1000, controller.signal)

    controller.abort()

    await expect(promise).rejects.toThrow("Polling abortado")

    // Listener should be removed on abort
    expect(removeListenerSpy).toHaveBeenCalledWith("abort", expect.any(Function))

    addListenerSpy.mockRestore()
    removeListenerSpy.mockRestore()
  })

  it("does not add listener if signal is already aborted", async () => {
    const controller = new AbortController()
    controller.abort()
    const addListenerSpy = vi.spyOn(controller.signal, "addEventListener")

    const promise = api.waitWithAbort(1000, controller.signal)

    await expect(promise).rejects.toThrow("Polling abortado")

    // Listener should NOT be added if signal already aborted
    expect(addListenerSpy).not.toHaveBeenCalled()

    addListenerSpy.mockRestore()
  })
})

describe("waitForEnhancementTask", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("polling lifecycle: QUEUED -> RUNNING -> COMPLETED", async () => {
    const mockTask: EnhancementTask = {
      id: "task123",
      project_id: "proj1",
      kind: "ENHANCEMENT",
      status: "QUEUED",
      provider: "ffmpeg",
      model: "",
      request: { profile: "DEFAULT", source_asset_id: "asset1" },
      request_hash: "hash123",
      cost_estimate: {
        cost_class: "LOCAL",
        monetary_cost_known: true,
        monetary_cost_usd: 0,
        estimated_seconds: 10,
        notes: "local FFmpeg",
      },
      approval_state: "APPROVED",
      approved_request_hash: "hash123",
      progress: 0,
      output_asset_id: null,
      error: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    let fetchCallCount = 0
    const originalFetch = global.fetch
    global.fetch = vi.fn().mockImplementation(async () => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return {
          ok: true,
          headers: { get: () => "application/json" },
          json: async () => ({ ...mockTask, status: "QUEUED" }),
        }
      }
      if (fetchCallCount === 2) {
        return {
          ok: true,
          headers: { get: () => "application/json" },
          json: async () => ({ ...mockTask, status: "RUNNING", progress: 0.5 }),
        }
      }
      return {
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({ ...mockTask, status: "COMPLETED", progress: 1.0, output_asset_id: "asset2" }),
      }
    })

    const promise = api.waitForEnhancementTask("proj1", "task123", {
      intervalMs: 100,
      timeoutMs: 5000,
    })

    await vi.advanceTimersByTimeAsync(100)
    await vi.advanceTimersByTimeAsync(100)
    await vi.advanceTimersByTimeAsync(100)

    const result = await promise
    expect(result.status).toBe("COMPLETED")
    expect(fetchCallCount).toBe(3)

    global.fetch = originalFetch
  })

  it("handles FAILED status", async () => {
    const mockTask: EnhancementTask = {
      id: "task1",
      project_id: "proj1",
      kind: "ENHANCEMENT",
      status: "QUEUED",
      provider: "ffmpeg",
      model: "",
      request: { profile: "DEFAULT", source_asset_id: "asset1" },
      request_hash: "hash123",
      cost_estimate: {
        cost_class: "LOCAL",
        monetary_cost_known: true,
        monetary_cost_usd: 0,
        estimated_seconds: 10,
        notes: "local FFmpeg",
      },
      approval_state: "APPROVED",
      approved_request_hash: "hash123",
      progress: 0,
      output_asset_id: null,
      error: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    let fetchCallCount = 0
    const originalFetch = global.fetch
    global.fetch = vi.fn().mockImplementation(async () => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return {
          ok: true,
          headers: { get: () => "application/json" },
          json: async () => ({ ...mockTask, status: "QUEUED" }),
        }
      }
      return {
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({ ...mockTask, status: "FAILED", error: "FFmpeg error" }),
      }
    })

    const promise = api.waitForEnhancementTask("proj1", "task1", {
      intervalMs: 100,
      timeoutMs: 5000,
    })
    const rejection = expect(promise).rejects.toThrow("FFmpeg error")

    await vi.advanceTimersByTimeAsync(100)
    await vi.advanceTimersByTimeAsync(100)

    await rejection
    global.fetch = originalFetch
  })

  it("handles CANCELLED status", async () => {
    const mockTask: EnhancementTask = {
      id: "task1",
      project_id: "proj1",
      kind: "ENHANCEMENT",
      status: "QUEUED",
      provider: "ffmpeg",
      model: "",
      request: { profile: "DEFAULT", source_asset_id: "asset1" },
      request_hash: "hash123",
      cost_estimate: {
        cost_class: "LOCAL",
        monetary_cost_known: true,
        monetary_cost_usd: 0,
        estimated_seconds: 10,
        notes: "local FFmpeg",
      },
      approval_state: "APPROVED",
      approved_request_hash: "hash123",
      progress: 0,
      output_asset_id: null,
      error: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    let fetchCallCount = 0
    const originalFetch = global.fetch
    global.fetch = vi.fn().mockImplementation(async () => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return {
          ok: true,
          headers: { get: () => "application/json" },
          json: async () => ({ ...mockTask, status: "QUEUED" }),
        }
      }
      return {
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({ ...mockTask, status: "CANCELLED" }),
      }
    })

    const promise = api.waitForEnhancementTask("proj1", "task1", {
      intervalMs: 100,
      timeoutMs: 5000,
    })

    await vi.advanceTimersByTimeAsync(100)
    await vi.advanceTimersByTimeAsync(100)

    const result = await promise
    expect(result.status).toBe("CANCELLED")
    global.fetch = originalFetch
  })

  it("handles timeout", async () => {
    const mockTask: EnhancementTask = {
      id: "task1",
      project_id: "proj1",
      kind: "ENHANCEMENT",
      status: "QUEUED",
      provider: "ffmpeg",
      model: "",
      request: { profile: "DEFAULT", source_asset_id: "asset1" },
      request_hash: "hash123",
      cost_estimate: {
        cost_class: "LOCAL",
        monetary_cost_known: true,
        monetary_cost_usd: 0,
        estimated_seconds: 10,
        notes: "local FFmpeg",
      },
      approval_state: "APPROVED",
      approved_request_hash: "hash123",
      progress: 0,
      output_asset_id: null,
      error: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    const originalFetch = global.fetch
    global.fetch = vi.fn().mockImplementation(async () => {
      return {
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({ ...mockTask, status: "QUEUED" }),
      }
    })

    const promise = api.waitForEnhancementTask("proj1", "task1", {
      intervalMs: 100,
      timeoutMs: 500,
    })
    const rejection = expect(promise).rejects.toThrow("La tarea de mejora excedió el tiempo límite")

    await vi.advanceTimersByTimeAsync(500)

    await rejection
    global.fetch = originalFetch
  })

  it("abort during wait stops polling and cleans up", async () => {
    const mockTask: EnhancementTask = {
      id: "task1",
      project_id: "proj1",
      kind: "ENHANCEMENT",
      status: "QUEUED",
      provider: "ffmpeg",
      model: "",
      request: { profile: "DEFAULT", source_asset_id: "asset1" },
      request_hash: "hash123",
      cost_estimate: {
        cost_class: "LOCAL",
        monetary_cost_known: true,
        monetary_cost_usd: 0,
        estimated_seconds: 10,
        notes: "local FFmpeg",
      },
      approval_state: "APPROVED",
      approved_request_hash: "hash123",
      progress: 0,
      output_asset_id: null,
      error: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    const controller = new AbortController()
    let fetchCallCount = 0
    const originalFetch = global.fetch
    global.fetch = vi.fn().mockImplementation(async () => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return {
          ok: true,
          headers: { get: () => "application/json" },
          json: async () => ({ ...mockTask, status: "QUEUED" }),
        }
      }
      return {
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({ ...mockTask, status: "RUNNING" }),
      }
    })

    const promise = api.waitForEnhancementTask("proj1", "task1", {
      intervalMs: 100,
      timeoutMs: 5000,
      signal: controller.signal,
    })

    await vi.advanceTimersByTimeAsync(100)

    controller.abort()

    await expect(promise).rejects.toThrow("Polling abortado")
    // No more fetches should happen after abort
    await vi.advanceTimersByTimeAsync(500)
    // Initial fetch + one polling fetch before abort
    expect(fetchCallCount).toBe(2)

    global.fetch = originalFetch
  })

  it("abort before first fetch rejects immediately and makes NO fetch", async () => {
    const controller = new AbortController()
    controller.abort()

    const originalFetch = global.fetch
    global.fetch = vi.fn()

    const promise = api.waitForEnhancementTask("proj1", "task1", {
      intervalMs: 100,
      timeoutMs: 5000,
      signal: controller.signal,
    })

    await expect(promise).rejects.toThrow("Polling abortado")
    // Should NOT have called fetch at all
    expect(global.fetch).not.toHaveBeenCalled()

    global.fetch = originalFetch
  })

  it("abort during fetch cancels and stops polling", async () => {
    const mockTask: EnhancementTask = {
      id: "task1",
      project_id: "proj1",
      kind: "ENHANCEMENT",
      status: "QUEUED",
      provider: "ffmpeg",
      model: "",
      request: { profile: "DEFAULT", source_asset_id: "asset1" },
      request_hash: "hash123",
      cost_estimate: {
        cost_class: "LOCAL",
        monetary_cost_known: true,
        monetary_cost_usd: 0,
        estimated_seconds: 10,
        notes: "local FFmpeg",
      },
      approval_state: "APPROVED",
      approved_request_hash: "hash123",
      progress: 0,
      output_asset_id: null,
      error: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    const controller = new AbortController()
    let fetchCallCount = 0
    let resolveFetch: (value: { ok: boolean; headers: { get: (name: string) => string }; json: () => Promise<EnhancementTask> }) => void
    const fetchPromise = new Promise((resolve) => { resolveFetch = resolve })
    
    const originalFetch = global.fetch
    global.fetch = vi.fn().mockImplementation(async () => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return {
          ok: true,
          headers: { get: () => "application/json" },
          json: async () => ({ ...mockTask, status: "QUEUED" }),
        }
      }
      // Second fetch - return a promise we can control
      return fetchPromise
    })

    const promise = api.waitForEnhancementTask("proj1", "task1", {
      intervalMs: 100,
      timeoutMs: 5000,
      signal: controller.signal,
    })

    // Wait for first fetch to complete
    await vi.advanceTimersByTimeAsync(100)
    
    // Abort during the second fetch
    controller.abort()
    
    // Resolve the fetch to allow cleanup
    resolveFetch!({
      ok: true,
      headers: { get: () => "application/json" },
      json: async () => ({ ...mockTask, status: "RUNNING" }),
    })

    await expect(promise).rejects.toThrow("Polling abortado")
    // No more polling after abort
    await vi.advanceTimersByTimeAsync(500)
    
    global.fetch = originalFetch
  })

  it("stops polling on COMPLETED terminal status", async () => {
    const mockTask: EnhancementTask = {
      id: "task1",
      project_id: "proj1",
      kind: "ENHANCEMENT",
      status: "QUEUED",
      provider: "ffmpeg",
      model: "",
      request: { profile: "DEFAULT", source_asset_id: "asset1" },
      request_hash: "hash123",
      cost_estimate: {
        cost_class: "LOCAL",
        monetary_cost_known: true,
        monetary_cost_usd: 0,
        estimated_seconds: 10,
        notes: "local FFmpeg",
      },
      approval_state: "APPROVED",
      approved_request_hash: "hash123",
      progress: 0,
      output_asset_id: null,
      error: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    let fetchCallCount = 0
    const originalFetch = global.fetch
    global.fetch = vi.fn().mockImplementation(async () => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return {
          ok: true,
          headers: { get: () => "application/json" },
          json: async () => ({ ...mockTask, status: "QUEUED" }),
        }
      }
      return {
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({ ...mockTask, status: "COMPLETED", progress: 1.0 }),
      }
    })

    const promise = api.waitForEnhancementTask("proj1", "task1", {
      intervalMs: 100,
      timeoutMs: 5000,
    })

    await vi.advanceTimersByTimeAsync(100)
    await vi.advanceTimersByTimeAsync(100)

    const result = await promise
    expect(result.status).toBe("COMPLETED")
    expect(fetchCallCount).toBe(2) // Initial fetch + one polling fetch

    global.fetch = originalFetch
  })

  it("stops polling on FAILED terminal status", async () => {
    const mockTask: EnhancementTask = {
      id: "task1",
      project_id: "proj1",
      kind: "ENHANCEMENT",
      status: "QUEUED",
      provider: "ffmpeg",
      model: "",
      request: { profile: "DEFAULT", source_asset_id: "asset1" },
      request_hash: "hash123",
      cost_estimate: {
        cost_class: "LOCAL",
        monetary_cost_known: true,
        monetary_cost_usd: 0,
        estimated_seconds: 10,
        notes: "local FFmpeg",
      },
      approval_state: "APPROVED",
      approved_request_hash: "hash123",
      progress: 0,
      output_asset_id: null,
      error: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    let fetchCallCount = 0
    const originalFetch = global.fetch
    global.fetch = vi.fn().mockImplementation(async () => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return {
          ok: true,
          headers: { get: () => "application/json" },
          json: async () => ({ ...mockTask, status: "QUEUED" }),
        }
      }
      return {
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({ ...mockTask, status: "FAILED", error: "failed" }),
      }
    })

    const promise = api.waitForEnhancementTask("proj1", "task1", {
      intervalMs: 100,
      timeoutMs: 5000,
    })
    const rejection = expect(promise).rejects.toThrow("failed")

    await vi.advanceTimersByTimeAsync(100)
    await vi.advanceTimersByTimeAsync(100)

    await rejection
    expect(fetchCallCount).toBe(2)

    global.fetch = originalFetch
  })

  it("stops polling on CANCELLED terminal status", async () => {
    const mockTask: EnhancementTask = {
      id: "task1",
      project_id: "proj1",
      kind: "ENHANCEMENT",
      status: "QUEUED",
      provider: "ffmpeg",
      model: "",
      request: { profile: "DEFAULT", source_asset_id: "asset1" },
      request_hash: "hash123",
      cost_estimate: {
        cost_class: "LOCAL",
        monetary_cost_known: true,
        monetary_cost_usd: 0,
        estimated_seconds: 10,
        notes: "local FFmpeg",
      },
      approval_state: "APPROVED",
      approved_request_hash: "hash123",
      progress: 0,
      output_asset_id: null,
      error: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    let fetchCallCount = 0
    const originalFetch = global.fetch
    global.fetch = vi.fn().mockImplementation(async () => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return {
          ok: true,
          headers: { get: () => "application/json" },
          json: async () => ({ ...mockTask, status: "QUEUED" }),
        }
      }
      return {
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({ ...mockTask, status: "CANCELLED" }),
      }
    })

    const promise = api.waitForEnhancementTask("proj1", "task1", {
      intervalMs: 100,
      timeoutMs: 5000,
    })

    await vi.advanceTimersByTimeAsync(100)
    await vi.advanceTimersByTimeAsync(100)

    const result = await promise
    expect(result.status).toBe("CANCELLED")
    expect(fetchCallCount).toBe(2)

    global.fetch = originalFetch
  })
})
