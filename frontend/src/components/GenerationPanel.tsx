import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { Select } from "./Select";
import type {
  EditorProject,
  EditorAsset,
  GenerationTask,
  GenerationCapability,
  GenerationRequest,
  GenerationEstimate,
  GenerationMode,
  GenerationProvider,
  GenerationAspectRatio,
  GenerationResolution,
} from "../lib/types";

const MODE_LABELS: Record<GenerationMode, string> = {
  text_to_video: "Texto a Video",
  image_to_video: "Imagen a Video",
};

const MODE_DESCRIPTIONS: Record<GenerationMode, string> = {
  text_to_video: "Genera un video desde cero a partir de una descripción textual.",
  image_to_video: "Anima una imagen de referencia para crear un video.",
};

const PROVIDER_LABELS: Record<GenerationProvider, string> = {
  local_ltx: "LTX-Video (Local, Gratis)",
  runway: "Runway Gen-4.5 (Pago)",
  luma: "Luma Ray 3.2 (Pago)",
};

const PROVIDER_HINTS: Record<GenerationProvider, string[]> = {
  local_ltx: ["Coste API: $0", "Procesamiento: GPU Local", "Engine: LTX-Video 2B"],
  runway: ["Coste API: Por uso", "Procesamiento: Cloud", "Requiere API Key"],
  luma: ["Coste API: Por uso", "Procesamiento: Cloud", "Requiere API Key"],
};

const ASPECT_LABELS: Record<GenerationAspectRatio, string> = {
  "16:9": "16:9 (Horizontal)",
  "9:16": "9:16 (Vertical/Shorts)",
  "1:1": "1:1 (Cuadrado)",
  "4:3": "4:3",
  "3:4": "3:4",
};

const RESOLUTION_LABELS: Record<GenerationResolution, string> = {
  SD: "SD (720p)",
  HD: "HD (1080p)",
};

interface GenerationPanelProps {
  project: EditorProject;
  assets: EditorAsset[];
  onClose: () => void;
  onTaskCompleted?: (task: GenerationTask) => void;
  onInserted?: () => Promise<void>;
}

export function GenerationPanel({ project, assets, onClose, onTaskCompleted, onInserted }: GenerationPanelProps) {
  const [step, setStep] = useState<"capability" | "config" | "estimate" | "polling">("capability");
  const [capability, setCapability] = useState<GenerationCapability | null>(null);
  const [mode, setMode] = useState<GenerationMode>("text_to_video");
  const [provider, setProvider] = useState<GenerationProvider>("local_ltx");
  const derivedModel = provider === "local_ltx"
    ? "ltx-video-2b-distilled-fp8"
    : provider === "runway"
    ? "gen4.5"
    : "ray-3.2";
  const [promptIntent, setPromptIntent] = useState("");
  const [visualDirection, setVisualDirection] = useState("");
  const [cameraDirection, setCameraDirection] = useState("");
  const [avoidConstraints, setAvoidConstraints] = useState("");
  const [durationSeconds, setDurationSeconds] = useState(5);
  const [aspectRatio, setAspectRatio] = useState<GenerationAspectRatio>("16:9");
  const [resolution, setResolution] = useState<GenerationResolution>("SD");
  const [referenceAssetIds, setReferenceAssetIds] = useState<string[]>([]);
  const [groundingAssetIds, setGroundingAssetIds] = useState<string[]>([]);
  const [seed, setSeed] = useState<number | "">("");
  const [generationParameters] = useState<Record<string, unknown>>({});
  const [estimate, setEstimate] = useState<GenerationEstimate | null>(null);
  const [task, setTask] = useState<GenerationTask | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [requestHash, setRequestHash] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const pollingTaskIdRef = useRef<string | null>(null);
  const completionTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isLocalProvider = provider === "local_ltx";
  const isImageToVideo = mode === "image_to_video";
  const referenceAssets = assets.filter((a) => referenceAssetIds.includes(a.id));
  const groundingAssets = assets.filter((a) => groundingAssetIds.includes(a.id));
  const availableReferenceAssets = assets.filter((a) => a.kind === "video" || a.kind === "image");
  const availableGroundingAssets = assets.filter((a) => a.kind === "video" || a.kind === "audio" || a.kind === "text");

  // Fetch capability on mount
  useEffect(() => {
    api.getGenerationCapability(project.id)
      .then((cap) => {
        setCapability(cap);
        // Do NOT auto-advance to config step - wait for user to click "Continuar a configuración"
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : "No se pudo obtener la capacidad de generación");
      });
  }, [project.id]);

  const validateForm = useCallback(() => {
    if (!promptIntent.trim()) {
      setError("El prompt es obligatorio");
      return false;
    }
    if (isImageToVideo && referenceAssetIds.length === 0) {
      setError("El modo Imagen a Video requiere al menos un asset de referencia");
      return false;
    }
    if (durationSeconds < 1 || durationSeconds > 30) {
      setError("La duración debe estar entre 1 y 30 segundos");
      return false;
    }
    setError(null);
    return true;
  }, [promptIntent, isImageToVideo, referenceAssetIds.length, durationSeconds]);

  const prepareGeneration = useCallback(async () => {
    if (!validateForm()) return;
    setBusy(true);
    setError(null);
    try {
      const body: GenerationRequest = {
        mode,
        provider,
        model: derivedModel,
        prompt_intent: promptIntent,
        visual_direction: visualDirection || undefined,
        camera_direction: cameraDirection || undefined,
        avoid_constraints: avoidConstraints || undefined,
        duration_seconds: durationSeconds,
        aspect_ratio: aspectRatio,
        resolution: resolution,
        reference_asset_ids: referenceAssetIds,
        grounding_asset_ids: groundingAssetIds,
        seed: seed !== "" ? Number(seed) : undefined,
        generation_parameters: generationParameters,
      };
      const result = await api.prepareGeneration(project.id, body);
      setTask(result);
      setEstimate(result.cost_estimate);
      setRequestHash(result.request_hash);
      setStep("estimate");
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo preparar la generación");
    } finally {
      setBusy(false);
    }
  }, [project.id, mode, provider, derivedModel, promptIntent, visualDirection, cameraDirection, avoidConstraints, durationSeconds, aspectRatio, resolution, referenceAssetIds, groundingAssetIds, seed, generationParameters, validateForm]);

  const approveAndGenerate = useCallback(async () => {
    if (!task || !requestHash) return;
    setBusy(true);
    setError(null);
    try {
      const result = await api.approveGenerationTask(project.id, task.id, requestHash);
      setTask(result);
      pollingTaskIdRef.current = result.id;
      setStep("polling");
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo aprobar la generación");
    } finally {
      setBusy(false);
    }
  }, [project.id, task, requestHash]);

  const cancelTask = useCallback(async () => {
    if (!task) return;
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      pollingTaskIdRef.current = null;
    }
    setBusy(true);
    try {
      await api.cancelGenerationTask(project.id, task.id);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo cancelar");
    } finally {
      setBusy(false);
    }
  }, [project.id, task, onClose]);

  const insertGeneratedAsset = useCallback(async () => {
    if (!task || !task.output_asset || task.status !== "COMPLETED") return;

    const videoTrack = project.timeline.tracks.find(t => t.type === "video");
    if (!videoTrack) {
      setError("No hay pista de video en el proyecto. Crea una pista de video antes de insertar.");
      return;
    }

    if (!onInserted) {
      setError("No hay callback de refresh configurado. No se puede insertar.");
      return;
    }

    // Stable idempotency key for this logical attempt
    const attemptKey = `insert-${task.output_asset.id}-${task.id}`;
    setBusy(true);
    setError(null);

    try {
      await api.editorOperation(project.id, {
        type: "insert",
        parameters: {
          asset_id: task.output_asset.id,
          track_id: videoTrack.id,
          source_in_us: 0,
          source_out_us: task.output_asset.duration_us,
          timeline_start_us: project.timeline.duration_us,
        },
        base_version: project.current_version,
        idempotency_key: attemptKey,
      });
      // Wait for project refresh via callback
      await onInserted();
      // Close panel after successful insert and refresh
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo insertar en timeline");
    } finally {
      setBusy(false);
    }
  }, [project.id, project.current_version, project.timeline.duration_us, project.timeline.tracks, task, onInserted, onClose]);

  // Polling effect
  useEffect(() => {
    if (step !== "polling") return;
    const taskId = pollingTaskIdRef.current;
    if (!taskId) return;

    const controller = new AbortController();
    abortControllerRef.current = controller;
    const startedAt = Date.now();
    const TIMEOUT_MS = 60 * 60 * 1000;

    const poll = async () => {
      try {
        let current = await api.getGenerationTask(project.id, taskId);
        if (controller.signal.aborted || pollingTaskIdRef.current !== current.id) return;
        setTask(current);

        while (current.status === "QUEUED" || current.status === "RUNNING") {
          if (controller.signal.aborted) return;
          if (Date.now() > startedAt + TIMEOUT_MS) throw new Error("Tiempo límite excedido");
          await new Promise((resolve) => setTimeout(resolve, 2000));
          current = await api.getGenerationTask(project.id, taskId);
          if (controller.signal.aborted || pollingTaskIdRef.current !== current.id) return;
          setTask(current);
        }

        if (controller.signal.aborted || pollingTaskIdRef.current !== current.id) return;
        setTask(current);

        if (current.status === "COMPLETED") {
          onTaskCompleted?.(current);
        }
      } catch (e) {
        if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "Error en polling");
      }
    };

    poll();
    return () => {
      controller.abort();
      if (completionTimeoutRef.current) {
        clearTimeout(completionTimeoutRef.current);
        completionTimeoutRef.current = null;
      }
      abortControllerRef.current = null;
    };
  }, [step, project.id, onTaskCompleted]);

  const localCapability = capability?.local_ltx;
  const hardwareSupported = localCapability?.hardware_supported ?? false;
  const runtimeReady = localCapability?.runtime_ready ?? false;
  const hardwareMessage = localCapability?.hardware_message ?? "Comprobando capacidad...";
  const physicalValidation = localCapability?.physical_validation ?? "UNKNOWN";

  const renderCapabilityStep = () => (
    <div className="generation-panel">
      <h3>Generación de Video</h3>
      <div className="space-y-3">
        <div className={`capability-card ${hardwareSupported ? "supported" : "unsupported"}`}>
          <div className="capability-header">
            <span className="provider-badge local">LTX-Video Local</span>
            <span className={`status-badge ${hardwareSupported ? "ok" : "unavailable"}`}>
              {hardwareSupported ? "Compatible" : "No compatible"}
            </span>
          </div>
          <p className="text-sm">{hardwareMessage}</p>
          <div className="text-xs text-ink-mute">
            Validación física: <strong>{physicalValidation}</strong>
          </div>
          {!hardwareSupported && (
            <p className="text-xs text-bad mt-2">
              La generación local LTX requiere GPU NVIDIA con arquitectura Ada Lovelace (RTX 40-series) o Hopper (H100)+,
              CUDA disponible y 8GB+ VRAM. En esta máquina no se cumplen los requisitos.
            </p>
          )}
          {hardwareSupported && !runtimeReady && (
            <p className="text-xs text-amber mt-2">
              Hardware compatible, pero la ejecución física LTX aún no está habilitada.
            </p>
          )}
        </div>

        {capability?.cloud_providers.length && (
          <div>
            <h4 className="text-sm font-medium mb-2">Proveedores Cloud (Opcionales)</h4>
            <div className="space-y-2">
              {capability.cloud_providers.map((cp) => (
                <div key={cp.provider} className="capability-card cloud">
                  <div className="capability-header">
                    <span className="provider-badge cloud">{PROVIDER_LABELS[cp.provider as GenerationProvider] ?? cp.provider}</span>
                    <span className="status-badge paid">Pago</span>
                  </div>
                  <p className="text-xs text-ink-mute">
                    {cp.message || "Disponible en el contrato, pero backend cloud no habilitado en esta versión."}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="flex gap-2 pt-2">
          <button className="btn-grad flex-1" onClick={() => setStep("config")} disabled={busy}>
            {busy ? "Comprobando…" : "Continuar a configuración"}
          </button>
          <button className="btn-ghost" onClick={onClose} disabled={busy}>
            Cerrar
          </button>
        </div>
      </div>
    </div>
  );

  const renderConfigStep = () => (
    <div className="generation-panel">
      <h3>Configurar Generación</h3>
      <p className="text-xs text-ink-mute mb-3">
        Proveedor: <strong>{PROVIDER_LABELS[provider]}</strong> · Modelo: <strong>{derivedModel}</strong>
        {isLocalProvider && !runtimeReady && (
          <span className="text-bad ml-2">
            {hardwareSupported ? "(Hardware compatible, backend físico no habilitado)" : "(No disponible en este hardware)"}
          </span>
        )}
      </p>

      <div className="space-y-3">
        <label>
          <span className="text-sm">Modo</span>
          <Select
            value={mode}
            options={Object.entries(MODE_LABELS).map(([value, label]) => ({ value, label }))}
            onChange={(v) => setMode(v as GenerationMode)}
            ariaLabel="Modo de generación"
            disabled={busy}
          />
        </label>

        <p className="text-xs text-ink-mute">{MODE_DESCRIPTIONS[mode]}</p>

        <label>
          <span className="text-sm">Proveedor</span>
          <Select
            value={provider}
            options={Object.entries(PROVIDER_LABELS)
              .map(([value, label]) => ({ value, label }))
              .filter((opt) => {
                if (opt.value === "local_ltx") return true;
                const cap = capability?.cloud_providers?.find(cp => cp.provider === opt.value);
                return cap?.ready === true;
              })
            }
            onChange={(v) => setProvider(v as GenerationProvider)}
            ariaLabel="Proveedor de generación"
            disabled={busy}
          />
        </label>

        <div className="grid grid-cols-2 gap-2 text-xs text-cyan">
          {PROVIDER_HINTS[provider].map((hint, i) => (
            <span key={i}>• {hint}</span>
          ))}
        </div>

        {capability?.cloud_providers?.some(cp => !cp.ready) && (
          <p className="text-xs text-amber">
            Runway y Luma están definidos en el contrato, pero el backend cloud no está habilitado en esta versión.
          </p>
        )}

        <label>
          <span className="text-sm">Prompt / Intención</span>
          <textarea
            className="field"
            value={promptIntent}
            onChange={(e) => setPromptIntent(e.target.value)}
            placeholder="Describe el video que quieres generar…"
            rows={3}
            maxLength={4000}
            disabled={busy}
          />
        </label>

        {isImageToVideo && (
          <label>
            <span className="text-sm">Asset de referencia (obligatorio para Imagen a Video)</span>
            <Select
              value={referenceAssetIds[0] ?? ""}
              options={availableReferenceAssets.map((a) => ({ value: a.id, label: a.filename }))}
              onChange={(v) => setReferenceAssetIds(v ? [v] : [])}
              ariaLabel="Asset de referencia"
              placeholder="Selecciona asset de referencia"
              disabled={busy}
            />
          </label>
        )}

        <label>
          <span className="text-sm">Assets de grounding (contexto opcional)</span>
          <Select
            value={groundingAssetIds[0] ?? ""}
            options={availableGroundingAssets.map((a) => ({ value: a.id, label: a.filename }))}
            onChange={(v) => setGroundingAssetIds(v ? [v] : [])}
            ariaLabel="Assets de grounding"
            placeholder="Opcional: assets de contexto"
            disabled={busy}
          />
        </label>

        <div className="grid grid-cols-2 gap-2">
          <label>
            <span className="text-sm">Dirección visual</span>
            <input
              type="text"
              className="field"
              value={visualDirection}
              onChange={(e) => setVisualDirection(e.target.value)}
              placeholder="Ej: cinematográfico, anime, fotorealista"
              disabled={busy}
            />
          </label>
          <label>
            <span className="text-sm">Dirección de cámara</span>
            <input
              type="text"
              className="field"
              value={cameraDirection}
              onChange={(e) => setCameraDirection(e.target.value)}
              placeholder="Ej: drone, estático, seguimiento"
              disabled={busy}
            />
          </label>
        </div>

        <label>
          <span className="text-sm">Evitar (restricciones)</span>
          <input
            type="text"
            className="field"
            value={avoidConstraints}
            onChange={(e) => setAvoidConstraints(e.target.value)}
            placeholder="Ej: personas, texto, marcas de agua"
            disabled={busy}
          />
        </label>

        <div className="grid grid-cols-3 gap-2">
          <label>
            <span className="text-sm">Duración (s)</span>
            <input
              type="number"
              className="field"
              value={durationSeconds}
              onChange={(e) => setDurationSeconds(Number(e.target.value) || 1)}
              min="1"
              max="30"
              disabled={busy}
            />
          </label>
          <label>
            <span className="text-sm">Aspecto</span>
            <Select
              value={aspectRatio}
              options={Object.entries(ASPECT_LABELS).map(([value, label]) => ({ value, label }))}
              onChange={(v) => setAspectRatio(v as GenerationAspectRatio)}
              ariaLabel="Aspect ratio"
              disabled={busy}
            />
          </label>
          <label>
            <span className="text-sm">Resolución</span>
            <Select
              value={resolution}
              options={Object.entries(RESOLUTION_LABELS).map(([value, label]) => ({ value, label }))}
              onChange={(v) => setResolution(v as GenerationResolution)}
              ariaLabel="Resolución"
              disabled={busy}
            />
          </label>
        </div>

        <label>
          <span className="text-sm">Seed (opcional)</span>
          <input
            type="number"
            className="field"
            value={seed}
            onChange={(e) => setSeed(e.target.value === "" ? "" : Number(e.target.value))}
            min="0"
            max={2**32 - 1}
            placeholder="Aleatorio"
            disabled={busy}
          />
        </label>

        {error && <div className="text-sm text-bad">Error: {error}</div>}

        <div className="flex gap-2 pt-2">
          <button 
            className="btn-grad flex-1" 
            onClick={prepareGeneration} 
            disabled={busy || (isLocalProvider && !runtimeReady) || (isImageToVideo && referenceAssetIds.length === 0)}
          >
            {busy ? "Preparando…" : "Preparar generación"}
          </button>
          <button className="btn-ghost" onClick={onClose} disabled={busy}>
            Cancelar
          </button>
        </div>
      </div>
    </div>
  );

  const renderEstimateStep = () => {
    if (!estimate) return null;
    const isBillable = estimate.billable;
    const costDisplay = isBillable ? `$${estimate.api_cost_usd.toFixed(2)} ${estimate.currency}` : "$0.00 (Local)";

    return (
      <div className="generation-panel">
        <h3>Estimación lista</h3>
        <p className="text-xs text-ink-mute mb-3">Revisa los detalles antes de aprobar.</p>

        <div className="space-y-2 text-sm">
          <div><strong>Modo:</strong> {MODE_LABELS[mode]}</div>
          <div><strong>Proveedor:</strong> {PROVIDER_LABELS[provider]}</div>
          <div><strong>Modelo:</strong> {derivedModel}</div>
          <div><strong>Coste API:</strong> <span className={isBillable ? "text-amber" : "text-green"}>{costDisplay}</span></div>
          {estimate.estimated_compute_class && <div><strong>Clase de cómputo:</strong> {estimate.estimated_compute_class}</div>}
          {estimate.estimated_vram_class && <div><strong>Clase VRAM:</strong> {estimate.estimated_vram_class}</div>}
          {estimate.estimated_duration_seconds && <div><strong>Tiempo estimado:</strong> ~{estimate.estimated_duration_seconds}s</div>}
          <div><strong>Duración:</strong> {durationSeconds}s</div>
          <div><strong>Aspecto:</strong> {ASPECT_LABELS[aspectRatio]}</div>
          <div><strong>Resolución:</strong> {RESOLUTION_LABELS[resolution]}</div>
          {referenceAssetIds.length > 0 && <div><strong>Referencia:</strong> {referenceAssets.map((a) => a.filename).join(", ")}</div>}
          {groundingAssetIds.length > 0 && <div><strong>Grounding:</strong> {groundingAssets.map((a) => a.filename).join(", ")}</div>}
          {visualDirection && <div><strong>Visual:</strong> {visualDirection}</div>}
          {cameraDirection && <div><strong>Cámara:</strong> {cameraDirection}</div>}
          {avoidConstraints && <div><strong>Evitar:</strong> {avoidConstraints}</div>}
          {seed !== "" && <div><strong>Seed:</strong> {seed}</div>}
        </div>

        <p className="text-xs text-ink-mute mt-3">
          Hash de la solicitud (para aprobación exacta):
          <code className="text-ink">{requestHash?.slice(0, 16)}…</code>
        </p>

        <div className="flex gap-2 pt-3">
          <button className="btn-grad flex-1" onClick={approveAndGenerate} disabled={busy}>
            {busy ? "Aprobando…" : "Aprobar y generar"}
          </button>
          <button className="btn-ghost" onClick={() => {
            setTask(null);
            setEstimate(null);
            setRequestHash(null);
            setStep("config");
          }} disabled={busy}>
            Volver
          </button>
        </div>
      </div>
    );
  };

  const renderPollingStep = () => {
    if (!task) return null;
    const statusLabels: Record<string, string> = {
      QUEUED: "En cola",
      RUNNING: "Generando…",
      COMPLETED: "Completado",
      FAILED: "Falló",
      CANCELLED: "Cancelado",
    };
    const isTerminal = ["COMPLETED", "FAILED", "CANCELLED"].includes(task.status);

    const videoTrack = project.timeline.tracks.find(t => t.type === "video");
    const timelineDurationSec = Math.floor(project.timeline.duration_us / 1_000_000);
    const timelineTimeStr = `${Math.floor(timelineDurationSec / 60).toString().padStart(2, "0")}:${(timelineDurationSec % 60).toString().padStart(2, "0")}`;

    return (
      <div className="generation-panel">
        <h3>Estado de la generación</h3>
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className={`status-badge ${task.status.toLowerCase()}`}>
              {task.status === "RUNNING" && <span className="spinner" />}
              {statusLabels[task.status] || task.status}
            </span>
          </div>

          {task.error && (
            <div className="text-sm text-bad">Error: {task.error}</div>
          )}

          {task.status === "COMPLETED" && task.output_asset && (
            <div className="space-y-3">
              <div className="text-sm text-green">
                ✓ Generación completada. Nuevo asset: <strong>{task.output_asset.filename}</strong>
                <div className="text-xs text-ink-mute mt-1">
                  GENERATED · ILLUSTRATIVE · No evidencia factual
                </div>
              </div>

              {videoTrack ? (
                <div className="space-y-2 text-xs text-ink-mute">
                  <div>
                    Destino: final de la pista <strong>Video</strong> · {timelineTimeStr}
                  </div>
                  <div>
                    Asset generado: <strong>{task.output_asset.filename}</strong> · {Math.floor(task.output_asset.duration_us / 1_000_000)}s · {task.output_asset.width}×{task.output_asset.height}
                  </div>
                </div>
              ) : (
                <div className="text-xs text-bad">
                  No hay pista de video en el proyecto. Crea una pista de video antes de insertar.
                </div>
              )}

              <div className="flex gap-2 pt-3">
                <button
                  className="btn-grad flex-1"
                  onClick={insertGeneratedAsset}
                  disabled={busy || !videoTrack || task.status !== "COMPLETED"}
                >
                  {busy ? "Insertando…" : "Insertar al final de la pista Video"}
                </button>
                <button className="btn-ghost" onClick={handleClose} disabled={busy}>
                  Cerrar
                </button>
              </div>
            </div>
          )}

          {task.status === "RUNNING" && (
            <div className="text-xs text-ink-mute">
              Generación ejecutándose en background. Esto puede tardar varios minutos.
              {task.cancel_requested && " Cancelación solicitada…"}
            </div>
          )}

          {!isTerminal && (
            <button className="btn-ghost w-full" onClick={cancelTask} disabled={busy}>
              {busy ? "Cancelando…" : "Cancelar tarea"}
            </button>
          )}

          {task.status === "FAILED" && !isLocalProvider && (
            <div className="text-xs text-ink-mute">
              La generación cloud falló. No hay fallback automático a local.
            </div>
          )}
        </div>
      </div>
    );
  };

  const handleClose = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      pollingTaskIdRef.current = null;
    }
    if (completionTimeoutRef.current) {
      clearTimeout(completionTimeoutRef.current);
      completionTimeoutRef.current = null;
    }
    onClose();
  }, [onClose]);

  return (
    <div data-testid="generation-panel-overlay" className="generation-panel-overlay" onClick={handleClose}>
      <div data-testid="generation-panel-content" className="generation-panel-content" onClick={(e) => e.stopPropagation()}>
        <button className="panel-close" onClick={handleClose} aria-label="Cerrar">✕</button>
        {step === "capability" && renderCapabilityStep()}
        {step === "config" && renderConfigStep()}
        {step === "estimate" && renderEstimateStep()}
        {step === "polling" && renderPollingStep()}
        {error && <div className="error-toast">{error}</div>}
      </div>
    </div>
  );
}
