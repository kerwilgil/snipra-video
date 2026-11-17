import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { Select } from "./Select";
import type { EditorAsset, EnhancementTask, EnhancementProfileValue, EnhancementEstimate } from "../lib/types";

const PROFILE_LABELS: Record<EnhancementProfileValue, string> = {
  DEFAULT: "DEFAULT — Mejora local y determinista",
  UPSCALE: "UPSCALE — Aumentar resolución",
  CLEANUP: "CLEANUP — Reducir ruido y limpiar",
  CUSTOM: "CUSTOM — Parámetros manuales",
};

const PROFILE_DESCRIPTIONS: Record<EnhancementProfileValue, string> = {
  DEFAULT: "Mejora local y determinista con FFmpeg. No utiliza IA generativa ni servicios externos.",
  UPSCALE: "Aumenta la resolución con reescalado Lanczos + recuperación ligera de detalle.",
  CLEANUP: "Reducción conservadora de ruido (hqdn3d) + sharpening prudente. No restaura detalle ni hace super-resolución.",
  CUSTOM: "Controles tipados: sharpen, contraste, brillo, saturación, denoise. Sin argumentos FFmpeg libres.",
};

const PROFILE_HINTS: Record<EnhancementProfileValue, string[]> = {
  DEFAULT: ["Coste API: $0", "Procesamiento: Local", "Engine: FFmpeg"],
  UPSCALE: ["Coste API: $0", "Procesamiento: Local", "Engine: FFmpeg", "Factor 2× o dimensiones compatibles"],
  CLEANUP: ["Coste API: $0", "Procesamiento: Local", "Engine: FFmpeg", "Denoise + sharpening conservador"],
  CUSTOM: ["Coste API: $0", "Procesamiento: Local", "Engine: FFmpeg", "Solo controles tipados"],
};

interface EnhancementPanelProps {
  projectId: string;
  asset: EditorAsset;
  onClose: () => void;
  onTaskCompleted?: (task: EnhancementTask) => void;
}

export function EnhancementPanel({ projectId, asset, onClose, onTaskCompleted }: EnhancementPanelProps) {
  const [step, setStep] = useState<"config" | "estimate" | "polling">("config");
  const [profile, setProfile] = useState<EnhancementProfileValue>("DEFAULT");
  const [params, setParams] = useState({
    target_width: "",
    target_height: "",
    upscale_factor: "",
    sharpen_strength: "",
    contrast: "",
    brightness: "",
    saturation: "",
    denoise_strength: "",
    preserve_audio: true,
  });
  const [estimate, setEstimate] = useState<EnhancementEstimate | null>(null);
  const [task, setTask] = useState<EnhancementTask | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [requestHash, setRequestHash] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const pollingTaskIdRef = useRef<string | null>(null);
  const completionTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const profileCanUpscale = profile === "UPSCALE";
  const profileIsCustom = profile === "CUSTOM";
  const showDimensions = profileCanUpscale || profileIsCustom;
  const showUpscaleFactor = profileCanUpscale;

  const handleParamChange = useCallback((key: string, value: string | boolean) => {
    setParams((prev) => ({ ...prev, [key]: value }));
  }, []);

  const prepareEnhancement = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const body = {
        profile,
        target_width: params.target_width ? Number(params.target_width) : undefined,
        target_height: params.target_height ? Number(params.target_height) : undefined,
        upscale_factor: params.upscale_factor ? Number(params.upscale_factor) : undefined,
        sharpen_strength: params.sharpen_strength ? Number(params.sharpen_strength) : undefined,
        contrast: params.contrast ? Number(params.contrast) : undefined,
        brightness: params.brightness ? Number(params.brightness) : undefined,
        saturation: params.saturation ? Number(params.saturation) : undefined,
        denoise_strength: params.denoise_strength ? Number(params.denoise_strength) : undefined,
        preserve_audio: params.preserve_audio,
      };
      const result = await api.prepareAssetEnhancement(projectId, asset.id, body);
      setTask(result);
      setEstimate(result.cost_estimate);
      setRequestHash(result.request_hash);
      setStep("estimate");
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo preparar la mejora");
    } finally {
      setBusy(false);
    }
  }, [projectId, asset.id, profile, params]);

  const approveAndEnhance = useCallback(async () => {
    if (!task || !requestHash) return;
    setBusy(true);
    setError(null);
    try {
      const result = await api.approveEnhancementTask(projectId, task.id, requestHash);
      setTask(result);
      setStep("polling");
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo aprobar la mejora");
    } finally {
      setBusy(false);
    }
  }, [projectId, task, requestHash]);

  const cancelTask = useCallback(async () => {
    if (!task) return;
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      pollingTaskIdRef.current = null;
    }
    setBusy(true);
    try {
      await api.cancelEnhancementTask(projectId, task.id);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo cancelar");
    } finally {
      setBusy(false);
    }
  }, [projectId, task, onClose]);

  useEffect(() => {
    if (step !== "polling" || !task) return;
    const controller = new AbortController();
    abortControllerRef.current = controller;
    pollingTaskIdRef.current = task.id;
    api.waitForEnhancementTask(projectId, task.id, {
      intervalMs: 1200,
      timeoutMs: 60 * 60 * 1000,
      onProgress: (t) => {
        if (!controller.signal.aborted && pollingTaskIdRef.current === t.id) {
          setTask(t);
        }
      },
      signal: controller.signal,
    })
      .then((completed) => {
        if (!controller.signal.aborted && pollingTaskIdRef.current === completed.id) {
          setTask(completed);
          if (completed.status === "COMPLETED") {
            onTaskCompleted?.(completed);
            completionTimeoutRef.current = setTimeout(() => {
              completionTimeoutRef.current = null;
              onClose();
            }, 1500);
          }
        }
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => {
      controller.abort();
      if (completionTimeoutRef.current) {
        clearTimeout(completionTimeoutRef.current);
        completionTimeoutRef.current = null;
      }
      abortControllerRef.current = null;
      pollingTaskIdRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, projectId, onTaskCompleted, onClose]);

  const renderConfigStep = () => (
    <div className="enhancement-panel">
      <h3>Mejorar asset</h3>
      <p className="text-xs text-ink-mute mb-3">
        El asset <strong>{asset.filename}</strong> se procesará localmente con FFmpeg.
        No hay coste de API ni llamadas externas.
      </p>

      <div className="space-y-3">
        <label>
          <span className="text-sm">Perfil</span>
          <Select
            value={profile}
            options={Object.entries(PROFILE_LABELS).map(([value, label]) => ({ value, label }))}
            onChange={(v) => setProfile(v as EnhancementProfileValue)}
            ariaLabel="Perfil de mejora"
          />
        </label>

        <p className="text-xs text-ink-mute">{PROFILE_DESCRIPTIONS[profile]}</p>

        <div className="grid grid-cols-2 gap-2 text-xs text-cyan">
          {PROFILE_HINTS[profile].map((hint, i) => (
            <span key={i}>• {hint}</span>
          ))}
        </div>

        {showDimensions && (
          <div className="grid grid-cols-2 gap-2">
            <label>
              <span className="text-sm">Ancho destino</span>
              <input
                type="number"
                className="field"
                value={params.target_width}
                onChange={(e) => handleParamChange("target_width", e.target.value)}
                min="16"
                max="15360"
                placeholder="Opcional"
              />
            </label>
            <label>
              <span className="text-sm">Alto destino</span>
              <input
                type="number"
                className="field"
                value={params.target_height}
                onChange={(e) => handleParamChange("target_height", e.target.value)}
                min="16"
                max="15360"
                placeholder="Opcional"
              />
            </label>
          </div>
        )}

        {showUpscaleFactor && (
          <label>
            <span className="text-sm">Factor de escala (ej. 2 = 2×)</span>
            <input
              type="number"
              className="field"
              value={params.upscale_factor}
              onChange={(e) => handleParamChange("upscale_factor", e.target.value)}
              step="0.1"
              min="1.0"
              max="8.0"
              placeholder="2"
            />
          </label>
        )}

        {profileIsCustom && (
          <div className="grid grid-cols-2 gap-2 border-t pt-3">
            <label>
              <span className="text-sm">Sharpen (0–3)</span>
              <input
                type="number"
                className="field"
                value={params.sharpen_strength}
                onChange={(e) => handleParamChange("sharpen_strength", e.target.value)}
                step="0.1"
                min="0"
                max="3"
                placeholder="Opcional"
              />
            </label>
            <label>
              <span className="text-sm">Contraste (0–3)</span>
              <input
                type="number"
                className="field"
                value={params.contrast}
                onChange={(e) => handleParamChange("contrast", e.target.value)}
                step="0.1"
                min="0"
                max="3"
                placeholder="Opcional"
              />
            </label>
            <label>
              <span className="text-sm">Brillo (-1 a 1)</span>
              <input
                type="number"
                className="field"
                value={params.brightness}
                onChange={(e) => handleParamChange("brightness", e.target.value)}
                step="0.1"
                min="-1"
                max="1"
                placeholder="Opcional"
              />
            </label>
            <label>
              <span className="text-sm">Saturación (0–3)</span>
              <input
                type="number"
                className="field"
                value={params.saturation}
                onChange={(e) => handleParamChange("saturation", e.target.value)}
                step="0.1"
                min="0"
                max="3"
                placeholder="Opcional"
              />
            </label>
            <label>
              <span className="text-sm">Denoise (0–3)</span>
              <input
                type="number"
                className="field"
                value={params.denoise_strength}
                onChange={(e) => handleParamChange("denoise_strength", e.target.value)}
                step="0.1"
                min="0"
                max="3"
                placeholder="Opcional"
              />
            </label>
          </div>
        )}

        <label className="check-field">
          <input
            type="checkbox"
            checked={params.preserve_audio}
            onChange={(e) => handleParamChange("preserve_audio", e.target.checked)}
          />
          Preservar audio
        </label>

        <div className="flex gap-2 pt-2">
          <button className="btn-grad flex-1" onClick={prepareEnhancement} disabled={busy}>
            {busy ? "Preparando…" : "Preparar mejora"}
          </button>
          <button className="btn-ghost" onClick={onClose} disabled={busy}>
            Cancelar
          </button>
        </div>
      </div>
    </div>
  );

  const renderEstimateStep = () => (
    <div className="enhancement-panel">
      <h3>Estimación lista</h3>
      <p className="text-xs text-ink-mute mb-3">Revisa los detalles antes de aprobar.</p>

      <div className="space-y-2 text-sm">
        <div><strong>Perfil:</strong> {PROFILE_LABELS[profile]}</div>
        <div><strong>Coste API:</strong> <span className="text-green">${estimate?.monetary_cost_usd ?? 0}</span></div>
        <div><strong>Procesamiento:</strong> Local (FFmpeg)</div>
        {estimate?.estimated_seconds && (
          <div><strong>Tiempo estimado:</strong> ~{estimate.estimated_seconds}s</div>
        )}
        <div><strong>Audio:</strong> {params.preserve_audio ? "Preservar" : "Remover"}</div>
        {params.target_width && <div><strong>Ancho:</strong> {params.target_width}</div>}
        {params.target_height && <div><strong>Alto:</strong> {params.target_height}</div>}
        {params.upscale_factor && <div><strong>Factor:</strong> {params.upscale_factor}×</div>}
        {params.sharpen_strength && <div><strong>Sharpen:</strong> {params.sharpen_strength}</div>}
        {params.contrast && <div><strong>Contraste:</strong> {params.contrast}</div>}
        {params.brightness && <div><strong>Brillo:</strong> {params.brightness}</div>}
        {params.saturation && <div><strong>Saturación:</strong> {params.saturation}</div>}
        {params.denoise_strength && <div><strong>Denoise:</strong> {params.denoise_strength}</div>}
      </div>

      <p className="text-xs text-ink-mute mt-3">
        Hash de la solicitud (para aprobación):
        <code className="text-ink">{requestHash?.slice(0, 16)}…</code>
      </p>

      <div className="flex gap-2 pt-3">
        <button className="btn-grad flex-1" onClick={approveAndEnhance} disabled={busy}>
          {busy ? "Encolando…" : "Aprobar y mejorar"}
        </button>
        <button className="btn-ghost" onClick={() => setStep("config")} disabled={busy}>
          Volver
        </button>
      </div>
    </div>
  );

  const renderPollingStep = () => {
    if (!task) return null;
    const statusLabels: Record<string, string> = {
      QUEUED: "En cola",
      RUNNING: "Procesando localmente…",
      COMPLETED: "Completado",
      FAILED: "Falló",
      CANCELLED: "Cancelado",
    };
    const isTerminal = ["COMPLETED", "FAILED", "CANCELLED"].includes(task.status);

    return (
      <div className="enhancement-panel">
        <h3>Estado de la tarea</h3>
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
            <div className="text-sm text-green">
              ✓ Mejora completada. Nuevo asset: <strong>{task.output_asset.filename}</strong>
            </div>
          )}

          {task.status === "RUNNING" && (
            <div className="text-xs text-ink-mute">
              FFmpeg ejecutándose en background. Esto puede tardar unos minutos.
            </div>
          )}

          {!isTerminal && (
            <button className="btn-ghost w-full" onClick={cancelTask} disabled={busy}>
              {busy ? "Cancelando…" : "Cancelar tarea"}
            </button>
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
    <div className="enhancement-panel-overlay" onClick={handleClose}>
      <div className="enhancement-panel-content" onClick={(e) => e.stopPropagation()}>
        <button className="panel-close" onClick={handleClose} aria-label="Cerrar">✕</button>
        {step === "config" && renderConfigStep()}
        {step === "estimate" && renderEstimateStep()}
        {step === "polling" && renderPollingStep()}
        {error && <div className="error-toast">{error}</div>}
      </div>
    </div>
  );
}