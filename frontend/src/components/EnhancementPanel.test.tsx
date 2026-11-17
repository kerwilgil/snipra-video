import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { EnhancementPanel } from "./EnhancementPanel";
import type { EditorAsset, EnhancementTask, EnhancementEstimate } from "../lib/types";

// Mock the api module
vi.mock("../lib/api", () => ({
  api: {
    prepareAssetEnhancement: vi.fn(),
    approveEnhancementTask: vi.fn(),
    cancelEnhancementTask: vi.fn(),
    waitForEnhancementTask: vi.fn(),
  },
}));

const onTaskCompleted = vi.fn();

const mockAsset: EditorAsset = {
  id: "asset123",
  filename: "test.mp4",
  kind: "video",
  duration_us: 5_000_000,
  width: 1280,
  height: 720,
  has_audio: true,
  fingerprint: "abc123",
};

const mockTask: EnhancementTask = {
  id: "task123",
  project_id: "proj123",
  kind: "ENHANCEMENT",
  status: "AWAITING_APPROVAL",
  provider: "ffmpeg",
  model: "",
  request: { profile: "DEFAULT", source_asset_id: "asset123" },
  request_hash: "abc123hash",
  cost_estimate: {
    cost_class: "LOCAL",
    monetary_cost_known: true,
    monetary_cost_usd: 0,
    estimated_seconds: 10,
    notes: "local FFmpeg",
  },
  approval_state: "PENDING",
  approved_request_hash: null,
  progress: 0,
  output_asset_id: null,
  error: null,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
};

const mockEstimate: EnhancementEstimate = {
  cost_class: "LOCAL",
  monetary_cost_known: true,
  monetary_cost_usd: 0,
  estimated_seconds: 10,
  notes: "local FFmpeg (CPU)",
};

describe("EnhancementPanel", () => {
  const onClose = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders config step by default", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />);
    expect(screen.getByText("Mejorar asset")).toBeInTheDocument();
    expect(screen.getByText("DEFAULT — Mejora local y determinista")).toBeInTheDocument();
    expect(screen.getByText("Preparar mejora")).toBeInTheDocument();
  });

  it("shows DEFAULT profile as default", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />);
    // Select component renders as button with aria-haspopup="listbox"
    const selectButton = screen.getByLabelText("Perfil de mejora");
    expect(selectButton).toBeInTheDocument();
    expect(selectButton).toHaveAttribute("aria-expanded", "false");
  });

  it("shows profile description for DEFAULT", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />);
    expect(screen.getByText("Mejora local y determinista con FFmpeg. No utiliza IA generativa ni servicios externos.")).toBeInTheDocument();
  });

  it("shows hints for DEFAULT profile", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />);
    // The hints are rendered as separate spans with bullet points
    expect(screen.getByText((content) => content.includes("Coste API: $0"))).toBeInTheDocument();
    expect(screen.getByText((content) => content.includes("Procesamiento: Local"))).toBeInTheDocument();
    expect(screen.getByText((content) => content.includes("Engine: FFmpeg"))).toBeInTheDocument();
  });

  it("has Preservar audio checkbox checked by default", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />);
    const checkbox = screen.getByLabelText("Preservar audio");
    expect(checkbox).toBeChecked();
  });

  it("calls prepareAssetEnhancement on prepare click", async () => {
    const { api } = await import("../lib/api");
    api.prepareAssetEnhancement.mockResolvedValue({ ...mockTask, cost_estimate: mockEstimate });

    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />);
    fireEvent.click(screen.getByText("Preparar mejora"));

    await vi.waitFor(() => {
      expect(api.prepareAssetEnhancement).toHaveBeenCalledWith(
        "proj123",
        "asset123",
        expect.objectContaining({ profile: "DEFAULT" })
      );
    });
  });

  it("does NOT call approve on prepare", async () => {
    const { api } = await import("../lib/api");
    api.prepareAssetEnhancement.mockResolvedValue({ ...mockTask, cost_estimate: mockEstimate });
    api.approveEnhancementTask.mockResolvedValue({ ...mockTask, status: "QUEUED" });

    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />);
    fireEvent.click(screen.getByText("Preparar mejora"));

    await vi.waitFor(() => {
      expect(screen.getByText("Aprobar y mejorar")).toBeInTheDocument();
    });

    expect(api.approveEnhancementTask).not.toHaveBeenCalled();
  });

  it("shows estimate with FFmpeg/Local/$0", async () => {
    const { api } = await import("../lib/api");
    api.prepareAssetEnhancement.mockResolvedValue({ ...mockTask, cost_estimate: mockEstimate });

    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />);
    fireEvent.click(screen.getByText("Preparar mejora"));

    await vi.waitFor(() => {
      // In estimate step: "Local (FFmpeg)" and "$0"
      expect(screen.getByText((content) => content.includes("FFmpeg"))).toBeInTheDocument();
      expect(screen.getByText((content) => content.includes("Local"))).toBeInTheDocument();
      expect(screen.getByText((content) => content.includes("$0"))).toBeInTheDocument();
    });
  });

  it("approve sends server request_hash", async () => {
    const { api } = await import("../lib/api");
    api.prepareAssetEnhancement.mockResolvedValue({ ...mockTask, cost_estimate: mockEstimate });
    api.approveEnhancementTask.mockResolvedValue({ ...mockTask, status: "QUEUED" });

    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />);
    fireEvent.click(screen.getByText("Preparar mejora"));
    await vi.waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y mejorar"));
    });

    await vi.waitFor(() => {
      expect(api.approveEnhancementTask).toHaveBeenCalledWith(
        "proj123",
        "task123",
        "abc123hash"
      );
    });
  });

  it("close button calls onClose", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />);
    fireEvent.click(screen.getByLabelText("Cerrar"));
    expect(onClose).toHaveBeenCalled();
  });

  it("overlay click calls onClose", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />);
    const overlay = screen.getByText("Mejorar asset").closest(".enhancement-panel-overlay");
    if (overlay) {
      fireEvent.click(overlay);
      expect(onClose).toHaveBeenCalled();
    }
  });

  it("shows UPSCALE profile options when selected", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />);
    // Click to open select
    fireEvent.click(screen.getByLabelText("Perfil de mejora"));
    // The select dropdown should show UPSCALE option
    expect(screen.getByText("UPSCALE — Aumentar resolución")).toBeInTheDocument();
  });

  it("shows CLEANUP profile options when selected", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />);
    fireEvent.click(screen.getByLabelText("Perfil de mejora"));
    expect(screen.getByText("CLEANUP — Reducir ruido y limpiar")).toBeInTheDocument();
  });

  it("shows CUSTOM profile options when selected", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />);
    fireEvent.click(screen.getByLabelText("Perfil de mejora"));
    expect(screen.getByText("CUSTOM — Parámetros manuales")).toBeInTheDocument();
  });

  // --- POLLING TESTS ---

  it("polling lifecycle: QUEUED -> RUNNING -> COMPLETED", async () => {
    const { api } = await import("../lib/api");
    api.prepareAssetEnhancement.mockResolvedValue({ ...mockTask, cost_estimate: mockEstimate });
    api.approveEnhancementTask.mockResolvedValue({ ...mockTask, status: "QUEUED" });

    // Mock waitForEnhancementTask to simulate polling with onProgress callbacks
    let onProgressCb: ((task: EnhancementTask) => void) | null = null
    let resolveFinal: (task: EnhancementTask) => void
    const finalPromise = new Promise<EnhancementTask>((resolve) => {
      resolveFinal = resolve
    })
    
    api.waitForEnhancementTask.mockImplementation(async (projectId, taskId, options) => {
      onProgressCb = options?.onProgress ?? null
      // Simulate initial QUEUED
      if (onProgressCb) onProgressCb({ ...mockTask, status: "QUEUED", progress: 0 })
      return finalPromise
    })

    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} onTaskCompleted={onTaskCompleted} />);

    // Prepare
    fireEvent.click(screen.getByText("Preparar mejora"))
    await waitFor(() => {
      expect(screen.getByText("Aprobar y mejorar")).toBeInTheDocument()
    })

    // Approve
    fireEvent.click(screen.getByText("Aprobar y mejorar"))
    await waitFor(() => {
      expect(screen.getByText("En cola")).toBeInTheDocument()
    })

    // Simulate RUNNING via onProgress
    act(() => {
      onProgressCb!({ ...mockTask, status: "RUNNING", progress: 0.5 })
    })
    await waitFor(() => {
      expect(screen.getByText((content) => content.includes("Procesando localmente"))).toBeInTheDocument()
    })

    // Simulate COMPLETED by resolving the final promise
    act(() => {
      resolveFinal!({
        ...mockTask,
        status: "COMPLETED",
        progress: 1.0,
        output_asset: { ...mockAsset, id: "enhanced123", filename: "enhanced.mp4" },
      })
    })
    await waitFor(() => {
      expect(screen.getByText("Completado")).toBeInTheDocument()
      expect(screen.getByText((content) => content.includes("Mejora completada"))).toBeInTheDocument()
    })

    await waitFor(() => {
      expect(onTaskCompleted).toHaveBeenCalledTimes(1)
    })
  })

  it("unmount aborts polling", async () => {
    const { api } = await import("../lib/api")
    api.prepareAssetEnhancement.mockResolvedValue({ ...mockTask, cost_estimate: mockEstimate })
    api.approveEnhancementTask.mockResolvedValue({ ...mockTask, status: "QUEUED" })

    // Track the AbortSignal passed to waitForEnhancementTask
    let capturedSignal: AbortSignal | null = null
    let pollingStarted = false
    api.waitForEnhancementTask.mockImplementation(async (_projectId, _taskId, options) => {
      capturedSignal = options?.signal ?? null
      pollingStarted = true
      return new Promise(() => {}) // Never resolves - stays in polling
    })

    const { unmount } = render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />)

    fireEvent.click(screen.getByText("Preparar mejora"))
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y mejorar"))
    })

    // Wait for polling to start
    await waitFor(() => {
      expect(pollingStarted).toBe(true)
    })

    // Verify signal was captured
    expect(capturedSignal).not.toBeNull()
    expect(capturedSignal!.aborted).toBe(false)

    // Unmount while polling
    unmount()

    // After unmount, the AbortController should have been triggered
    await waitFor(() => {
      expect(capturedSignal!.aborted).toBe(true)
    })

    // onClose should NOT have been called (completion timeout shouldn't fire after unmount)
    expect(onClose).not.toHaveBeenCalled()
  })

  it("does not create duplicate polling loops", async () => {
    const { api } = await import("../lib/api")
    api.prepareAssetEnhancement.mockResolvedValue({ ...mockTask, cost_estimate: mockEstimate })
    api.approveEnhancementTask.mockResolvedValue({ ...mockTask, status: "QUEUED" })

    let resolvePoll: (value: EnhancementTask) => void
    const pollPromise = new Promise<EnhancementTask>((resolve) => {
      resolvePoll = resolve
    })
    api.waitForEnhancementTask.mockReturnValue(pollPromise)

    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} onTaskCompleted={onTaskCompleted} />)

    fireEvent.click(screen.getByText("Preparar mejora"))
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y mejorar"))
    })

    // Simulate multiple progress updates
    act(() => {
      resolvePoll!({ ...mockTask, status: "RUNNING", progress: 0.3 })
    })
    await waitFor(() => {
      expect(screen.getByText((content) => content.includes("Procesando localmente"))).toBeInTheDocument()
    })

    act(() => {
      resolvePoll!({ ...mockTask, status: "RUNNING", progress: 0.6 })
    })
    await waitFor(() => {
      expect(screen.getByText((content) => content.includes("Procesando localmente"))).toBeInTheDocument()
    })

    // waitForEnhancementTask should only have been called once
    expect(api.waitForEnhancementTask).toHaveBeenCalledTimes(1)
  })

  it("shows error on FAILED status", async () => {
    const { api } = await import("../lib/api")
    api.prepareAssetEnhancement.mockResolvedValue({ ...mockTask, cost_estimate: mockEstimate })
    api.approveEnhancementTask.mockResolvedValue({ ...mockTask, status: "QUEUED" })

    let resolvePoll: (value: EnhancementTask) => void
    const pollPromise = new Promise<EnhancementTask>((resolve) => {
      resolvePoll = resolve
    })
    api.waitForEnhancementTask.mockReturnValue(pollPromise)

    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />)

    fireEvent.click(screen.getByText("Preparar mejora"))
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y mejorar"))
    })

    act(() => {
      resolvePoll!({
        ...mockTask,
        status: "FAILED",
        error: "FFmpeg failed",
      })
    })

    await waitFor(() => {
      expect(screen.getByText("Falló")).toBeInTheDocument()
      expect(screen.getByText("Error: FFmpeg failed")).toBeInTheDocument()
    })
  })

  it("shows error on CANCELLED status", async () => {
    const { api } = await import("../lib/api")
    api.prepareAssetEnhancement.mockResolvedValue({ ...mockTask, cost_estimate: mockEstimate })
    api.approveEnhancementTask.mockResolvedValue({ ...mockTask, status: "QUEUED" })

    let resolvePoll: (value: EnhancementTask) => void
    const pollPromise = new Promise<EnhancementTask>((resolve) => {
      resolvePoll = resolve
    })
    api.waitForEnhancementTask.mockReturnValue(pollPromise)

    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />)

    fireEvent.click(screen.getByText("Preparar mejora"))
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y mejorar"))
    })

    act(() => {
      resolvePoll!({
        ...mockTask,
        status: "CANCELLED",
      })
    })

    await waitFor(() => {
      expect(screen.getByText("Cancelado")).toBeInTheDocument()
    })
  })

  it("shows enhanced badge on asset after completion", async () => {
    const { api } = await import("../lib/api")
    api.prepareAssetEnhancement.mockResolvedValue({ ...mockTask, cost_estimate: mockEstimate })
    api.approveEnhancementTask.mockResolvedValue({ ...mockTask, status: "QUEUED" })

    let resolvePoll: (value: EnhancementTask) => void
    const pollPromise = new Promise<EnhancementTask>((resolve) => {
      resolvePoll = resolve
    })
    api.waitForEnhancementTask.mockReturnValue(pollPromise)

    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} onTaskCompleted={onTaskCompleted} />)

    fireEvent.click(screen.getByText("Preparar mejora"))
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y mejorar"))
    })

    act(() => {
      resolvePoll!({
        ...mockTask,
        status: "COMPLETED",
        progress: 1.0,
        output_asset: { ...mockAsset, id: "enhanced123", filename: "enhanced.mp4" },
      })
    })

    await waitFor(() => {
      expect(screen.getByText((content) => content.includes("enhanced.mp4"))).toBeInTheDocument()
    })
  })

  it("cancel button calls cancelEnhancementTask", async () => {
    const { api } = await import("../lib/api")
    api.prepareAssetEnhancement.mockResolvedValue({ ...mockTask, cost_estimate: mockEstimate })
    api.approveEnhancementTask.mockResolvedValue({ ...mockTask, status: "QUEUED" })
    api.cancelEnhancementTask.mockResolvedValue({ ...mockTask, status: "CANCELLED" })
    // Mock waitForEnhancementTask to return a never-resolving promise so the component stays in polling state
    api.waitForEnhancementTask.mockImplementation(() => new Promise(() => {}))

    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />)

    fireEvent.click(screen.getByText("Preparar mejora"))
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y mejorar"))
    })

    await waitFor(() => {
      fireEvent.click(screen.getByText("Cancelar tarea"))
    })

    await waitFor(() => {
      expect(api.cancelEnhancementTask).toHaveBeenCalledWith("proj123", "task123")
    })
  })

  it("close button calls onClose", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />)
    fireEvent.click(screen.getByLabelText("Cerrar"))
    expect(onClose).toHaveBeenCalled()
  })

  it("overlay click calls onClose", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />)
    const overlay = screen.getByText("Mejorar asset").closest(".enhancement-panel-overlay")
    if (overlay) {
      fireEvent.click(overlay)
      expect(onClose).toHaveBeenCalled()
    }
  })

  it("shows UPSCALE profile options when selected", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />)
    fireEvent.click(screen.getByLabelText("Perfil de mejora"))
    expect(screen.getByText("UPSCALE — Aumentar resolución")).toBeInTheDocument()
  })

  it("shows CLEANUP profile options when selected", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />)
    fireEvent.click(screen.getByLabelText("Perfil de mejora"))
    expect(screen.getByText("CLEANUP — Reducir ruido y limpiar")).toBeInTheDocument()
  })

  it("shows CUSTOM profile options when selected", () => {
    render(<EnhancementPanel projectId="proj123" asset={mockAsset} onClose={onClose} />)
    fireEvent.click(screen.getByLabelText("Perfil de mejora"))
    expect(screen.getByText("CUSTOM — Parámetros manuales")).toBeInTheDocument()
  })
});