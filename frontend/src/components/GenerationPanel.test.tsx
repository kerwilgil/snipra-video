import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { GenerationPanel } from "./GenerationPanel";
import type { EditorProject, EditorAsset } from "../lib/types";
import { api } from "../lib/api";

const mockProject: EditorProject = {
  id: "proj-123",
  name: "Test Project",
  source_job_id: "job-123",
  current_version: 1,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  timeline: { duration_us: 30_000_000, tracks: [{ id: "video-1", type: "video", locked: false, muted: false, solo: false, clips: [] }] },
  assets: [],
};

const mockAssets: EditorAsset[] = [
  {
    id: "asset-1",
    filename: "reference.mp4",
    kind: "video",
    duration_us: 5_000_000,
    width: 1920,
    height: 1080,
    has_audio: true,
    fingerprint: "fp1",
  },
  {
    id: "asset-2",
    filename: "grounding.mp4",
    kind: "video",
    duration_us: 10_000_000,
    width: 1280,
    height: 720,
    has_audio: false,
    fingerprint: "fp2",
  },
];

vi.mock("../lib/api", () => ({
  api: {
    getGenerationCapability: vi.fn(),
    prepareGeneration: vi.fn(),
    getGenerationTask: vi.fn(),
    approveGenerationTask: vi.fn(),
    rejectGenerationTask: vi.fn(),
    cancelGenerationTask: vi.fn(),
    editorOperation: vi.fn(),
  },
}));

const mockCapability = {
  local_ltx: {
    provider: "local_ltx",
    model: "ltx-video-2b-distilled-fp8",
    hardware_supported: true,
    hardware_message: "Hardware compatible",
    hardware_reason: "",
    runtime_ready: true,
    runtime_message: "Hardware compatible, pero la ejecución física LTX aún no está habilitada",
    physical_validation: "DEFERRED",
    paid: false,
    billable: false,
    requires_api_key: false,
    enabled_by_default: true,
  },
  cloud_providers: [],
};

const mockTask = {
  id: "task-123",
  project_id: "proj-123",
  kind: "GENERATION",
  status: "ESTIMATING",
  provider: "local_ltx",
  model: "ltx-video-2b-distilled-fp8",
  request: {
    mode: "text_to_video",
    provider: "local_ltx",
    model: "ltx-video-2b-distilled-fp8",
    prompt_intent: "A beautiful sunset",
    visual_direction: undefined,
    camera_direction: undefined,
    avoid_constraints: undefined,
    duration_seconds: 5,
    aspect_ratio: "16:9",
    resolution: "SD",
    reference_asset_ids: [],
    grounding_asset_ids: [],
    seed: undefined,
    generation_parameters: {},
  },
  request_hash: "abc123hash",
  cost_estimate: {
    provider: "local_ltx",
    model: "ltx-video-2b-distilled-fp8",
    request_hash: "abc123hash",
    api_cost_usd: 0.0,
    currency: "USD",
    estimated_compute_class: "LOCAL_GPU",
    estimated_vram_class: "VRAM_8GB",
    estimated_duration_seconds: 30,
    billable: false,
    requires_monetary_approval: false,
    estimate_version: "gen_est_v1",
    is_exact: true,
  },
  approval_state: "PENDING",
  approved_request_hash: null,
  progress: 0,
  output_asset_id: null,
  error: null,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
};

const mockProjectWithTrack: EditorProject = {
  id: "proj-123",
  name: "Test Project",
  source_job_id: "job-123",
  current_version: 1,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  timeline: { duration_us: 30_000_000, tracks: [{ id: "video-1", type: "video", locked: false, muted: false, solo: false, clips: [] }] },
  assets: [],
};

describe("GenerationPanel", () => {
  const onClose = vi.fn();
  const onTaskCompleted = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (api.getGenerationCapability as ReturnType<typeof vi.fn>).mockResolvedValue(mockCapability);
  });

  it("renders capability step on mount", async () => {
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      expect(screen.getByText("Generación de Video")).toBeInTheDocument();
    });
    
    expect(screen.getByText("LTX-Video Local")).toBeInTheDocument();
    expect(screen.getByText("Compatible")).toBeInTheDocument();
    expect(screen.getByText("Continuar a configuración")).toBeInTheDocument();
  });

  it("shows local unavailable message when hardware not supported", async () => {
    (api.getGenerationCapability as vi.Mock).mockResolvedValue({
      ...mockCapability,
      local_ltx: {
        ...mockCapability.local_ltx,
        hardware_supported: false,
        runtime_ready: false,
        hardware_message: "Generación local no disponible en este hardware: No NVIDIA GPU",
        hardware_reason: "NO_NVIDIA_GPU",
        physical_validation: "DEFERRED",
      },
    });

    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      expect(screen.getByText("No compatible")).toBeInTheDocument();
      expect(screen.getByText(/requiere GPU NVIDIA/)).toBeInTheDocument();
    });
  });

  it("navigates to config step when clicking continue", async () => {
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      expect(screen.getByText("Continuar a configuración")).toBeInTheDocument();
    });
    
    fireEvent.click(screen.getByText("Continuar a configuración"));
    
    await waitFor(() => {
      expect(screen.getByText("Configurar Generación")).toBeInTheDocument();
    });
  });

  it("renders T2V config form with all fields", async () => {
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Configurar Generación")).toBeInTheDocument();
    });
    
    expect(screen.getByLabelText("Modo")).toBeInTheDocument();
    expect(screen.getByLabelText("Proveedor")).toBeInTheDocument();
    expect(screen.getByLabelText("Prompt / Intención")).toBeInTheDocument();
    expect(screen.getByLabelText("Duración (s)")).toBeInTheDocument();
    expect(screen.getByLabelText("Aspecto")).toBeInTheDocument();
    expect(screen.getByLabelText("Resolución")).toBeInTheDocument();
    expect(screen.getByLabelText("Seed (opcional)")).toBeInTheDocument();
    expect(screen.getByLabelText("Dirección visual")).toBeInTheDocument();
    expect(screen.getByLabelText("Dirección de cámara")).toBeInTheDocument();
    expect(screen.getByLabelText("Evitar (restricciones)")).toBeInTheDocument();
    expect(screen.getByText("Preparar generación")).toBeInTheDocument();
  });

  it("shows reference asset selector for I2V mode", async () => {
    (api.getGenerationCapability as vi.Mock).mockResolvedValue(mockCapability);
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    await waitFor(() => {
      // Change to image_to_video
      fireEvent.click(screen.getByLabelText("Modo"));
      fireEvent.click(screen.getByText("Imagen a Video"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Asset de referencia (obligatorio para Imagen a Video)")).toBeInTheDocument();
    });
  });

  it("disables Prepare button for I2V without reference asset", async () => {
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    await waitFor(() => {
      fireEvent.click(screen.getByLabelText("Modo"));
      fireEvent.click(screen.getByText("Imagen a Video"));
    });
    
    expect(screen.getByText("Preparar generación")).toBeDisabled();
  });

  it("enables Prepare button for I2V with reference asset", async () => {
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    await waitFor(() => {
      fireEvent.click(screen.getByLabelText("Modo"));
      fireEvent.click(screen.getByText("Imagen a Video"));
    });
    
    fireEvent.click(screen.getByLabelText("Asset de referencia (obligatorio para Imagen a Video)"));
    fireEvent.click(screen.getByText("reference.mp4"));
    
    expect(screen.getByText("Preparar generación")).not.toBeDisabled();
  });

  it("calls prepareGeneration API on submit", async () => {
    (api.prepareGeneration as vi.Mock).mockResolvedValue(mockTask);
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset over mountains" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      expect(api.prepareGeneration).toHaveBeenCalledWith("proj-123", expect.objectContaining({
        mode: "text_to_video",
        provider: "local_ltx",
        prompt_intent: "A sunset over mountains",
      }));
    });
  });

  it("shows estimate step after prepare", async () => {
    (api.prepareGeneration as vi.Mock).mockResolvedValue(mockTask);
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      expect(screen.getByText("Estimación lista")).toBeInTheDocument();
    });
    
    expect(screen.getByText("Coste API:")).toBeInTheDocument();
    expect(screen.getByText("$0.00 (Local)")).toBeInTheDocument();
    expect(screen.getByText("Aprobar y generar")).toBeInTheDocument();
  });

  it("calls approveGenerationTask on approve", async () => {
    (api.prepareGeneration as vi.Mock).mockResolvedValue(mockTask);
    (api.approveGenerationTask as vi.Mock).mockResolvedValue({ ...mockTask, status: "QUEUED" });
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(api.approveGenerationTask).toHaveBeenCalledWith("proj-123", "task-123", "abc123hash");
    });
  });

  it("rejects wrong hash on approve", async () => {
    (api.prepareGeneration as vi.Mock).mockResolvedValue(mockTask);
    (api.approveGenerationTask as vi.Mock).mockRejectedValue(new Error("HASH_MISMATCH: approval hash does not match"));
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText(/HASH_MISMATCH/)).toBeInTheDocument();
    });
  });

  it("shows polling step after approve", async () => {
    (api.prepareGeneration as vi.Mock).mockResolvedValue(mockTask);
    (api.approveGenerationTask as vi.Mock).mockResolvedValue({ ...mockTask, status: "QUEUED" });
    (api.getGenerationTask as vi.Mock).mockResolvedValue({ ...mockTask, status: "RUNNING" });
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Estado de la generación")).toBeInTheDocument();
      expect(screen.getByText("Generando…")).toBeInTheDocument();
    });
  });

  it("renders COMPLETED state with GENERATED/ILLUSTRATIVE badges", async () => {
    const completedTask = {
      ...mockTask,
      status: "COMPLETED",
      output_asset: {
        id: "asset-generated",
        filename: "generated_sunset.mp4",
        kind: "video",
        duration_us: 5_000_000,
        width: 1920,
        height: 1080,
        has_audio: true,
        fingerprint: "gen123",
        metadata: {
          asset_provenance: {
            origin: "GENERATED",
            semantic_role: "ILLUSTRATIVE",
            parent_asset_id: null,
            root_source_asset_id: null,
          },
        },
      },
    };
    
    (api.prepareGeneration as vi.Mock).mockResolvedValue(mockTask);
    (api.approveGenerationTask as vi.Mock).mockResolvedValue({ ...mockTask, status: "QUEUED" });
    (api.getGenerationTask as vi.Mock).mockResolvedValue(completedTask);
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} onTaskCompleted={onTaskCompleted} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Completado")).toBeInTheDocument();
      expect(screen.getByText((content) => content.includes("GENERATED"))).toBeInTheDocument();
      expect(screen.getByText((content) => content.includes("ILLUSTRATIVE"))).toBeInTheDocument();
      expect(screen.getByText((content) => content.includes("No evidencia factual"))).toBeInTheDocument();
    });
  });

  it("calls cancelGenerationTask on cancel", async () => {
    (api.prepareGeneration as vi.Mock).mockResolvedValue(mockTask);
    (api.approveGenerationTask as vi.Mock).mockResolvedValue({ ...mockTask, status: "QUEUED" });
    (api.getGenerationTask as vi.Mock).mockResolvedValue({ ...mockTask, status: "RUNNING" });
    (api.cancelGenerationTask as vi.Mock).mockResolvedValue({ ...mockTask, status: "CANCELLED" });
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Cancelar tarea"));
    });
    
    await waitFor(() => {
      expect(api.cancelGenerationTask).toHaveBeenCalledWith("proj-123", "task-123");
    });
  });

  it("does not auto-switch provider to cloud when local unavailable", async () => {
    (api.getGenerationCapability as vi.Mock).mockResolvedValue({
      ...mockCapability,
      local_ltx: {
        ...mockCapability.local_ltx,
        hardware_supported: false,
        runtime_ready: false,
        hardware_message: "No NVIDIA GPU",
        hardware_reason: "NO_NVIDIA_GPU",
        physical_validation: "DEFERRED",
      },
    });
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    // Provider selector should still show local_ltx as option but disabled
    expect(screen.getByLabelText("Proveedor")).toBeInTheDocument();
    // Should NOT automatically switch to runway/luma
    const providerButton = screen.getByLabelText("Proveedor");
    expect(providerButton).toHaveTextContent("LTX-Video (Local, Gratis)");
  });

  it("shows cloud providers as paid options but does not auto-select", async () => {
    (api.getGenerationCapability as vi.Mock).mockResolvedValue({
      ...mockCapability,
      cloud_providers: [
        {
          provider: "runway",
          model: "gen4.5",
          supported: true,
          available: true,
          ready: true,
          physical_validation: "N/A",
          message: "Proveedor cloud disponible",
          paid: true,
          billable: true,
          requires_api_key: true,
          enabled_by_default: false,
        },
        {
          provider: "luma",
          model: "ray-3.2",
          supported: true,
          available: true,
          ready: true,
          physical_validation: "N/A",
          message: "Proveedor cloud disponible",
          paid: true,
          billable: true,
          requires_api_key: true,
          enabled_by_default: false,
        },
      ],
    });
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    // Cloud providers should be visible in capability step (before clicking continue)
    await waitFor(() => {
      expect(screen.getByText((content) => content.includes("Runway Gen-4.5 (Pago)"))).toBeInTheDocument();
      expect(screen.getByText((content) => content.includes("Luma Ray 3.2 (Pago)"))).toBeInTheDocument();
    });
    
    fireEvent.click(screen.getByText("Continuar a configuración"));
    
    // Provider selector should have cloud options but default to local_ltx
    fireEvent.click(screen.getByLabelText("Proveedor"));
    
    // Wait for dropdown to open
    await waitFor(() => {
      expect(screen.getByText((content) => content.includes("Runway Gen-4.5 (Pago)"))).toBeInTheDocument();
      expect(screen.getByText((content) => content.includes("Luma Ray 3.2 (Pago)"))).toBeInTheDocument();
    });
    
    // Should NOT automatically switch to cloud - still on local_ltx
    const providerButton = screen.getByLabelText("Proveedor");
    expect(providerButton).toHaveTextContent("LTX-Video (Local, Gratis)");
  });

  it("shows no evidence eligible warning for GENERATED assets", async () => {
    const completedTask = {
      ...mockTask,
      status: "COMPLETED",
      output_asset: {
        id: "asset-generated",
        filename: "generated.mp4",
        kind: "video",
        duration_us: 5_000_000,
        width: 1920,
        height: 1080,
        has_audio: true,
        fingerprint: "gen123",
        metadata: {
          asset_provenance: {
            origin: "GENERATED",
            semantic_role: "ILLUSTRATIVE",
          },
        },
      },
    };
    
    (api.prepareGeneration as vi.Mock).mockResolvedValue(mockTask);
    (api.approveGenerationTask as vi.Mock).mockResolvedValue({ ...mockTask, status: "QUEUED" });
    (api.getGenerationTask as vi.Mock).mockResolvedValue(completedTask);
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} onTaskCompleted={onTaskCompleted} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText((content) => content.includes("No evidencia factual"))).toBeInTheDocument();
    });
  });

  it("cleans up polling on unmount", async () => {
    (api.prepareGeneration as vi.Mock).mockResolvedValue(mockTask);
    (api.approveGenerationTask as vi.Mock).mockResolvedValue({ ...mockTask, status: "QUEUED" });
    (api.getGenerationTask as vi.Mock).mockResolvedValue({ ...mockTask, status: "RUNNING" });
    
    const { unmount } = render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Generando…")).toBeInTheDocument();
    });
    
    unmount();
    
    // Should not throw and should clean up properly
  });

  it("closes panel on backdrop click", async () => {
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    // Click on overlay (backdrop)
    fireEvent.click(screen.getByTestId("generation-panel-overlay"));
    
    expect(onClose).toHaveBeenCalled();
  });

  it("does not close panel when clicking inside content", async () => {
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    // Click on content area
    fireEvent.click(screen.getByTestId("generation-panel-content"));
    
    expect(onClose).not.toHaveBeenCalled();
  });

  it("provider selector filters based on capability - local unavailable", async () => {
    (api.getGenerationCapability as vi.Mock).mockResolvedValue({
      ...mockCapability,
      local_ltx: {
        ...mockCapability.local_ltx,
        hardware_supported: false,
        runtime_ready: false,
        hardware_message: "No NVIDIA GPU",
        hardware_reason: "NO_NVIDIA_GPU",
        physical_validation: "DEFERRED",
      },
    });
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    // Provider selector should still show local_ltx as option but disabled
    expect(screen.getByLabelText("Proveedor")).toBeInTheDocument();
    // Should NOT automatically switch to runway/luma
    const providerButton = screen.getByLabelText("Proveedor");
    expect(providerButton).toHaveTextContent("LTX-Video (Local, Gratis)");
  });

it("provider selector shows cloud options when configured", async () => {
    (api.getGenerationCapability as vi.Mock).mockResolvedValue({
      ...mockCapability,
      cloud_providers: [
        {
          provider: "runway",
          model: "gen4.5",
          supported: true,
          available: true,
          ready: true,
          physical_validation: "N/A",
          message: "Proveedor cloud disponible",
          paid: true,
          billable: true,
          requires_api_key: true,
          enabled_by_default: false,
        },
        {
          provider: "luma",
          model: "ray-3.2",
          supported: true,
          available: true,
          ready: true,
          physical_validation: "N/A",
          message: "Proveedor cloud disponible",
          paid: true,
          billable: true,
          requires_api_key: true,
          enabled_by_default: false,
        },
      ],
    });

    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);

    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });

    // Provider selector should have cloud options but default to local_ltx
    fireEvent.click(screen.getByLabelText("Proveedor"));
    expect(screen.getByText("Runway Gen-4.5 (Pago)")).toBeInTheDocument();
    expect(screen.getByText("Luma Ray 3.2 (Pago)")).toBeInTheDocument();

    // Should NOT automatically switch to cloud - still on local_ltx
    const providerButton = screen.getByLabelText("Proveedor");
    expect(providerButton).toHaveTextContent("LTX-Video (Local, Gratis)");
  });

  it("shows 'Insertar al final de la pista Video' button after COMPLETED", async () => {
    const completedTask = {
      ...mockTask,
      status: "COMPLETED",
      output_asset: {
        id: "asset-generated",
        filename: "generated_sunset.mp4",
        kind: "video",
        duration_us: 5_000_000,
        width: 1920,
        height: 1080,
        has_audio: true,
        fingerprint: "gen123",
        metadata: {
          asset_provenance: {
            origin: "GENERATED",
            semantic_role: "ILLUSTRATIVE",
            parent_asset_id: null,
            root_source_asset_id: null,
          },
        },
      },
    };
    
    (api.prepareGeneration as vi.Mock).mockResolvedValue(mockTask);
    (api.approveGenerationTask as vi.Mock).mockResolvedValue({ ...mockTask, status: "QUEUED" });
    (api.getGenerationTask as vi.Mock).mockResolvedValue(completedTask);
    
    render(<GenerationPanel project={mockProjectWithTrack} assets={mockAssets} onClose={onClose} onTaskCompleted={onTaskCompleted} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Completado")).toBeInTheDocument();
      expect(screen.getByText("Insertar al final de la pista Video")).toBeInTheDocument();
    });
  });

  it("shows GENERATED and ILLUSTRATIVE badges in COMPLETED state", async () => {
    const completedTask = {
      ...mockTask,
      status: "COMPLETED",
      output_asset: {
        id: "asset-generated",
        filename: "generated.mp4",
        kind: "video",
        duration_us: 5_000_000,
        width: 1920,
        height: 1080,
        has_audio: true,
        fingerprint: "gen123",
        metadata: {
          asset_provenance: {
            origin: "GENERATED",
            semantic_role: "ILLUSTRATIVE",
          },
        },
      },
    };
    
    (api.prepareGeneration as vi.Mock).mockResolvedValue(mockTask);
    (api.approveGenerationTask as vi.Mock).mockResolvedValue({ ...mockTask, status: "QUEUED" });
    (api.getGenerationTask as vi.Mock).mockResolvedValue(completedTask);
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} onTaskCompleted={onTaskCompleted} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText((content) => content.includes("GENERATED"))).toBeInTheDocument();
      expect(screen.getByText((content) => content.includes("ILLUSTRATIVE"))).toBeInTheDocument();
      expect(screen.getByText((content) => content.includes("No evidencia factual"))).toBeInTheDocument();
    });
  });

  it("Volver button clears task, estimate, and requestHash", async () => {
    (api.prepareGeneration as vi.Mock).mockResolvedValue(mockTask);
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      expect(screen.getByText("Estimación lista")).toBeInTheDocument();
    });
    
    fireEvent.click(screen.getByText("Volver"));
    
    await waitFor(() => {
      expect(screen.getByText("Configurar Generación")).toBeInTheDocument();
    });
    
    // After going back to config, task should be null
    expect(screen.queryByText("Preparar generación")).toBeInTheDocument();
  });

  it("shows cancel_requested in polling step", async () => {
    (api.prepareGeneration as vi.Mock).mockResolvedValue(mockTask);
    (api.approveGenerationTask as vi.Mock).mockResolvedValue({ ...mockTask, status: "QUEUED" });
    (api.getGenerationTask as vi.Mock).mockResolvedValue({ ...mockTask, status: "RUNNING", cancel_requested: true });
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText((content) => content.includes("Cancelación solicitada"))).toBeInTheDocument();
    });
  });

  it("does not auto-switch provider to cloud when local unavailable", async () => {
    (api.getGenerationCapability as vi.Mock).mockResolvedValue({
      ...mockCapability,
      local_ltx: {
        ...mockCapability.local_ltx,
        hardware_supported: false,
        runtime_ready: false,
        hardware_message: "No NVIDIA GPU",
        hardware_reason: "NO_NVIDIA_GPU",
        physical_validation: "DEFERRED",
      },
      cloud_providers: [
        {
          provider: "runway",
          model: "gen4.5",
          supported: true,
          available: true,
          ready: true,
          physical_validation: "N/A",
          message: "Proveedor cloud disponible",
          paid: true,
          billable: true,
          requires_api_key: true,
          enabled_by_default: false,
        },
      ],
    });
    
    render(<GenerationPanel project={mockProject} assets={mockAssets} onClose={onClose} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    // Provider selector should still show local_ltx as option but disabled
    expect(screen.getByLabelText("Proveedor")).toBeInTheDocument();
    // Should NOT automatically switch to runway/luma
    const providerButton = screen.getByLabelText("Proveedor");
    expect(providerButton).toHaveTextContent("LTX-Video (Local, Gratis)");
  });
});

describe("GenerationPanel - Insert Generated Asset", () => {
  const onClose = vi.fn();
  const onInserted = vi.fn();
  const onTaskCompleted = vi.fn();

const completedTask = {
    ...mockTask,
    status: "COMPLETED",
    output_asset: {
      id: "asset-generated",
      filename: "generated_sunset.mp4",
      kind: "video",
      duration_us: 5_000_000,
      width: 1920,
      height: 1080,
      has_audio: true,
      fingerprint: "gen123",
      metadata: {
        asset_provenance: {
          origin: "GENERATED",
          semantic_role: "ILLUSTRATIVE",
          parent_asset_id: null,
          root_source_asset_id: null,
        },
      },
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (api.getGenerationCapability as ReturnType<typeof vi.fn>).mockResolvedValue(mockCapability);
    (api.prepareGeneration as ReturnType<typeof vi.fn>).mockResolvedValue(mockTask);
    (api.approveGenerationTask as ReturnType<typeof vi.fn>).mockResolvedValue({ ...mockTask, status: "QUEUED" });
    (api.getGenerationTask as ReturnType<typeof vi.fn>).mockResolvedValue(completedTask);
    (api.editorOperation as ReturnType<typeof vi.fn>).mockResolvedValue({ version: 2, timeline: { duration_us: 35_000_000, tracks: [] } });
    (api.editorOperation as ReturnType<typeof vi.fn>).mockClear();
  });

  it("shows destination visibly before insert", async () => {
    render(<GenerationPanel project={mockProjectWithTrack} assets={mockAssets} onClose={onClose} onTaskCompleted={onTaskCompleted} onInserted={onInserted} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Completado")).toBeInTheDocument();
    });
    
    // Verify destination is shown
    await waitFor(() => {
      expect(screen.getByText((content) => content.includes("Destino: final de la pista"))).toBeInTheDocument();
      expect(screen.getByText((content) => content.includes("Insertar al final de la pista Video"))).toBeInTheDocument();
      // Check for asset details section
      expect(screen.getByText((content) => content.includes("Asset generado"))).toBeInTheDocument();
      expect(screen.getByText((content) => content.includes("5s") || content.includes("5 s"))).toBeInTheDocument();
    });
  });

  it("shows 'Insertar al final de la pista Video' button text", async () => {
    render(<GenerationPanel project={mockProjectWithTrack} assets={mockAssets} onClose={onClose} onTaskCompleted={onTaskCompleted} onInserted={onInserted} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Completado")).toBeInTheDocument();
    });
    
    await waitFor(() => {
      expect(screen.getByText("Insertar al final de la pista Video")).toBeInTheDocument();
    });
  });

  it("calls editorOperation with correct parameters on insert click", async () => {
    render(<GenerationPanel project={mockProjectWithTrack} assets={mockAssets} onClose={onClose} onTaskCompleted={onTaskCompleted} onInserted={onInserted} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Completado")).toBeInTheDocument();
    });
    
    // Click insert button
    await waitFor(() => {
      fireEvent.click(screen.getByText("Insertar al final de la pista Video"));
    });
    
    // Verify editorOperation was called with correct parameters
    await waitFor(() => {
      expect(api.editorOperation).toHaveBeenCalledWith("proj-123", expect.objectContaining({
        type: "insert",
        parameters: expect.objectContaining({
          asset_id: "asset-generated",
          track_id: "video-1",
          source_in_us: 0,
          source_out_us: 5_000_000,
          timeline_start_us: 30_000_000,
        }),
        base_version: 1,
        idempotency_key: expect.any(String),
      }));
    });
  });

  it("calls onInserted callback after editorOperation succeeds", async () => {
    render(<GenerationPanel project={mockProjectWithTrack} assets={mockAssets} onClose={onClose} onTaskCompleted={onTaskCompleted} onInserted={onInserted} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Completado")).toBeInTheDocument();
    });
    
    // Click insert button
    fireEvent.click(screen.getByText("Insertar al final de la pista Video"));
    
    // Verify onInserted is called after editorOperation
    await waitFor(() => {
      expect(onInserted).toHaveBeenCalled();
    });
  });

  it("calls onClose after onInserted completes", async () => {
    let resolveEditorOp: (value: { version: number; timeline: { duration_us: number; tracks: Array<{ id: string; type: string }> } }) => void;
    let resolveOnInserted: () => void;
    
    const editorOpPromise = new Promise((resolve) => {
      resolveEditorOp = resolve;
    });
    const onInsertedPromise = new Promise((resolve) => {
      resolveOnInserted = resolve;
    });
    
    (api.editorOperation as ReturnType<typeof vi.fn>).mockImplementation(() => editorOpPromise);
    (onInserted as ReturnType<typeof vi.fn>).mockImplementation(() => onInsertedPromise);
    
    render(<GenerationPanel project={mockProjectWithTrack} assets={mockAssets} onClose={onClose} onTaskCompleted={onTaskCompleted} onInserted={onInserted} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Completado")).toBeInTheDocument();
    });
    
    // Click insert button
    fireEvent.click(screen.getByText("Insertar al final de la pista Video"));
    
    // Before editorOperation resolves: onInserted not called, onClose not called
    expect(onInserted).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
    
    // Resolve editorOperation
    resolveEditorOp!({ version: 2, timeline: { duration_us: 35_000_000, tracks: [] } });
    
    // After editorOperation resolves but before onInserted resolves: onInserted called, onClose NOT called
    await waitFor(() => {
      expect(onInserted).toHaveBeenCalledTimes(1);
    });
    expect(onClose).not.toHaveBeenCalled();
    
    // Resolve onInserted
    resolveOnInserted!();
    
    // After onInserted resolves: onClose called
    await waitFor(() => {
      expect(onClose).toHaveBeenCalledTimes(1);
    });
  });

  it("disables insert button while request is pending", async () => {
    // Make editorOperation slow
    let resolveEditorOp: (value: { version: number; timeline: { duration_us: number; tracks: Array<{ id: string; type: string }> } }) => void;
    const editorOpPromise = new Promise((resolve) => {
      resolveEditorOp = resolve;
    });
    (api.editorOperation as ReturnType<typeof vi.fn>).mockImplementation(() => editorOpPromise);
    
    render(<GenerationPanel project={mockProjectWithTrack} assets={mockAssets} onClose={onClose} onTaskCompleted={onTaskCompleted} onInserted={onInserted} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Completado")).toBeInTheDocument();
    });
    
    // Click insert button
    fireEvent.click(screen.getByText("Insertar al final de la pista Video"));
    
    // Button should be disabled while pending
    await waitFor(() => {
      expect(screen.getByText((content) => content.includes("Insertando"))).toBeInTheDocument();
      expect(screen.queryByText("Insertar al final de la pista Video")).not.toBeInTheDocument();
    });
    
    // Resolve the operation
    resolveEditorOp!({ version: 2, timeline: { duration_us: 35_000_000, tracks: [] } });
  });

  it("does not create second operation on double click", async () => {
    render(<GenerationPanel project={mockProjectWithTrack} assets={mockAssets} onClose={onClose} onTaskCompleted={onTaskCompleted} onInserted={onInserted} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Completado")).toBeInTheDocument();
    });
    
    // Click insert button twice rapidly
    fireEvent.click(screen.getByText("Insertar al final de la pista Video"));
    fireEvent.click(screen.getByText("Insertando…")); // Second click on disabled button
    
    // Wait a bit
    await waitFor(() => {
      expect(api.editorOperation).toHaveBeenCalledTimes(1);
    });
  });

  it("shows error if editorOperation fails and does not call onInserted", async () => {
    (api.editorOperation as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("Editor operation failed"));
    
    render(<GenerationPanel project={mockProjectWithTrack} assets={mockAssets} onClose={onClose} onTaskCompleted={onTaskCompleted} onInserted={onInserted} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Completado")).toBeInTheDocument();
    });
    
    // Click insert button
    fireEvent.click(screen.getByText("Insertar al final de la pista Video"));
    
    // Wait for error to be shown
    await waitFor(() => {
      expect(screen.getByText((content) => content.includes("Editor operation failed") || content.includes("No se pudo insertar"))).toBeInTheDocument();
    });
    
    // onInserted should NOT be called on failure
    expect(onInserted).not.toHaveBeenCalled();
  });

  it("shows error and disables insert when no video track exists", async () => {
    const projectWithoutVideoTrack: EditorProject = {
      id: "proj-123",
      name: "Test Project",
      source_job_id: "job-123",
      current_version: 1,
      created_at: "2024-01-01T00:00:00Z",
      updated_at: "2024-01-01T00:00:00Z",
      timeline: { duration_us: 30_000_000, tracks: [] },
      assets: [],
    };
    
    render(<GenerationPanel project={projectWithoutVideoTrack} assets={mockAssets} onClose={onClose} onTaskCompleted={onTaskCompleted} onInserted={onInserted} />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Continuar a configuración"));
    });
    
    fireEvent.change(screen.getByLabelText("Prompt / Intención"), { target: { value: "A sunset" } });
    fireEvent.click(screen.getByText("Preparar generación"));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText("Aprobar y generar"));
    });
    
    await waitFor(() => {
      expect(screen.getByText("Completado")).toBeInTheDocument();
    });
    
    // Should show error about no video track
    await waitFor(() => {
      expect(screen.getByText((content) => content.includes("No hay pista de video"))).toBeInTheDocument();
    });
    
    // Insert button should be disabled
    await waitFor(() => {
      expect(screen.getByText("Insertar al final de la pista Video")).toBeDisabled();
    });
  });
});
