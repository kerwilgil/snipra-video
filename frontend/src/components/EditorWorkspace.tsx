import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import { Select } from "./Select";
import { NvidiaModelField } from "./NvidiaModelField";
import { HighlightInspector } from "./HighlightInspector";
import { EnhancementPanel } from "./EnhancementPanel";
import { GenerationPanel } from "./GenerationPanel";
import { EditorIcon } from "./EditorIcon";
import type {
  EditorEvent,
  EditorPreviewPlan,
  EditorProject,
  EditorTimelineClip,
  EditorTranscriptSegment,
  ShortBatchPlanResponse,
  EditorAsset,
  EnhancementTask,
  GenerationTask,
} from "../lib/types";

const us = (value: number) =>
  `${Math.floor(value / 60_000_000)}:${String(Math.floor(value / 1_000_000) % 60).padStart(2, "0")}.${String(Math.floor((value % 1_000_000) / 100_000)).padStart(1, "0")}`;
const operationKey = () => crypto.randomUUID();
const number = (value: string) => Number(value);
const field = "field mt-1 w-full px-2 py-1.5 text-xs";
const assetLabel = (filename: string) =>
  filename.replace(/^(?:[a-f0-9]{32}-)+/i, "").replaceAll("+", " ");
const timelineTime = (value: number) => {
  const seconds = Math.max(0, Math.floor(value / 1_000_000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  return hours
    ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
};
const preciseTimelineTime = (value: number) => {
  const milliseconds = Math.max(0, Math.round(value / 1_000));
  const hours = Math.floor(milliseconds / 3_600_000);
  const minutes = Math.floor((milliseconds % 3_600_000) / 60_000);
  const seconds = Math.floor((milliseconds % 60_000) / 1_000);
  const remainder = milliseconds % 1_000;
  const prefix = hours
    ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`
    : String(minutes).padStart(2, "0");
  return `${prefix}:${String(seconds).padStart(2, "0")}.${String(remainder).padStart(3, "0")}`;
};
const clipTimelineEnd = (clip: EditorTimelineClip) =>
  clip.timeline_end_us ??
  clip.timeline_start_us +
    ((clip.source_out_us ?? 0) - (clip.source_in_us ?? 0)) /
      clip.playback_rate;
const AGENT_MODELS: Record<string, string[]> = {
  claude: ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"],
  openai: ["gpt-4o-mini", "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"],
  claude_code: ["claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-4-8"],
  codex: ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"],
};

export function EditorWorkspace({
  project,
  onProject,
  onAssetRemoved,
  dedicated = false,
}: {
  project: EditorProject;
  onProject: (project: EditorProject) => void;
  onAssetRemoved: () => Promise<unknown>;
  dedicated?: boolean;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const viewerShell = useRef<HTMLDivElement>(null);
  const [plan, setPlan] = useState<EditorPreviewPlan | null>(null),
    [playhead, setPlayhead] = useState(0),
    [selected, setSelected] = useState<string | null>(null),
    [playing, setPlaying] = useState(false);
  const [agentCollapsed, setAgentCollapsed] = useState(false);
  const [agentWidth, setAgentWidth] = useState(() =>
    Number(localStorage.getItem("snipra-editor-agent-width")) || 340,
  );
  const agentSplitter = useRef<{ startX: number; startWidth: number } | null>(null);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [previewMode, setPreviewMode] = useState<"fit" | "fill">("fit");
  const selectItem = (value: string | null) => { setSelected(value); setInspectorOpen(Boolean(value)); };
  const [timelineHeight, setTimelineHeight] = useState(() =>
    Number(localStorage.getItem("snipra-editor-timeline-height")) || (window.innerHeight <= 800 ? 300 : 360),
  );
  const splitter = useRef<{ startY: number; startHeight: number } | null>(null);
  const [mediaWidth, setMediaWidth] = useState(() =>
    Number(localStorage.getItem("snipra-editor-media-width")) || 340,
  );
  const mediaSplitter = useRef<{ startX: number; startWidth: number } | null>(null);
  const highlightRequest = useRef(0);
  const [message, setMessage] = useState(""),
    [events, setEvents] = useState<EditorEvent[]>([]),
    [transcript, setTranscript] = useState<EditorTranscriptSegment[]>([]),
    [agentRequest, setAgentRequest] = useState(""),
    [agentState, setAgentState] = useState("");
  const [highlightPlan, setHighlightPlan] = useState<ShortBatchPlanResponse | null>(null);
  const [highlightBusy, setHighlightBusy] = useState(false);
  const [highlightError, setHighlightError] = useState<string | null>(null);
  const [highlightCount, setHighlightCount] = useState(3);
  const [cropDraft, setCropDraft] = useState({
    x: 0,
    y: 0,
    width: 1,
    height: 1,
  });
  const active = plan?.segments.find(
    (s) => s.timeline_start_us <= playhead && playhead < s.timeline_end_us,
  );
  const videoTrack = project.timeline.tracks.find((t) => t.type === "video");
  const textTrack = project.timeline.tracks.find((t) => t.type === "text");
  const allContent = plan?.content ?? [];
  const selectedItem = allContent.find((item) => item.id === selected);
  const selectedClip = project.timeline.tracks
    .flatMap((track) => track.clips)
    .find((clip) => clip.id === selected && !clip.kind);
  const canvas = plan?.output_canvas ?? { width: 1280, height: 720 };
  useEffect(() => {
    const shell = viewerShell.current;
    if (!shell) return;
    const fitCanvas = () => {
      const availableWidth = Math.max(0, shell.clientWidth - 20);
      const availableHeight = Math.max(0, shell.clientHeight - 20);
      const scale = previewMode === "fill"
        ? Math.max(availableWidth / canvas.width, availableHeight / canvas.height)
        : Math.min(availableWidth / canvas.width, availableHeight / canvas.height);
      shell.style.setProperty("--preview-width", `${Math.floor(canvas.width * scale)}px`);
      shell.style.setProperty("--preview-height", `${Math.floor(canvas.height * scale)}px`);
    };
    const observer = new ResizeObserver(fitCanvas);
    observer.observe(shell);
    fitCanvas();
    return () => observer.disconnect();
  }, [canvas.width, canvas.height, previewMode]);
  const activeContent = allContent.filter(
    (item) =>
      item.timeline_start_us <= playhead &&
      playhead < (item.timeline_end_us ?? 0),
  );
  const refresh = useCallback(async () => {
    const [p, preview, e] = await Promise.all([
      api.getEditorProject(project.id),
      api.editorPreviewPlan(project.id),
      api.editorEvents(project.id),
    ]);
    onProject(p);
    setPlan(preview);
    setEvents(e.best_event_pool);
  }, [onProject, project.id]);
  useEffect(() => {
    void Promise.resolve()
      .then(refresh)
      .catch((e) =>
        setMessage(
          e instanceof Error ? e.message : "No se pudo abrir el proyecto",
        ),
      );
  }, [refresh]);
  useEffect(() => {
    void api
      .editorTranscript(project.id, 0, project.timeline.duration_us)
      .then((r) => setTranscript(r.segments));
  }, [project.id, project.timeline.duration_us]);
  useEffect(() => {
    if (!selectedClip) return;
    const timer = window.setTimeout(() =>
      setCropDraft(
        selectedClip.transform?.crop ?? { x: 0, y: 0, width: 1, height: 1 },
      ),
    );
    return () => window.clearTimeout(timer);
  }, [selectedClip]);
  useEffect(() => {
    if (!video.current || !active) return;
    const source =
      active.source_in_us / 1e6 +
      ((playhead - active.timeline_start_us) / 1e6) * active.playback_rate;
    if (video.current.dataset.clipId !== active.clip_id) {
      video.current.src = active.video_url;
      video.current.dataset.clipId = active.clip_id;
    }
    video.current.playbackRate = active.playback_rate;
    if (Math.abs(video.current.currentTime - source) >= 0.25)
      video.current.currentTime = source;
    const clipTime = playhead - active.timeline_start_us,
      duration = active.timeline_end_us - active.timeline_start_us,
      audio = active.audio;
    let gain = audio.muted ? 0 : Math.min(1, audio.volume);
    if (audio.fade_in_us) gain *= Math.min(1, clipTime / audio.fade_in_us);
    if (audio.fade_out_us)
      gain *= Math.min(1, Math.max(0, duration - clipTime) / audio.fade_out_us);
    video.current.volume = gain;
    if (playing) void video.current.play();
  }, [active, playhead, playing]);
  const apply = async (type: string, parameters: Record<string, unknown>) => {
    try {
      const r = await api.editorOperation(project.id, {
        type,
        parameters,
        base_version: project.current_version,
        idempotency_key: operationKey(),
      });
      onProject({
        ...project,
        current_version: r.version,
        timeline: r.timeline,
      });
      await refresh();
      setMessage("Cambio guardado. Undo disponible.");
    } catch (e) {
      setMessage(e instanceof Error ? e.message : "La operación falló");
    }
  };
  const undo = async () => {
    try {
      const r = await api.editorUndo(project.id);
      onProject({
        ...project,
        current_version: r.version,
        timeline: r.timeline,
      });
      await refresh();
      setMessage("Último cambio revertido.");
    } catch (e) {
      setMessage(e instanceof Error ? e.message : "No se pudo deshacer");
    }
  };
  const editContent = (kind: "caption" | "text", form: HTMLFormElement) => {
    const data = new FormData(form),
      position = {
        x: number(String(data.get("x"))),
        y: number(String(data.get("y"))),
      },
      style = {
        font_size: number(String(data.get("font_size"))),
        color: String(data.get("color")),
      };
    const parameters = {
      text: String(data.get("text")),
      start_us: Math.round(number(String(data.get("start"))) * 1e6),
      end_us: Math.round(number(String(data.get("end"))) * 1e6),
      position,
      style,
    };
    void apply(
      selectedItem ? `edit_${kind}` : `add_${kind}`,
      selectedItem
        ? { item_id: selectedItem.id, ...parameters }
        : { track_id: textTrack?.id, ...parameters },
    );
  };
  const validCrop =
    cropDraft.x >= 0 &&
    cropDraft.y >= 0 &&
    cropDraft.width > 0 &&
    cropDraft.height > 0 &&
    cropDraft.x + cropDraft.width <= 1 &&
    cropDraft.y + cropDraft.height <= 1;
  const togglePlayback = async () => {
    if (!video.current) return;
    if (playing) {
      video.current.pause();
      setPlaying(false);
    } else if (active) {
      await video.current.play();
      setPlaying(true);
    }
  };
  const seek = (value: number) =>
    setPlayhead(Math.max(0, Math.min(project.timeline.duration_us, value)));
  const analyzeHighlights = async () => {
    const requestId = ++highlightRequest.current;
    setHighlightBusy(true);
    setHighlightError(null);
    try {
      const next = await api.createShortBatch(project.id, { requested_count: highlightCount, intent: "highlight" });
      if (requestId === highlightRequest.current) setHighlightPlan(next);
    } catch (error) {
      if (requestId === highlightRequest.current) setHighlightError(error instanceof Error ? error.message : "No se pudieron analizar los highlights");
    } finally {
      if (requestId === highlightRequest.current) setHighlightBusy(false);
    }
  };
  useEffect(() => {
    const removeSelected = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        !selected ||
        target?.matches("input, textarea, select, [contenteditable='true']") ||
        (event.key !== "Delete" && event.key !== "Backspace")
      ) return;
      event.preventDefault();
      void apply("delete", { clip_id: selected });
    };
    window.addEventListener("keydown", removeSelected);
    return () => window.removeEventListener("keydown", removeSelected);
  // The shortcut listener deliberately reads the current operation closure.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, project.current_version]);
  const onSplitterPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    splitter.current = { startY: event.clientY, startHeight: timelineHeight };
  };
  const onSplitterPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!splitter.current) return;
    const next = Math.max(220, Math.min(window.innerHeight - 230, splitter.current.startHeight - (event.clientY - splitter.current.startY)));
    setTimelineHeight(next);
  };
  const onSplitterPointerUp = () => {
    if (!splitter.current) return;
    localStorage.setItem("snipra-editor-timeline-height", String(timelineHeight));
    splitter.current = null;
  };
  const onMediaSplitterPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    mediaSplitter.current = { startX: event.clientX, startWidth: mediaWidth };
  };
  const onMediaSplitterPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!mediaSplitter.current) return;
    const maxWidth = Math.min(620, window.innerWidth * 0.45);
    setMediaWidth(Math.max(220, Math.min(maxWidth, mediaSplitter.current.startWidth + event.clientX - mediaSplitter.current.startX)));
  };
  const onMediaSplitterPointerUp = () => {
    if (!mediaSplitter.current) return;
    localStorage.setItem("snipra-editor-media-width", String(mediaWidth));
    mediaSplitter.current = null;
  };
  const onAgentSplitterPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    agentSplitter.current = { startX: event.clientX, startWidth: agentWidth };
  };
  const onAgentSplitterPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!agentSplitter.current) return;
    const next = Math.max(240, Math.min(480, agentSplitter.current.startWidth + event.clientX - agentSplitter.current.startX));
    setAgentWidth(next);
  };
  const onAgentSplitterPointerUp = () => {
    if (!agentSplitter.current) return;
    localStorage.setItem("snipra-editor-agent-width", String(agentWidth));
    agentSplitter.current = null;
  };
  const tick = () => {
    if (video.current && active)
      seek(
        active.timeline_start_us +
          Math.round(
            ((video.current.currentTime - active.source_in_us / 1e6) /
              active.playback_rate) *
              1e6,
          ),
      );
  };
  const transformStyle = useMemo(() => {
    const t = active?.transform ?? { x: 0, y: 0, scale: 1 },
      c = t.crop ?? { x: 0, y: 0, width: 1, height: 1 };
    return {
      left: `${50 + t.x * 50}%`,
      top: `${50 + t.y * 50}%`,
      width: `${c.width * t.scale * 100}%`,
      height: `${c.height * t.scale * 100}%`,
      transform: "translate(-50%, -50%)",
      overflow: "hidden" as const,
    };
  }, [active]);
  return (
    <div
      className={`editor-workspace ${dedicated ? "dedicated-workspace" : ""} ${agentCollapsed ? "agent-collapsed" : ""} ${inspectorOpen ? "inspector-open" : ""}`}
      style={{ "--timeline-height": `${timelineHeight}px`, "--media-width": `${mediaWidth}px`, "--agent-width": `${agentWidth}px` } as React.CSSProperties}
    >
      <aside className={`editor-agent-sidebar ${agentCollapsed ? "is-collapsed" : ""}`}>
        <button className="agent-collapse" onClick={() => setAgentCollapsed((value) => !value)} aria-label={agentCollapsed ? "Expandir Agente IA" : "Colapsar Agente IA"} title={agentCollapsed ? "Expandir Agente IA" : "Colapsar Agente IA"}>{agentCollapsed ? "›" : "‹"}</button>
        <AgentPanel projectId={project.id} agentRequest={agentRequest} setAgentRequest={setAgentRequest} agentState={agentState} setAgentState={setAgentState} refresh={refresh} />
      </aside>
      <div
        className="agent-splitter"
        role="separator"
        aria-label="Redimensionar panel del Agente IA"
        aria-orientation="vertical"
        aria-valuemin={240}
        aria-valuemax={480}
        aria-valuenow={agentWidth}
        tabIndex={0}
        onPointerDown={onAgentSplitterPointerDown}
        onPointerMove={onAgentSplitterPointerMove}
        onPointerUp={onAgentSplitterPointerUp}
        onKeyDown={(event) => {
          if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
          event.preventDefault();
          const next = Math.max(240, Math.min(480, agentWidth + (event.key === "ArrowRight" ? 20 : -20)));
          setAgentWidth(next);
          localStorage.setItem("snipra-editor-agent-width", String(next));
        }}
      />
      <div className="editor-main">
      <div className={`editor-upper ${inspectorOpen ? "has-inspector" : ""}`}>
      <MediaPanel
        project={project}
        videoTrack={videoTrack}
        apply={apply}
        refresh={refresh}
        onAssetRemoved={onAssetRemoved}
        events={events}
        transcript={transcript}
        highlightPlan={highlightPlan}
        highlightBusy={highlightBusy}
        highlightError={highlightError}
        highlightCount={highlightCount}
        setHighlightCount={setHighlightCount}
        analyzeHighlights={analyzeHighlights}
        seek={seek}
      />
      <div
        className="media-splitter"
        role="separator"
        aria-label="Redimensionar biblioteca"
        aria-orientation="vertical"
        aria-valuemin={220}
        aria-valuemax={620}
        aria-valuenow={mediaWidth}
        tabIndex={0}
        onPointerDown={onMediaSplitterPointerDown}
        onPointerMove={onMediaSplitterPointerMove}
        onPointerUp={onMediaSplitterPointerUp}
        onKeyDown={(event) => {
          if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
          event.preventDefault();
          const next = Math.max(220, Math.min(620, mediaWidth + (event.key === "ArrowRight" ? 24 : -24)));
          setMediaWidth(next);
          localStorage.setItem("snipra-editor-media-width", String(next));
        }}
      />
      <section className="editor-stage">
        <div className="preview-heading"><span>Preview</span><button type="button" className={inspectorOpen ? "active" : ""} aria-label={inspectorOpen ? "Cerrar inspector" : "Abrir inspector"} title={inspectorOpen ? "Cerrar inspector" : "Abrir inspector"} aria-pressed={inspectorOpen} onClick={() => setInspectorOpen((value) => !value)}>Propiedades</button></div>
        <div ref={viewerShell} className={`viewer-shell ${previewMode === "fill" ? "viewer-fill" : ""}`}>
          <div
            className="viewer-canvas"
            style={{ aspectRatio: `${canvas.width}/${canvas.height}` }}
          >
            <div className="absolute" style={transformStyle}>
              {active && (
                <video
                  ref={video}
                  onTimeUpdate={tick}
                  onEnded={() => {
                    setPlaying(false);
                    seek(active.timeline_end_us);
                  }}
                  className="absolute max-w-none"
                  style={{
                    width: `${100 / (active.transform.crop?.width ?? 1)}%`,
                    height: `${100 / (active.transform.crop?.height ?? 1)}%`,
                    left: `${(-(active.transform.crop?.x ?? 0) / (active.transform.crop?.width ?? 1)) * 100}%`,
                    top: `${(-(active.transform.crop?.y ?? 0) / (active.transform.crop?.height ?? 1)) * 100}%`,
                  }}
                  controls={false}
                />
              )}
            </div>
            {activeContent.map((item) => (
              <div
                key={item.id}
                className="pointer-events-none absolute whitespace-pre-wrap text-center font-semibold drop-shadow"
                style={{
                  left: `${((item.position?.x ?? 0) + 1) * 50}%`,
                  top: `${((item.position?.y ?? 0.75) + 1) * 50}%`,
                  transform: "translate(-50%, -50%)",
                  color: item.style?.color ?? "#fff",
                  fontSize: `${((item.style?.font_size ?? 42) / canvas.height) * 100}vw`,
                }}
              >
                {item.text}
              </div>
            ))}
          </div>
        </div>
        <Transport
          playing={playing}
          playhead={playhead}
          duration={project.timeline.duration_us}
          toggle={togglePlayback}
          seek={seek}
          previewMode={previewMode}
          setPreviewMode={setPreviewMode}
        />
      </section>
      {inspectorOpen && <aside className="editor-inspector-drawer"><PropertiesInspector
        project={project}
        selectedClip={selectedClip}
        selectedItem={selectedItem}
        selected={selected}
        setSelected={selectItem}
        apply={apply}
        cropDraft={cropDraft}
        setCropDraft={setCropDraft}
        validCrop={validCrop}
        playhead={playhead}
        editContent={editContent}
        allContent={allContent}
        message={message}
      /></aside>}
      </div>
      <div className="timeline-splitter" role="separator" aria-label="Redimensionar timeline" onPointerDown={onSplitterPointerDown} onPointerMove={onSplitterPointerMove} onPointerUp={onSplitterPointerUp} />
      <TimelinePanel
        project={project}
        selected={selected}
        setSelected={selectItem}
        playhead={playhead}
        seek={seek}
        apply={apply}
        undo={undo}
        togglePlayback={togglePlayback}
      />
      </div>
    </div>
  );
}

function MediaPanel({
  project,
  videoTrack,
  apply,
  refresh,
  onAssetRemoved,
  events,
  transcript,
  highlightPlan,
  highlightBusy,
  highlightError,
  highlightCount,
  setHighlightCount,
  analyzeHighlights,
  seek,
}: {
  project: EditorProject;
  videoTrack: EditorProject["timeline"]["tracks"][number] | undefined;
  apply: (type: string, params: Record<string, unknown>) => Promise<void>;
  refresh: () => Promise<void>;
  onAssetRemoved: () => Promise<unknown>;
  events: EditorEvent[];
  transcript: EditorTranscriptSegment[];
  highlightPlan: ShortBatchPlanResponse | null;
  highlightBusy: boolean;
  highlightError: string | null;
  highlightCount: number;
  setHighlightCount: (value: number) => void;
  analyzeHighlights: () => void;
  seek: (value: number) => void;
}) {
  const [tab, setTab] = useState<"media" | "events" | "highlights" | "transcript" | "generar">("media");
  const [assetMenu, setAssetMenu] = useState<string | null>(null);
  const [enhancementAsset, setEnhancementAsset] = useState<EditorAsset | null>(null);
  const remove = async (assetId: string) => {
    try {
      await api.removeEditorAsset(project.id, assetId);
      await refresh();
      await onAssetRemoved();
    } catch (error) {
      window.alert(
        error instanceof Error ? error.message : "No se pudo quitar el medio",
      );
    }
  };

  const openEnhancement = (asset: EditorAsset) => {
    setEnhancementAsset(asset);
  };

  const closeEnhancement = async () => {
    setEnhancementAsset(null);
    await refresh();
  };

  const handleEnhancementCompleted = async (task: EnhancementTask) => {
    if (task.output_asset) {
      await refresh();
    }
  };

  const handleGenerationCompleted = async (task: GenerationTask) => {
    if (task.output_asset) {
      await refresh();
    }
  };
  return (
<aside className="editor-panel media-panel">
      <div className="media-navigation">
        <span className="section-label">Biblioteca</span>
        <div className="editor-tabs" role="tablist" aria-label="Secciones de biblioteca">
          {(["media", "events", "highlights", "transcript", "generar"] as const).map((item) => (
            <button
              key={item}
              type="button"
              className={tab === item ? "active" : ""}
              role="tab"
              aria-selected={tab === item}
              onClick={() => setTab(item)}
            >
              {item === "media" ? "Medios" : item === "events" ? "Eventos" : item === "highlights" ? "Destacados" : item === "transcript" ? "Transcripción" : "Generar"}
            </button>
          ))}
        </div>
      </div>
      {tab === "media" && (
        <>
          <label className="media-import">
            Importar media
            <input
              type="file"
              className="hidden"
              accept="video/*,audio/*"
              onChange={async (e) => {
                const file = e.target.files?.[0];
                if (file) {
                  await api.uploadEditorAsset(project.id, file);
                  await refresh();
                }
              }}
            />
          </label>
          <div className="editor-list">
            {project.assets.map((a) => {
              const provenance = a.metadata?.asset_provenance;
              const isEnhanced = provenance?.origin === "ENHANCED";
              const isGenerated = provenance?.origin === "GENERATED";
              const isIllustrative = provenance?.semantic_role === "ILLUSTRATIVE";
              const isVideo = a.kind === "video";
              return (
                <div className="media-item" key={a.id}>
                <div className="media-item-info">
                  <strong>{assetLabel(a.filename)}</strong>
                  <span>
                    {us(a.duration_us)} · {a.width}×{a.height}
                    {isEnhanced && <span className="enhanced-badge">ENHANCED</span>}
                    {isGenerated && (
                      <>
                        <span className="generated-badge">GENERATED</span>
                        {isIllustrative && <span className="illustrative-badge">ILLUSTRATIVE</span>}
                      </>
                    )}
                  </span>
                </div>
                <div className="media-actions">
                  {videoTrack && <button
                    type="button"
                    className="media-add"
                    aria-label={`Añadir ${assetLabel(a.filename)} a la timeline`}
                    title="Añadir al final de la timeline"
                    onClick={() => void apply("insert", {
                      asset_id: a.id,
                      track_id: videoTrack.id,
                      source_in_us: 0,
                      source_out_us: a.duration_us,
                      timeline_start_us: project.timeline.duration_us,
                    })}
                  >Añadir</button>}
                  <button
                    className="media-more"
                    aria-label={`Opciones para ${assetLabel(a.filename)}`}
                    aria-expanded={assetMenu === a.id}
                    title="Opciones de media"
                    onClick={(event) => {
                      event.stopPropagation();
                      setAssetMenu((current) => current === a.id ? null : a.id);
                    }}
                  >
                    Opciones
                  </button>
                </div>
                {assetMenu === a.id && (
                    <div className="media-menu" role="group" aria-label={`Opciones de ${assetLabel(a.filename)}`}>
                      {isVideo && !isEnhanced && !isGenerated && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setAssetMenu(null);
                            openEnhancement(a);
                          }}
                        >
                          Mejorar
                        </button>
                      )}
                      {isEnhanced && (
                        <span className="media-menu-hint">
                          ✓ Mejorado (FFmpeg)
                        </span>
                      )}
                      {isGenerated && (
                        <span className="media-menu-hint">
                          GENERATED · ILLUSTRATIVE · No evidencia factual
                        </span>
                      )}
                      <button
                        onClick={() => {
                          setAssetMenu(null);
                          void remove(a.id);
                        }}
                      >
                        Quitar del proyecto
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}
      {tab === "events" && (
        <div className="editor-list events-list">
          {events.length ? (
            events.map((event) => (
              <div key={event.id}>
                <strong>{event.title}</strong>
                <span>
                  {us(event.start_us)}–{us(event.end_us)}
                </span>
              </div>
            ))
          ) : (
            <p>Sin eventos disponibles.</p>
          )}
        </div>
      )}
      {tab === "highlights" && <HighlightInspector plan={highlightPlan} loading={highlightBusy} error={highlightError} requestedCount={highlightCount} onRequestedCountChange={setHighlightCount} onAnalyze={analyzeHighlights} onSeek={(seconds) => seek(Math.round(seconds * 1e6))} />}
      {tab === "transcript" && (
        <div className="editor-list transcript-list">
          {transcript.length ? (
            transcript.map((segment, index) => (
              <button
                key={index}
                type="button"
                className="transcript-entry"
                title={segment.evidence.content}
                onClick={() => seek(segment.start_us)}
              >
                <time>{us(segment.start_us)}</time>
                <span>{segment.evidence.content}</span>
              </button>
            ))
          ) : (
            <p>Sin transcripción disponible.</p>
          )}
        </div>
      )}
      {enhancementAsset && (
        <EnhancementPanel
          projectId={project.id}
          asset={enhancementAsset}
          onClose={closeEnhancement}
          onTaskCompleted={handleEnhancementCompleted}
        />
      )}
      {tab === "generar" && (
        <GenerationPanel
          project={project}
          assets={project.assets}
          onClose={() => { setTab("media"); }}
          onTaskCompleted={handleGenerationCompleted}
          onInserted={refresh}
        />
      )}
    </aside>
  );
}

function Transport({
  playing,
  playhead,
  duration,
  toggle,
  seek,
  previewMode,
  setPreviewMode,
}: {
  playing: boolean;
  playhead: number;
  duration: number;
  toggle: () => Promise<void>;
  seek: (value: number) => void;
  previewMode: "fit" | "fill";
  setPreviewMode: (value: "fit" | "fill") => void;
}) {
  return (
    <div className="transport">
      <input
        aria-label="Posición de reproducción"
        className="transport-seek"
        type="range"
        min="0"
        max={Math.max(1, duration)}
        value={playhead}
        onChange={(e) => seek(number(e.target.value))}
      />
      <div className="transport-row">
        <div className="transport-view" role="group" aria-label="Vista del reproductor">
          <button type="button" aria-pressed={previewMode === "fit"} title="Mostrar el cuadro completo" onClick={() => setPreviewMode("fit")}>Ajustar</button>
          <button type="button" aria-pressed={previewMode === "fill"} title="Llenar el visor; los bordes se recortan solo en la vista" onClick={() => setPreviewMode("fill")}>Llenar</button>
        </div>
        <div className="transport-playback">
        <button
          aria-label="Ir al inicio"
          title="Ir al inicio"
          onClick={() => seek(0)}
        >
          <EditorIcon name="start" />
        </button>
        <button
          aria-label={playing ? "Pausar" : "Reproducir"}
          title={playing ? "Pausar" : "Reproducir"}
          className="transport-play"
          onClick={() => void toggle()}
        >
          <EditorIcon name={playing ? "pause" : "play"} />
        </button>
        <button
          aria-label="Ir al final"
          title="Ir al final"
          onClick={() => seek(duration)}
        >
          <EditorIcon name="end" />
        </button>
        </div>
        <span className="num">
          {us(playhead)} <i>/</i> {us(duration)}
        </span>
      </div>
    </div>
  );
}

type InspectorProps = {
  project: EditorProject;
  selectedClip: EditorTimelineClip | undefined;
  selectedItem: EditorTimelineClip | undefined;
  selected: string | null;
  setSelected: (value: string | null) => void;
  apply: (type: string, params: Record<string, unknown>) => Promise<void>;
  cropDraft: { x: number; y: number; width: number; height: number };
  setCropDraft: (value: { x: number; y: number; width: number; height: number }) => void;
  validCrop: boolean;
  playhead: number;
  editContent: (kind: "caption" | "text", form: HTMLFormElement) => void;
  allContent: EditorTimelineClip[];
  message: string;
};

export function AgentPanel({
  projectId,
  agentRequest,
  setAgentRequest,
  agentState,
  setAgentState,
  refresh,
}: {
  projectId: string;
  agentRequest: string;
  setAgentRequest: (value: string) => void;
  agentState: string;
  setAgentState: (value: string) => void;
  refresh: () => Promise<void>;
}) {
  const [agentModel, setAgentModel] = useState("");
  const [agentProvider, setAgentProvider] = useState("");
  const [modelMessage, setModelMessage] = useState("");
  const [agentBusy, setAgentBusy] = useState(false);
  const [pendingRunId, setPendingRunId] = useState<string | null>(null);
  const [pendingPlan, setPendingPlan] = useState<{ objective: string; operations: Array<{ type: string; reason?: string }> } | null>(null);
  const [conversation, setConversation] = useState<Array<{ role: "user" | "director"; text: string; error?: boolean }>>([]);

  useEffect(() => {
    void api.getSettings().then((settings) => {
      setAgentModel(settings.model);
      setAgentProvider(settings.provider);
    });
  }, []);

  const updateAgentModel = async (model: string) => {
    setAgentModel(model);
    setModelMessage("Guardando modelo…");
    try {
      const settings = await api.updateSettings({ model });
      setAgentModel(settings.model);
      setAgentProvider(settings.provider);
      setModelMessage("");
    } catch (error) {
      setModelMessage(error instanceof Error ? error.message : "No se pudo guardar el modelo");
    }
  };

  const modelOptions = Array.from(
    new Set([agentModel, ...(AGENT_MODELS[agentProvider] ?? [])].filter(Boolean)),
  );

  const cancelPending = () => {
    setPendingRunId(null);
    setPendingPlan(null);
  };

  const confirmPending = async () => {
    if (!pendingRunId) return;
    setAgentBusy(true);
    try {
      const r = await api.editorAgentConfirm(projectId, pendingRunId);
      setAgentState(
        r.report
          ? `${r.status}: ${r.report.changes} cambio(s). ${r.report.verification}`
          : r.status,
      );
      setConversation((items) => [...items, { role: "director", text: r.report ? `${r.status}: ${r.report.changes} cambio(s). ${r.report.verification}` : r.status }]);
      setPendingRunId(null);
      setPendingPlan(null);
      await refresh();
    } catch (e) {
      const msg = e instanceof Error ? e.message : "La confirmación falló";
      setAgentState(msg);
      setConversation((items) => [...items, { role: "director", text: msg, error: true }]);
      if (msg.includes("STALE_VERSION") || msg.includes("timeline cambió")) {
        setPendingRunId(null);
        setPendingPlan(null);
      }
    } finally {
      setAgentBusy(false);
    }
  };

  const send = async () => {
    if (!agentRequest.trim() || agentBusy) return;
    setAgentBusy(true);
    setAgentState("EXECUTING");
    setConversation((items) => [...items, { role: "user", text: agentRequest.trim() }]);
    try {
      const r = await api.editorAgent(projectId, {
        request: agentRequest,
        execute: true,
        idempotency_key: operationKey(),
      });
      if (r.status === "NEEDS_USER" && r.run_id && r.plan) {
        setPendingRunId(r.run_id);
        setPendingPlan(r.plan);
        setAgentState("El agente propone cambios que requieren tu aprobación.");
        setConversation((items) => [...items, { role: "director", text: "El agente propone cambios que requieren tu aprobación." }]);
      } else {
        const response = r.report ? `${r.status}: ${r.report.changes} cambio(s). ${r.report.verification}` : r.status;
        setAgentState(response);
        setConversation((items) => [...items, { role: "director", text: response }]);
        setAgentRequest("");
        await refresh();
      }
    } catch (e) {
      const response = e instanceof Error ? e.message : "El agente no pudo ejecutar el plan";
      setAgentState(response);
      setConversation((items) => [...items, { role: "director", text: response, error: true }]);
    } finally {
      setAgentBusy(false);
    }
  };

  return (
    <section className={`agent-panel ${agentState ? "has-conversation" : "agent-ready"}`}>
      <header>
        <span className="section-label">Agente IA</span>
        <div className="agent-header-actions">
          {agentProvider === 'nvidia' ? (
            <NvidiaModelField compact value={agentModel} onChange={(value) => void updateAgentModel(value)} label="Modelo global del Agente IA" />
          ) : agentModel && (
            <label className="agent-model">
              <Select
                value={agentModel}
                options={modelOptions.map((model) => ({ value: model, label: model }))}
                onChange={(value) => void updateAgentModel(value)}
                ariaLabel="Modelo global del Agente IA"
                className="agent-model-select"
              />
            </label>
          )}
        </div>
      </header>
      <div className="agent-conversation" aria-label="Conversación del Agente IA">
        {conversation.map((entry, index) => <div key={index} className={`agent-message ${entry.role}${entry.error ? " agent-message-error" : ""}`} role={entry.error ? "alert" : undefined}><b>{entry.role === "user" ? "TÚ" : "SNIPRA / DIRECTOR"}</b><p>{entry.text}</p></div>)}
        {agentBusy && <div className="agent-message director" role="status"><b>SNIPRA / DIRECTOR</b><p>{agentState}</p></div>}
      </div>
      {pendingPlan && (
        <div className="agent-confirmation">
          <div className="agent-confirmation-summary">
            <strong>Objetivo:</strong> {pendingPlan.objective}
            <ul>
              {pendingPlan.operations.map((op, idx) => (
                <li key={idx}>
                  <code>{op.type}</code> {op.reason ? `— ${op.reason}` : null}
                </li>
              ))}
            </ul>
          </div>
          <div className="agent-confirmation-actions">
            <button className="btn-grad" onClick={confirmPending} disabled={agentBusy}>
              {agentBusy ? "Aplicando…" : "Confirmar cambios"}
            </button>
            <button className="btn-ghost" onClick={cancelPending} disabled={agentBusy}>
              Cancelar
            </button>
          </div>
        </div>
      )}
      <div className="agent-composer">
        <textarea
          value={agentRequest}
          onChange={(e) => setAgentRequest(e.target.value)}
          placeholder="Describe el edit…"
          rows={3}
          disabled={!!pendingPlan}
        />
        <button className="btn-grad" onClick={() => void send()} disabled={agentBusy || !!pendingPlan}>
          {agentBusy ? "Aplicando…" : "Aplicar plan"}
        </button>
        {modelMessage && <p className="agent-model-message">{modelMessage}</p>}
      </div>
    </section>
  );
}

function PropertiesInspector({
  selectedClip,
  selectedItem,
  selected,
  setSelected,
  apply,
  cropDraft,
  setCropDraft,
  validCrop,
  playhead,
  editContent,
  allContent,
  message,
  project,
}: InspectorProps) {
  return (
    <section className="inspector">
      <header>
        <span className="section-label">Propiedades</span>
        {selectedClip && (
          <span className="text-xs text-ink-mute">Clip seleccionado</span>
        )}
      </header>
      <div className="inspector-scroll">
        {selectedClip ? (
          <>
            <InspectorSection title="Transformación">
              <div className="inspector-grid">
                <Input
                  label="Posición X"
                  value={selectedClip.transform?.x ?? 0}
                  min={-1}
                  max={1}
                  step={0.05}
                  onChange={(value) =>
                    void apply("set_position", {
                      clip_id: selectedClip.id,
                      transform: { x: value },
                    })
                  }
                />
                <Input
                  label="Posición Y"
                  value={selectedClip.transform?.y ?? 0}
                  min={-1}
                  max={1}
                  step={0.05}
                  onChange={(value) =>
                    void apply("set_position", {
                      clip_id: selectedClip.id,
                      transform: { y: value },
                    })
                  }
                />
                <Input
                  label="Escala"
                  value={selectedClip.transform?.scale ?? 1}
                  min={0.1}
                  max={8}
                  step={0.05}
                  onChange={(value) =>
                    void apply("set_scale", {
                      clip_id: selectedClip.id,
                      scale: value,
                    })
                  }
                />
              </div>
              <div className="crop-grid">
                {(["x", "y", "width", "height"] as const).map((key) => (
                  <Input
                    key={key}
                    label={`Recorte ${key}`}
                    value={cropDraft[key]}
                    min={key === "width" || key === "height" ? 0.01 : 0}
                    max={1}
                    step={0.05}
                    onChange={(value) =>
                      setCropDraft({ ...cropDraft, [key]: value })
                    }
                  />
                ))}
              </div>
              <button
                disabled={!validCrop}
                className="inspector-action"
                onClick={() =>
                  void apply("clip_properties", {
                    clip_id: selectedClip.id,
                    properties: { transform: { crop: cropDraft } },
                  })
                }
              >
                Aplicar recorte
              </button>
            </InspectorSection>
            {selectedClip && (() => {
              const enhancedAssets = project.assets.filter(
                (a) => a.metadata?.asset_provenance?.origin === "ENHANCED" &&
                  a.metadata?.asset_provenance?.parent_asset_id === selectedClip.asset_id
              );
              if (enhancedAssets.length > 0) {
                return (
                  <InspectorSection title="Versión mejorada disponible">
                    <p className="text-xs text-ink-mute mb-2">
                      Se encontró una versión mejorada de este asset. Puedes reemplazar el clip actual.
                    </p>
                    {enhancedAssets.map((ea) => (
                      <button
                        key={ea.id}
                        className="inspector-action btn-grad"
                        onClick={() => void apply("replace_asset", {
                          clip_id: selectedClip.id,
                          asset_id: ea.id,
                        })}
                      >
                        Usar en clip seleccionado ({ea.filename})
                      </button>
                    ))}
                  </InspectorSection>
                );
              }
              return null;
            })()}
            <InspectorSection title="Audio">
              <div className="inspector-grid">
                <Input
                  label="Volumen"
                  value={selectedClip.audio?.volume ?? 1}
                  min={0}
                  max={4}
                  step={0.1}
                  onChange={(value) =>
                    void apply("set_volume", {
                      clip_id: selectedClip.id,
                      volume: value,
                    })
                  }
                />
                <Input
                  label="Entrada gradual"
                  suffix="s"
                  value={(selectedClip.audio?.fade_in_us ?? 0) / 1e6}
                  min={0}
                  step={0.1}
                  onChange={(value) =>
                    void apply("set_fade", {
                      clip_id: selectedClip.id,
                      audio: { fade_in_us: Math.round(value * 1e6) },
                    })
                  }
                />
                <Input
                  label="Salida gradual"
                  suffix="s"
                  value={(selectedClip.audio?.fade_out_us ?? 0) / 1e6}
                  min={0}
                  step={0.1}
                  onChange={(value) =>
                    void apply("set_fade", {
                      clip_id: selectedClip.id,
                      audio: { fade_out_us: Math.round(value * 1e6) },
                    })
                  }
                />
              </div>
              <label className="check-field">
                <input
                  type="checkbox"
                  checked={selectedClip.audio?.muted ?? false}
                  onChange={(e) =>
                    void apply("set_mute", {
                      clip_id: selectedClip.id,
                      muted: e.target.checked,
                    })
                  }
                />{" "}
                Silenciar
              </label>
            </InspectorSection>
            <InspectorSection title="Velocidad">
              <Input
                label="Velocidad de reproducción"
                suffix="×"
                value={selectedClip.playback_rate}
                min={0.25}
                max={4}
                step={0.25}
                onChange={(value) =>
                  void apply("change_speed", {
                    clip_id: selectedClip.id,
                    playback_rate: value,
                  })
                }
              />
            </InspectorSection>
          </>
        ) : (
          <p className="agent-empty">
            Selecciona un clip para editar sus propiedades.
          </p>
        )}
        <InspectorSection title="Texto y subtítulos">
          <div className="content-add-actions">
            <button
              className="inspector-action"
              onClick={() => setSelected("new-caption")}
            >
              Añadir subtítulo
            </button>
            <button
              className="inspector-action"
              onClick={() => setSelected("new-text")}
            >
              Añadir texto
            </button>
          </div>
          {(selectedItem ||
            selected === "new-caption" ||
            selected === "new-text") && (
            <form
              className="content-form"
              onSubmit={(e) => {
                e.preventDefault();
                editContent(
                  selectedItem?.kind === "text" || selected === "new-text"
                    ? "text"
                    : "caption",
                  e.currentTarget,
                );
              }}
            >
              <label className="content-text-field">Texto
                <input
                  className={field}
                  name="text"
                  required
                  defaultValue={selectedItem?.text}
                  placeholder="Escribe el texto"
                />
              </label>
              <div className="inspector-grid">
                <InputForm
                  label="Inicio"
                  name="start"
                  value={(selectedItem?.timeline_start_us ?? playhead) / 1e6}
                />
                <InputForm
                  label="Fin"
                  name="end"
                  value={
                    (selectedItem?.timeline_end_us ?? playhead + 2_000_000) /
                    1e6
                  }
                />
                <InputForm
                  label="X"
                  name="x"
                  value={selectedItem?.position?.x ?? 0}
                />
                <InputForm
                  label="Y"
                  name="y"
                  value={selectedItem?.position?.y ?? 0.75}
                />
                <InputForm
                  label="Tamaño"
                  name="font_size"
                  value={selectedItem?.style?.font_size ?? 42}
                />
                <label>
                  Color
                  <input
                    className={field}
                    name="color"
                    required
                    defaultValue={selectedItem?.style?.color ?? "#ffffff"}
                  />
                </label>
              </div>
              <div className="content-form-actions">
                <button className="btn-grad" type="submit">Guardar</button>
              {selectedItem && (
                <button
                  type="button"
                  className="text-xs text-bad"
                  onClick={() =>
                    void apply("delete", { clip_id: selectedItem.id })
                  }
                >
                  Eliminar
                </button>
              )}
              </div>
            </form>
          )}
        </InspectorSection>
        {allContent.map((item) => (
          <button
            key={item.id}
            className="content-item"
            onClick={() => setSelected(item.id)}
          >
            {item.kind}: {item.text}
          </button>
        ))}
      </div>
      {message && <p className="inspector-status">{message}</p>}
    </section>
  );
}

function InspectorSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="inspector-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}
function Input({
  label,
  value,
  onChange,
  suffix,
  ...rest
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  suffix?: string;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label>
      {label}
      <span className="input-suffix">
        <input
          className={field}
          type="number"
          value={value}
          onChange={(e) => onChange(number(e.target.value))}
          {...rest}
        />
        {suffix}
      </span>
    </label>
  );
}
function InputForm({
  label,
  name,
  value,
}: {
  label: string;
  name: string;
  value: number;
}) {
  return (
    <label>
      {label}
      <input
        className={field}
        name={name}
        required
        type="number"
        step=".05"
        defaultValue={value}
      />
    </label>
  );
}

function TimelinePanel({
  project,
  selected,
  setSelected,
  playhead,
  seek,
  apply,
  undo,
  togglePlayback,
}: {
  project: EditorProject;
  selected: string | null;
  setSelected: (value: string) => void;
  playhead: number;
  seek: (value: number) => void;
  apply: (type: string, params: Record<string, unknown>) => Promise<void>;
  undo: () => Promise<void>;
  togglePlayback: () => Promise<void>;
}) {
  const scroll = useRef<HTMLDivElement>(null);
  const playheadLine = useRef<HTMLDivElement>(null);
  const scrub = useRef<{ active: boolean; frame: number | null; value: number }>({ active: false, frame: null, value: 0 });
  const [viewportWidth, setViewportWidth] = useState(900);
  const [zoom, setZoom] = useState(1);
  const [ripple, setRipple] = useState(false);
  const [tool, setTool] = useState<"select" | "blade" | "trim">("select");
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; clipId: string; trackId: string } | null>(null);
  const dragReadout = useRef<HTMLDivElement>(null);
  const duration = Math.max(project.timeline.duration_us, 10_000_000);
  const seconds = duration / 1e6;
  const headerWidth = 158;
  const fitScale = Math.min(
    80,
    Math.max(0.06, (viewportWidth - headerWidth) / seconds),
  );
  const scale = Math.min(180, Math.max(0.06, fitScale * zoom));
  const width = Math.max(viewportWidth, headerWidth + seconds * scale);
  const majorTickSeconds =
    [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600].find(
      (value) => value * scale >= 112,
    ) ?? 3600;
  const minorTickSeconds = Math.max(
    1,
    majorTickSeconds / (majorTickSeconds >= 60 ? 3 : 5),
  );
  const ruler = Array.from(
    { length: Math.ceil(seconds / minorTickSeconds) + 1 },
    (_, index) => Math.min(duration, Math.round(index * minorTickSeconds * 1e6)),
  ).filter((value, index, values) => index === 0 || value !== values[index - 1]);
  const assets = useMemo(
    () => new Map(project.assets.map((asset) => [asset.id, asset])),
    [project.assets],
  );
  const selectedTrack = project.timeline.tracks.find((track) =>
    track.clips.some((clip) => clip.id === selected),
  );
  const selectedClip = selectedTrack?.clips.find((clip) => clip.id === selected);
  const selectedEnd = selectedClip
    ? clipTimelineEnd(selectedClip)
    : 0;
  useEffect(() => {
    if (!scroll.current) return;
    const update = () => setViewportWidth(scroll.current?.clientWidth ?? 900);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(scroll.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!scrub.current.active && playheadLine.current)
      playheadLine.current.style.left = `${headerWidth + (playhead / 1e6) * scale}px`;
  }, [playhead, scale]);
  useEffect(() => {
    const shortcuts = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.matches("input, textarea, select, [contenteditable='true']")) return;
      if (
        (event.ctrlKey || event.metaKey) &&
        !event.shiftKey &&
        event.key.toLowerCase() === "z"
      ) {
        event.preventDefault();
        void undo();
      }
      if (event.key === " ") {
        event.preventDefault();
        void togglePlayback();
      }
    };
    window.addEventListener("keydown", shortcuts);
    return () => window.removeEventListener("keydown", shortcuts);
  }, [togglePlayback, undo]);
  const positionToTime = (clientX: number, canvas: HTMLDivElement) => {
    const bounds = canvas.getBoundingClientRect();
    return Math.max(0, Math.min(duration, ((clientX - bounds.left - headerWidth) / scale) * 1e6));
  };
  const updateScrub = (clientX: number, canvas: HTMLDivElement) => {
    const value = positionToTime(clientX, canvas);
    scrub.current.value = value;
    if (playheadLine.current) playheadLine.current.style.left = `${headerWidth + (value / 1e6) * scale}px`;
    if (scrub.current.frame === null) {
      scrub.current.frame = requestAnimationFrame(() => {
        scrub.current.frame = null;
        seek(scrub.current.value);
      });
    }
  };
  const trackName = (type: string, index: number) =>
    type === "video"
      ? `V${index + 1}`
      : type === "audio"
        ? `A${index + 1}`
        : type === "text" || type === "caption"
          ? "CC"
          : "TXT";
  const contextTrack = contextMenu
    ? project.timeline.tracks.find((track) => track.id === contextMenu.trackId)
    : undefined;
  const contextClip = contextTrack?.clips.find(
    (clip) => clip.id === contextMenu?.clipId,
  );
  const contextClipEnd = contextClip
    ? clipTimelineEnd(contextClip)
    : 0;
  const contextCanSplit = Boolean(
    contextClip &&
      !contextTrack?.locked &&
      !contextClip.kind &&
      contextClip.timeline_start_us < playhead &&
      playhead < contextClipEnd,
  );
  const showDragFeedback = useCallback((value: string) => {
    const node = dragReadout.current;
    if (!node) return;
    node.hidden = !value;
    node.textContent = value;
  }, []);
  return (
    <section className="timeline-panel">
      <header>
        <div>
          <span className="section-label">Timeline</span>
          <span className="ml-3 text-xs text-ink-mute">{us(duration)}</span>
        </div>
        <div className="timeline-tools">
          <div className="timeline-tool-group" aria-label="Edición">
            <span className="tool-group-label">EDIT</span>
            <button className={`timeline-mode ${tool === "select" ? "active" : ""}`} title="Seleccionar y mover" aria-label="Seleccionar y mover" aria-pressed={tool === "select"} onClick={() => setTool("select")}><EditorIcon name="select" /></button>
            <button className={`timeline-mode ${tool === "blade" ? "active" : ""}`} title="Cortar clip" aria-label="Cortar clip" aria-pressed={tool === "blade"} onClick={() => setTool("blade")}><EditorIcon name="cut" /></button>
            <button className={`timeline-mode ${tool === "trim" ? "active" : ""}`} title="Recortar bordes" aria-label="Recortar bordes" aria-pressed={tool === "trim"} onClick={() => setTool("trim")}><EditorIcon name="trim" /></button>
            <button title="Dividir clip en el cabezal" aria-label="Dividir clip en el cabezal" disabled={!selectedClip || !!selectedClip.kind || !!selectedTrack?.locked || playhead <= selectedClip.timeline_start_us || playhead >= selectedEnd} onClick={() => selectedClip && void apply("split", { clip_id: selectedClip.id, at_us: playhead })}><EditorIcon name="split" /></button>
          </div>
          <div className="timeline-tool-group" aria-label="Estructura">
            <span className="tool-group-label">STRUCTURE</span>
          <button
            title="Eliminar clip (Delete)"
            aria-label="Eliminar clip"
            disabled={!selected || selectedTrack?.locked}
            onClick={() =>
              selected && void apply("delete", { clip_id: selected, ripple })
            }
          >
            <EditorIcon name="delete" />
          </button>
          <button
            className="timeline-duplicate"
            title="Duplicar clip seleccionado"
            aria-label="Duplicar clip seleccionado"
            disabled={!selected || selectedTrack?.locked}
            onClick={() =>
              selected &&
              selectedTrack &&
              void apply("duplicate", {
                clip_id: selected,
                track_id: selectedTrack.id,
                timeline_start_us: selectedEnd,
              })
            }
          >
            <EditorIcon name="duplicate" />
          </button>
          <button
            className={`timeline-ripple ${ripple ? "active" : ""}`}
            title={ripple ? "Ripple activo al eliminar" : "Activar ripple al eliminar"}
            aria-label={ripple ? "Desactivar ripple" : "Activar ripple"}
            onClick={() => setRipple((value) => !value)}
          >
            <EditorIcon name="ripple" />
          </button>
          <button className={selectedTrack?.locked ? "active" : ""} title={selectedTrack ? (selectedTrack.locked ? "Desbloquear pista seleccionada" : "Bloquear pista seleccionada") : "Selecciona un clip para bloquear su pista"} aria-label="Bloquear pista seleccionada" disabled={!selectedTrack} onClick={() => selectedTrack && void apply("track_lock", { track_id: selectedTrack.id, locked: !selectedTrack.locked })}>
            <EditorIcon name={selectedTrack?.locked ? "unlock" : "lock"} />
          </button>
          <button title="Deshacer (Ctrl+Z)" aria-label="Deshacer" onClick={() => void undo()}>
            <EditorIcon name="undo" />
          </button>
          </div>
          <div className="timeline-tool-group timeline-view-tools" aria-label="Vista de timeline">
          <span className="tool-group-label">VIEW</span>
          <button
            title="Alejar timeline"
            aria-label="Alejar timeline"
            disabled={zoom <= 0.25}
            onClick={() => setZoom((value) => Math.max(0.25, value / 1.35))}
          >
            <EditorIcon name="zoomOut" />
          </button>
          <button
            className="timeline-zoom-readout"
            title="Ajustar timeline completa"
            aria-label="Ajustar timeline completa"
            onClick={() => setZoom(1)}
          >
            <EditorIcon name="zoom" />
          </button>
          <button
            title="Acercar timeline"
            aria-label="Acercar timeline"
            disabled={zoom >= 24}
            onClick={() => setZoom((value) => Math.min(24, value * 1.35))}
          >
            <EditorIcon name="zoomIn" />
          </button>
          <button
            className="timeline-fit"
            title="Ajustar timeline completa"
            aria-label="Ajustar timeline completa"
            onClick={() => setZoom(1)}
          >
            <EditorIcon name="fit" />
          </button>
          </div>
        </div>
      </header>
      <div
        className="timeline-scroll"
        ref={scroll}
        onWheel={(event) => {
          if (!event.shiftKey || !event.deltaY) return;
          event.preventDefault();
          event.currentTarget.scrollLeft += event.deltaY;
        }}
      >
        <div
          className="timeline-canvas"
          style={{ width }}
          onPointerDown={(event) => {
            if ((event.target as HTMLElement).closest(".track-header")) return;
            setContextMenu(null);
            event.currentTarget.setPointerCapture(event.pointerId);
            scrub.current.active = true;
            updateScrub(event.clientX, event.currentTarget);
          }}
          onPointerMove={(event) => {
            if (scrub.current.active) updateScrub(event.clientX, event.currentTarget);
          }}
          onPointerUp={(event) => {
            if (!scrub.current.active) return;
            updateScrub(event.clientX, event.currentTarget);
            scrub.current.active = false;
            if (scrub.current.frame !== null) cancelAnimationFrame(scrub.current.frame);
            scrub.current.frame = null;
            seek(scrub.current.value);
          }}
        >
          <div className="timeline-ruler">
            <span className="timeline-ruler-gutter" />
            {ruler.map((value) => (
              <span
                className={
                  value % Math.round(majorTickSeconds * 1e6) === 0
                    ? "timeline-major-tick"
                    : "timeline-minor-tick"
                }
                key={value}
                style={{ left: headerWidth + (value / 1e6) * scale }}
              >
                {value % Math.round(majorTickSeconds * 1e6) === 0
                  ? timelineTime(value)
                  : ""}
              </span>
            ))}
          </div>
          {project.timeline.tracks.map((track, index) => {
            const audioFromVideo =
              track.type === "audio" && track.clips.length === 0
                ? project.timeline.tracks
                    .filter((item) => item.type === "video")
                    .flatMap((item) => item.clips)
                    .filter((clip) => assets.get(clip.asset_id)?.has_audio)
                : track.clips;
            return (
              <div className={`track-row ${track.type}`} key={track.id}>
                <div className="track-header">
                  <strong>
                    {trackName(
                      track.type,
                      project.timeline.tracks
                        .slice(0, index)
                        .filter((item) => item.type === track.type).length,
                    )}
                  </strong>
                  <span className="track-kind">
                    {track.type === "video"
                      ? "Video"
                      : track.type === "audio"
                        ? "Audio"
                        : track.type === "text" || track.type === "caption"
                          ? "Subt."
                          : "Texto"}
                  </span>
                  <button type="button" className="track-visibility" title="Visibilidad de pista no disponible en este proyecto" aria-label="Visibilidad de pista no disponible" disabled><EditorIcon name="visible" /></button>
                  {track.type === "audio" ? (
                    <button
                      className={`track-mute ${track.muted ? "is-muted" : ""}`}
                      title={track.muted ? "Activar audio de pista" : "Silenciar pista de audio"}
                      aria-label={track.muted ? "Activar audio de pista" : "Silenciar pista de audio"}
                      aria-pressed={track.muted}
                      onClick={(event) => {
                        event.stopPropagation();
                        void apply("track_properties", {
                          track_id: track.id,
                          properties: { muted: !track.muted },
                        });
                      }}
                    ><EditorIcon name={track.muted ? "mute" : "sound"} /></button>
                  ) : <button type="button" title="Mute solo disponible para pistas de audio" aria-label="Mute solo disponible para pistas de audio" disabled><EditorIcon name="sound" /></button>}
                  <button
                    className={track.locked ? "is-locked" : ""}
                    title={track.locked ? "Desbloquear pista" : "Bloquear pista"}
                    aria-label={track.locked ? "Desbloquear pista" : "Bloquear pista"}
                    onClick={(event) => {
                      event.stopPropagation();
                      void apply("track_lock", {
                        track_id: track.id,
                        locked: !track.locked,
                      });
                    }}
                  >
                    <EditorIcon name={track.locked ? "lock" : "unlock"} />
                  </button>
                </div>
                <div className="track-lane">
                  {audioFromVideo.map((clip) => (
                    <TimelineClip
                      asset={assets.get(clip.asset_id)}
                      clip={clip}
                      key={`${track.id}-${clip.id}`}
                      locked={track.locked}
                      projectId={project.id}
                      scale={scale}
                      selected={selected === clip.id}
                      trackId={track.id}
                      waveform={track.type === "audio"}
                      interactive={track.type !== "audio" || track.clips.includes(clip)}
                      tool={tool}
                      onSelect={setSelected}
                      onOperation={apply}
                      onFeedback={showDragFeedback}
                      onBlade={(atUs) => {
                        setSelected(clip.id);
                        void apply("split", { clip_id: clip.id, at_us: atUs });
                      }}
                      onContext={(event) =>
                        setContextMenu({
                          x: event.clientX,
                          y: event.clientY,
                          clipId: clip.id,
                          trackId: track.id,
                        })
                      }
                    />
                  ))}
                </div>
              </div>
            );
          })}
          <div
            ref={playheadLine}
            className="playhead"
            style={{ left: headerWidth + (playhead / 1e6) * scale }}
          />
          <div ref={dragReadout} className="timeline-drag-readout" aria-live="polite" hidden />
        </div>
      </div>
      {contextMenu && (
        <div className="timeline-context-menu" style={{ left: contextMenu.x, top: contextMenu.y }}>
          <button disabled={!contextCanSplit} onClick={() => { if (contextClip) { setSelected(contextClip.id); void apply("split", { clip_id: contextClip.id, at_us: playhead }); } setContextMenu(null); }}>Dividir en playhead</button>
          <button disabled={!contextClip || contextTrack?.locked} onClick={() => { if (contextClip && contextTrack) void apply("duplicate", { clip_id: contextClip.id, track_id: contextTrack.id, timeline_start_us: contextClipEnd }); setContextMenu(null); }}>Duplicar</button>
          <button disabled={!contextClip || contextTrack?.locked} onClick={() => { if (contextClip) void apply("delete", { clip_id: contextClip.id, ripple }); setContextMenu(null); }}>Eliminar clip</button>
          <button disabled={!contextTrack} onClick={() => { if (contextTrack) void apply("track_lock", { track_id: contextTrack.id, locked: !contextTrack.locked }); setContextMenu(null); }}>{contextTrack?.locked ? "Desbloquear pista" : "Bloquear pista"}</button>
        </div>
      )}
    </section>
  );
}

const TimelineClip = memo(function TimelineClip({
  asset,
  clip,
  locked,
  projectId,
  scale,
  selected,
  trackId,
  waveform,
  interactive,
  tool,
  onSelect,
  onOperation,
  onFeedback,
  onBlade,
  onContext,
}: {
  asset: EditorProject["assets"][number] | undefined;
  clip: EditorTimelineClip;
  locked: boolean;
  projectId: string;
  scale: number;
  selected: boolean;
  trackId: string;
  waveform: boolean;
  interactive: boolean;
  tool: "select" | "blade" | "trim";
  onSelect: (id: string) => void;
  onOperation: (type: string, params: Record<string, unknown>) => Promise<void>;
  onFeedback: (value: string) => void;
  onBlade: (atUs: number) => void;
  onContext: (event: React.MouseEvent<HTMLButtonElement>) => void;
}) {
  const drag = useRef<{
    kind: "move" | "trim-in" | "trim-out";
    startX: number;
    timelineStartUs: number;
    sourceInUs: number;
    sourceOutUs: number;
    preview?: { timelineStartUs: number; sourceInUs: number; sourceOutUs: number };
  } | null>(null);
  const [preview, setPreview] = useState<{
    timelineStartUs: number;
    sourceInUs: number;
    sourceOutUs: number;
  } | null>(null);
  const [bladeX, setBladeX] = useState<number | null>(null);
  const isMediaClip = !clip.kind;
  const sourceIn = preview?.sourceInUs ?? clip.source_in_us ?? 0;
  const sourceOut =
    preview?.sourceOutUs ??
    clip.source_out_us ??
    (clip.timeline_end_us ?? clip.timeline_start_us) - clip.timeline_start_us;
  const timelineStart = preview?.timelineStartUs ?? clip.timeline_start_us;
  const previewEnd = timelineStart + (sourceOut - sourceIn) / clip.playback_rate;
  const width = Math.max(28, ((previewEnd - timelineStart) / 1e6) * scale);
  const thumbnails =
    width < 56 ? 0 : Math.min(4, Math.max(1, Math.floor(width / 150)));
  const clipLabel = clip.text || (asset ? assetLabel(asset.filename) : "Contenido");
  const mediaUrl = (path: "thumbnail" | "waveform", atUs?: number) =>
    `/api/editor/projects/${projectId}/assets/${clip.asset_id}/${path}${atUs === undefined ? "" : `?at_us=${Math.round(atUs)}`}`;
  const startDrag = (
    event: React.PointerEvent<HTMLElement>,
    kind: "move" | "trim-in" | "trim-out",
  ) => {
    if (locked || !interactive || (!isMediaClip && kind !== "move") || (kind === "move" && tool !== "select")) return;
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = {
      kind,
      startX: event.clientX,
      timelineStartUs: clip.timeline_start_us,
      sourceInUs: sourceIn,
      sourceOutUs: sourceOut,
    };
    onSelect(clip.id);
  };
  const updateDrag = (event: React.PointerEvent<HTMLElement>) => {
    const current = drag.current;
    if (!current) return;
    const deltaTimelineUs = Math.round(((event.clientX - current.startX) / scale) * 1e6);
    const deltaSourceUs = Math.round(deltaTimelineUs * clip.playback_rate);
    const minimum = 100_000;
    const next = {
      timelineStartUs: current.timelineStartUs,
      sourceInUs: current.sourceInUs,
      sourceOutUs: current.sourceOutUs,
    };
    if (current.kind === "move") next.timelineStartUs = Math.max(0, current.timelineStartUs + deltaTimelineUs);
    if (current.kind === "trim-in") {
      next.sourceInUs = Math.max(0, Math.min(current.sourceOutUs - minimum, current.sourceInUs + deltaSourceUs));
      next.timelineStartUs = current.timelineStartUs + Math.round((next.sourceInUs - current.sourceInUs) / clip.playback_rate);
    }
    if (current.kind === "trim-out") {
      const assetEnd = asset?.duration_us ?? Number.MAX_SAFE_INTEGER;
      next.sourceOutUs = Math.min(
        assetEnd,
        Math.max(current.sourceInUs + minimum, current.sourceOutUs + deltaSourceUs),
      );
    }
    current.preview = next;
    setPreview(next);
    onFeedback(
      current.kind === "move"
        ? `Inicio ${preciseTimelineTime(next.timelineStartUs)}`
        : `In ${preciseTimelineTime(next.sourceInUs)} · Out ${preciseTimelineTime(next.sourceOutUs)} · Dur ${preciseTimelineTime((next.sourceOutUs - next.sourceInUs) / clip.playback_rate)}`,
    );
  };
  const finishDrag = () => {
    const current = drag.current;
    drag.current = null;
    setPreview(null);
    onFeedback("");
    if (!current?.preview) return;
    if (current.kind === "move")
      void onOperation("move", { clip_id: clip.id, track_id: trackId, timeline_start_us: current.preview.timelineStartUs });
    else
      void onOperation("trim", { clip_id: clip.id, source_in_us: current.preview.sourceInUs, source_out_us: current.preview.sourceOutUs });
  };
  return (
    <button
      className={`clip-block ${waveform ? "audio-clip" : clip.kind === "caption" ? "caption-clip" : clip.kind === "text" ? "text-clip" : "video-clip"} ${selected ? "selected" : ""} ${locked ? "locked" : ""} ${interactive ? "" : "derived-clip"} ${preview ? "is-dragging" : ""} ${tool === "blade" ? "blade-active" : ""} ${tool === "trim" ? "trim-active" : ""}`}
      onClick={(event) => {
        event.stopPropagation();
        if (!interactive) return;
        if (tool === "blade" && isMediaClip && !locked) {
          const bounds = event.currentTarget.getBoundingClientRect();
          const atUs = clip.timeline_start_us + Math.round(((event.clientX - bounds.left) / scale) * 1e6);
          if (atUs > clip.timeline_start_us && atUs < clipTimelineEnd(clip)) onBlade(atUs);
          return;
        }
        onSelect(clip.id);
      }}
      onContextMenu={(event) => {
        event.preventDefault();
        event.stopPropagation();
        if (!interactive) return;
        onSelect(clip.id);
        onContext(event);
      }}
      onPointerDown={(event) => {
        if (!interactive) return;
        if (tool === "blade") event.stopPropagation();
        else startDrag(event, "move");
      }}
      onPointerMove={(event) => {
        if (tool === "blade") {
          const bounds = event.currentTarget.getBoundingClientRect();
          setBladeX(Math.max(0, Math.min(bounds.width, event.clientX - bounds.left)));
          return;
        }
        updateDrag(event);
      }}
      onPointerLeave={() => setBladeX(null)}
      onPointerUp={finishDrag}
      style={{ left: (timelineStart / 1e6) * scale, width }}
      title={clipLabel}
    >
      {tool === "blade" && bladeX !== null && <span className="blade-cut-guide" style={{ left: bladeX }} aria-hidden="true" />}
      {waveform && asset?.has_audio && (
        <img
          className="clip-waveform"
          src={mediaUrl("waveform")}
          alt="Forma de onda del audio"
          loading="lazy"
        />
      )}
      {!waveform && !clip.kind && asset && thumbnails > 0 && (
        <span className="clip-thumbnails" aria-hidden="true">
          {Array.from({ length: thumbnails }, (_, index) => {
            const atUs =
              clip.source_in_us +
              ((index + 0.5) / thumbnails) *
                (clip.source_out_us - clip.source_in_us);
            return (
              <img
                key={index}
                src={mediaUrl("thumbnail", atUs)}
                alt=""
                loading="lazy"
              />
            );
          })}
        </span>
      )}
      {interactive && isMediaClip && <span
          className="clip-handle clip-handle-in"
          aria-label="Trim inicio"
          onPointerDown={(event) => startDrag(event, "trim-in")}
        />}
      <b>{clipLabel}</b>
      {interactive && isMediaClip && <span
          className="clip-handle clip-handle-out"
          aria-label="Trim final"
          onPointerDown={(event) => startDrag(event, "trim-out")}
        />}
    </button>
  );
});
