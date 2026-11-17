import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { EditorWorkspace } from "../components/EditorWorkspace";
import { IconCheck, IconDownload } from "../components/icons";
import brandLight from "../../image/snipra-brand-v2/logos/snipra-primary-on-light.png";
import { api } from "../lib/api";
import type { EditorProject } from "../lib/types";

export function DedicatedEditorPage() {
  const { projectId } = useParams();
  const [project, setProject] = useState<EditorProject | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!projectId) return;
    void api
      .getEditorProject(projectId)
      .then(setProject)
      .catch((cause) => setError(cause instanceof Error ? cause.message : "No se pudo abrir el proyecto"));
  }, [projectId]);

  if (error) return <main className="dedicated-editor-error">{error}</main>;
  if (!project) return <main className="dedicated-editor-error">Abriendo proyecto…</main>;
  return (
    <main className="dedicated-editor-page">
      <header className="dedicated-editor-topbar">
        <span className="dedicated-brand-surface"><img className="dedicated-brand-mark" src={brandLight} alt="Snipra" title="Snipra" /></span>
        <Link to="/editor" className="dedicated-back">← Proyectos</Link>
        <span className="dedicated-project-name">{project.name}</span>
        <span className="dedicated-saved"><IconCheck /> Guardado · v{project.current_version}</span>
        <button className="top-export" disabled title="Exportar desde el flujo existente"><IconDownload /> Exportar</button>
      </header>
      <EditorWorkspace project={project} onProject={setProject} onAssetRemoved={async () => undefined} dedicated />
    </main>
  );
}
