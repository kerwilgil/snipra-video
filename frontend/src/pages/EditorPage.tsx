import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { IconPlus } from "../components/icons";
import { TopbarActions, TopbarTitle } from "../components/topbar";
import { api } from "../lib/api";
import type { EditorProjectListItem } from "../lib/types";

export function EditorPage() {
  const [projects, setProjects] = useState<EditorProjectListItem[]>([]);
  const [name, setName] = useState("Nuevo proyecto");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const load = async () => {
    setLoading(true);
    setError("");
    try {
      setProjects(await api.listEditorProjects());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "No se pudieron cargar los proyectos");
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    // Initial project retrieval is an external async synchronization.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, []);
  const create = async () => {
    const tab = window.open("", "_blank");
    if (!tab) {
      setError("El navegador bloqueó la nueva pestaña. Permite popups para Snipra.");
      return;
    }
    try {
      const project = await api.createEditorProject(name.trim() || "Nuevo proyecto");
      tab.location.replace(
        new URL(`/editor/projects/${project.id}`, window.location.origin).href,
      );
      setName("Nuevo proyecto");
      await load();
    } catch (cause) {
      tab.close();
      setError(cause instanceof Error ? cause.message : "No se pudo crear el proyecto");
    }
  };
  const rename = async (project: EditorProjectListItem) => {
    const value = window.prompt("Renombrar proyecto", project.name)?.trim();
    if (!value) return;
    await api.renameEditorProject(project.id, value);
    await load();
  };
  const remove = async (project: EditorProjectListItem) => {
    if (!window.confirm(`Eliminar proyecto "${project.name}"?\n\nSe eliminarán la timeline, historial y configuración del proyecto. Los archivos de video originales no serán eliminados del disco.`)) return;
    await api.deleteEditorProject(project.id);
    await load();
  };
  return (
    <section className="page-container page-container--form section-stack">
      <TopbarTitle>
        <span className="topbar-context">Editor IA</span>
        <strong>Proyectos</strong>
      </TopbarTitle>
      <TopbarActions>
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          className="field w-40 px-3 py-2 text-xs"
          aria-label="Nombre del proyecto"
        />
        <button className="btn-primary" onClick={() => void create()}>
          <IconPlus className="h-3.5 w-3.5" /> Nuevo proyecto
        </button>
      </TopbarActions>
      <header className="page-header">
        <h1>Editor IA</h1>
        <p className="page-description">Abre un proyecto para trabajar en su timeline y medios.</p>
      </header>
      {error && <div role="alert" className="page-error"><span>No se pudieron cargar los proyectos: {error}</span><button className="btn-secondary" onClick={() => void load()}>Reintentar</button></div>}
      <section className="project-list">
        <header><span className="section-label">Proyectos recientes</span></header>
        {loading && <p className="page-description" role="status">Cargando proyectos…</p>}
        {!loading && !error && projects.length === 0 && (
          <p className="mt-3 text-sm text-ink-mute">Todavia no hay proyectos. Crea el primero para abrir el editor dedicado.</p>
        )}
        {projects.map((project) => (
          <article key={project.id}>
            <div><strong>{project.name}</strong><span>Actualizado {new Date(project.updated_at).toLocaleString()}</span>{project.unavailable_reason && <span className="text-bad">No disponible: {project.unavailable_reason}</span>}</div>
            <div>
              {project.unavailable_reason ? <span className="btn-secondary" aria-disabled="true">Abrir</span> : <Link className="btn-secondary" to={`/editor/projects/${project.id}`} target="_blank">Abrir</Link>}
              <button className="btn-secondary" onClick={() => void rename(project)}>Renombrar</button>
              <button className="btn-danger" onClick={() => void remove(project)}>Eliminar proyecto</button>
            </div>
          </article>
        ))}
      </section>
    </section>
  );
}
