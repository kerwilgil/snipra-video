import type { SVGProps } from "react";

type Name = "select" | "cut" | "trim" | "split" | "delete" | "duplicate" | "ripple" | "lock" | "unlock" | "undo" | "zoomOut" | "zoom" | "zoomIn" | "fit" | "visible" | "hidden" | "mute" | "sound" | "play" | "pause" | "start" | "end";

export function EditorIcon({ name, ...props }: SVGProps<SVGSVGElement> & { name: Name }) {
  const shapes: Record<Name, React.ReactNode> = {
    select: <><path d="m5 3 13 9-6 1-3 7-4-17Z" /></>,
    cut: <><circle cx="6" cy="6" r="2" /><circle cx="6" cy="18" r="2" /><path d="M8 8 20 20M8 16 20 4" /></>,
    trim: <><path d="M4 5v14m16-14v14M7 12h10m-7-3-3 3 3 3m4-6 3 3-3 3" /></>,
    split: <><path d="M12 3v18M4 6h5M4 18h5m6-12h5m-5 12h5" /></>,
    delete: <><path d="M4 7h16M9 7V4h6v3m3 0-1 13H7L6 7m4 4v6m4-6v6" /></>,
    duplicate: <><rect x="8" y="8" width="12" height="12" rx="1" /><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3" /></>,
    ripple: <><path d="M3 7h8l3 5-3 5H3m14-10 4 5-4 5M7 12h10" /></>,
    lock: <><rect x="5" y="10" width="14" height="11" rx="1" /><path d="M8 10V7a4 4 0 0 1 8 0v3" /></>,
    unlock: <><rect x="5" y="10" width="14" height="11" rx="1" /><path d="M8 10V7a4 4 0 0 1 8 0" /></>,
    undo: <><path d="M9 7 4 12l5 5M4 12h10a6 6 0 0 1 0 12" transform="translate(0 -3)" /></>,
    zoomOut: <><circle cx="10" cy="10" r="7" /><path d="M7 10h6m2 5 6 6" /></>,
    zoom: <><circle cx="10" cy="10" r="7" /><path d="M15 15 21 21" /></>,
    zoomIn: <><circle cx="10" cy="10" r="7" /><path d="M7 10h6m-3-3v6m5 2 6 6" /></>,
    fit: <><path d="M4 9V4h5m6 0h5v5m0 6v5h-5M9 20H4v-5M4 4l6 6m10-6-6 6m6 10-6-6M4 20l6-6" /></>,
    visible: <><path d="M2 12s4-6 10-6 10 6 10 6-4 6-10 6S2 12 2 12Z" /><circle cx="12" cy="12" r="2.5" /></>,
    hidden: <><path d="M3 3 21 21M9 6.4A12 12 0 0 1 12 6c6 0 10 6 10 6a17 17 0 0 1-3.2 3.3M6 8.2C3.5 9.8 2 12 2 12s4 6 10 6c1 0 2-.2 2.9-.5" /></>,
    mute: <><path d="M4 9h4l5-4v14l-5-4H4V9Zm13 0 4 6m0-6-4 6" /></>,
    sound: <><path d="M4 9h4l5-4v14l-5-4H4V9Zm13-1a6 6 0 0 1 0 8m2-11a10 10 0 0 1 0 14" /></>,
    play: <path d="m8 5 11 7-11 7V5Z" />,
    pause: <><path d="M8 5v14M16 5v14" /></>,
    start: <><path d="M5 5v14m14-14L9 12l10 7V5Z" /></>,
    end: <><path d="M19 5v14M5 5l10 7-10 7V5Z" /></>,
  };
  return <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{shapes[name]}</svg>;
}
