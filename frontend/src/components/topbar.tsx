/* eslint-disable react-refresh/only-export-components -- modulo de contexto del shell:
   el contexto y los portales viajan juntos; ninguno se hot-reloadea aislado. */
import { createContext, useContext, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

interface TopbarSlots {
  titleSlot: HTMLElement | null
  actionsSlot: HTMLElement | null
  searchSlot: HTMLElement | null
}

export const TopbarSlotsContext = createContext<TopbarSlots>({ titleSlot: null, actionsSlot: null, searchSlot: null })

/** Contexto/contextual title of the current page, rendered inside the shell topbar. */
export function TopbarTitle({ children }: { children: ReactNode }) {
  const { titleSlot } = useContext(TopbarSlotsContext)
  if (!titleSlot || children == null || children === false) return null
  return createPortal(children, titleSlot)
}

/** Existing page-level main actions, lifted into the shell topbar (never duplicated). */
export function TopbarActions({ children }: { children: ReactNode }) {
  const { actionsSlot } = useContext(TopbarSlotsContext)
  if (!actionsSlot || children == null || children === false) return null
  return createPortal(children, actionsSlot)
}

export function TopbarSearch({ children }: { children: ReactNode }) {
  const { searchSlot } = useContext(TopbarSlotsContext)
  if (!searchSlot || children == null || children === false) return null
  return createPortal(children, searchSlot)
}
