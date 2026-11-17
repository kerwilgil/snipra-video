import type { ReactNode } from 'react'

/** Shared geometry for sibling configuration cards, independent of copy length. */
export function ConfigCard({ title, children, description, extra }: {
  title: string
  children: ReactNode
  description: ReactNode
  extra?: ReactNode
}) {
  return <div className="section-card config-card">
    <div className="config-card-title text-sm font-semibold text-ink">{title}</div>
    <div className="config-card-control">{children}</div>
    <p className="config-description text-xs leading-5 text-ink-soft">{description}</p>
    {extra}
  </div>
}
