import { useEffect } from 'react'

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = 'Confirmar',
  cancelLabel = 'Cancelar',
  destructive = true,
  onConfirm,
  onCancel,
}: {
  open: boolean
  title: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  destructive?: boolean
  onConfirm: () => void
  onCancel: () => void
}) {
  useEffect(() => {
    if (!open) return
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onCancel()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, onCancel])

  if (!open) return null

  return (
    <div
      role="presentation"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onCancel}
    >
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        aria-describedby="confirm-dialog-message"
        onClick={(e) => e.stopPropagation()}
        className="card w-full max-w-sm p-5 shadow-2xl shadow-black/60"
      >
        <h2 id="confirm-dialog-title" className="text-sm font-semibold text-ink">
          {title}
        </h2>
        <p
          id="confirm-dialog-message"
          className="mt-2 break-words text-sm leading-5 text-ink-soft"
        >
          {message}
        </p>
        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="cursor-pointer rounded-lg border border-border px-3 py-2 text-sm font-medium text-ink-soft transition-colors hover:border-border-strong hover:text-ink"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            autoFocus
            onClick={onConfirm}
            className={`cursor-pointer rounded-lg px-3 py-2 text-sm font-semibold transition-colors ${
              destructive
                ? 'border border-bad/40 bg-bad/10 text-bad hover:bg-bad/20'
                : 'btn-grad'
            }`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
