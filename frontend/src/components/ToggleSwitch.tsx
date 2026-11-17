function SwitchTrack({ checked }: { checked: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={`relative inline-block h-6 w-11 shrink-0 rounded-full transition-colors ${
        checked ? 'bg-cyan' : 'bg-white/15'
      }`}
    >
      <span
        className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all ${
          checked ? 'left-[22px]' : 'left-0.5'
        }`}
      />
    </span>
  )
}

/** Switch aislado sin texto propio — usar cuando el contexto ya da la etiqueta (ej. una fila). */
export function ToggleSwitch({
  checked,
  onChange,
  ariaLabel,
  title,
}: {
  checked: boolean
  onChange: (value: boolean) => void
  ariaLabel?: string
  title?: string
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={ariaLabel}
      title={title}
      onClick={() => onChange(!checked)}
      className="cursor-pointer"
    >
      <SwitchTrack checked={checked} />
    </button>
  )
}

/** Fila clickeable completa con label + switch — usar en formularios/ajustes. */
export function ToggleField({
  label,
  checked,
  onChange,
}: {
  label: string
  checked: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={`flex h-10 w-full cursor-pointer items-center justify-between gap-2 rounded-lg border px-3 text-sm transition-colors ${
        checked
          ? 'border-cyan/40 bg-cyan/[0.06] text-ink'
          : 'border-border bg-base/40 text-ink-soft hover:border-border-strong'
      }`}
    >
      <span className="min-w-0 truncate text-left">{label}</span>
      <SwitchTrack checked={checked} />
    </button>
  )
}
