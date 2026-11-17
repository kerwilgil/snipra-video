import { useEffect, useRef, useState } from 'react'

export interface SelectOption {
  value: string
  label: string
  swatch?: string
}

export function Select({
  value,
  options,
  onChange,
  placeholder,
  ariaLabel,
  className,
  disabled,
}: {
  value: string
  options: SelectOption[]
  onChange: (value: string) => void
  placeholder?: string
  ariaLabel?: string
  className?: string
  disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
  const [openUp, setOpenUp] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([])
  const current = options.find((o) => o.value === value)

  useEffect(() => {
    if (!open) return
    function onDoc(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setOpen(false)
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        setOpen(false)
        triggerRef.current?.focus()
      }
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  // al abrir, mueve el foco real a la opcion seleccionada (o la primera) para que
  // las flechas naveguen de inmediato — sin esto el listbox era inalcanzable por teclado
  useEffect(() => {
    if (!open) return
    const selectedIndex = options.findIndex((o) => o.value === value)
    optionRefs.current[selectedIndex >= 0 ? selectedIndex : 0]?.focus()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  function handleToggle() {
    if (disabled) return
    if (!open && triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect()
      const spaceBelow = window.innerHeight - rect.bottom
      const estimated = Math.min(options.length * 38 + 12, 288)
      setOpenUp(spaceBelow < estimated)
    }
    setOpen((o) => !o)
  }

  function handleTriggerKeyDown(e: React.KeyboardEvent<HTMLButtonElement>) {
    if (disabled) return
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault()
      if (!open) handleToggle()
    }
  }

  function handleOptionKeyDown(e: React.KeyboardEvent<HTMLButtonElement>, index: number) {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      optionRefs.current[Math.min(index + 1, options.length - 1)]?.focus()
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      optionRefs.current[Math.max(index - 1, 0)]?.focus()
    } else if (e.key === 'Home') {
      e.preventDefault()
      optionRefs.current[0]?.focus()
    } else if (e.key === 'End') {
      e.preventDefault()
      optionRefs.current[options.length - 1]?.focus()
    }
  }

  return (
    <div ref={containerRef} className="relative">
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={ariaLabel}
        onClick={handleToggle}
        onKeyDown={handleTriggerKeyDown}
        disabled={disabled}
        className={`field flex h-10 w-full cursor-pointer items-center justify-between gap-2 px-3 text-sm ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'} ${className ?? ''}`}
      >
        <span className="flex min-w-0 items-center gap-2 truncate">
          {current?.swatch && (
            <span
              className="h-3.5 w-3.5 shrink-0 rounded-full border border-white/25"
              style={{ background: current.swatch }}
            />
          )}
          <span className="truncate">{current?.label ?? placeholder ?? 'Seleccionar'}</span>
        </span>
        <svg
          className={`h-4 w-4 shrink-0 text-ink-mute transition-transform ${open ? 'rotate-180' : ''}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>

      {open && (
        <div
          role="listbox"
          className={`absolute z-20 w-full overflow-auto rounded-xl border border-border-strong bg-raised p-1.5 shadow-2xl shadow-black/60 ${
            openUp ? 'bottom-full mb-2' : 'mt-2'
          }`}
          style={{ maxHeight: '18rem' }}
        >
          {options.map((o, index) => {
            const selected = o.value === value
            return (
              <button
                key={o.value}
                ref={(el) => {
                  optionRefs.current[index] = el
                }}
                type="button"
                role="option"
                aria-selected={selected}
                onClick={() => {
                  onChange(o.value)
                  setOpen(false)
                  triggerRef.current?.focus()
                }}
                onKeyDown={(e) => handleOptionKeyDown(e, index)}
                className={`flex w-full cursor-pointer items-center justify-between gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                  selected
                    ? 'bg-cyan/[0.12] text-ink'
                    : 'text-ink-soft hover:bg-white/[0.06] hover:text-ink'
                }`}
              >
                <span className="flex min-w-0 items-center gap-2 truncate">
                  {o.swatch && (
                    <span
                      className="h-3.5 w-3.5 shrink-0 rounded-full border border-white/25"
                      style={{ background: o.swatch }}
                    />
                  )}
                  <span className="truncate">{o.label}</span>
                </span>
                {selected && (
                  <svg className="h-4 w-4 shrink-0 text-cyan" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M20 6 9 17l-5-5" />
                  </svg>
                )}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
