import { X } from 'lucide-react'
import { useEffect, useRef } from 'react'
import type { ReactNode } from 'react'

/*
 * Side drawer for investigation detail.
 *
 * Escape closes, focus moves in on open and returns to the trigger on
 * close, and focus is trapped while open. The overlay is a plain scrim, not
 * a blur.
 */

export function Drawer({
  open,
  onClose,
  title,
  subtitle,
  children,
  footer,
}: {
  open: boolean
  onClose: () => void
  title: ReactNode
  subtitle?: ReactNode
  children: ReactNode
  footer?: ReactNode
}) {
  const panel = useRef<HTMLDivElement>(null)
  const restoreTo = useRef<HTMLElement | null>(null)

  useEffect(() => {
    if (!open) return

    restoreTo.current = document.activeElement as HTMLElement | null

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        onClose()
        return
      }

      if (event.key !== 'Tab' || !panel.current) return

      const focusable = panel.current.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input, select, textarea, [tabindex]:not([tabindex="-1"])',
      )

      if (focusable.length === 0) return

      const first = focusable[0]
      const last = focusable[focusable.length - 1]

      if (!first || !last) return

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onKeyDown)

    // Move focus into the panel without stealing it from a text field the
    // operator may have been typing in when the row was activated.
    const timer = window.setTimeout(() => panel.current?.focus(), 0)

    return () => {
      document.removeEventListener('keydown', onKeyDown)
      window.clearTimeout(timer)
      restoreTo.current?.focus?.()
    }
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <button
        type="button"
        aria-label="Close detail"
        onClick={onClose}
        className="absolute inset-0 bg-scrim animate-overlay-in
                   motion-reduce:animate-none"
        tabIndex={-1}
      />

      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-label={typeof title === 'string' ? title : 'Detail'}
        tabIndex={-1}
        className="scroll-y relative flex h-full w-full max-w-[560px]
                   flex-col border-l border-border bg-surface
                   shadow-overlay outline-none animate-panel-in
                   motion-reduce:animate-none"
      >
        <header
          className="flex items-start justify-between gap-4 border-b
                     border-border px-5 py-3.5"
        >
          <div className="min-w-0">
            <div className="text-sm font-medium text-text">{title}</div>
            {subtitle ? (
              <div className="mt-0.5 text-xs text-text-2">{subtitle}</div>
            ) : null}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="btn-quiet -mr-1.5 rounded p-1.5"
            aria-label="Close"
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        </header>

        <div className="scroll-y flex-1 overflow-y-auto px-5 py-5">
          {children}
        </div>

        {footer ? (
          <footer className="border-t border-border px-5 py-3">
            {footer}
          </footer>
        ) : null}
      </div>
    </div>
  )
}
