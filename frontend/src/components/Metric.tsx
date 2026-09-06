import clsx from 'clsx'
import type { ReactNode } from 'react'

/*
 * A metric is a value, a label and optionally one line of context.
 *
 * Deliberately not a card: metrics sit in a row separated by vertical
 * rules. Boxing each one would add five borders that carry no information.
 */

export function Metric({
  label,
  value,
  unit,
  note,
  tone,
  variant = 'number',
  stale = false,
  loading = false,
}: {
  label: string
  value: ReactNode
  unit?: string
  note?: ReactNode
  tone?: 'default' | 'danger'
  /*
   * A state word is not a measurement. Rendering "Receiving flows" at the
   * size of a figure reads as though it were one, and it crowds out the
   * numbers beside it.
   */
  variant?: 'number' | 'status'
  /** Last known value, not a current one. Rendered muted; see StaleBanner. */
  stale?: boolean
  loading?: boolean
}) {
  return (
    <div className="min-w-0 px-4 py-3 first:pl-0">
      <div className="label-field">{label}</div>

      {loading ? (
        <div className="mt-1.5 h-6 w-20 animate-pulse rounded-sm bg-surface-3" />
      ) : (
        <div className="mt-0.5 flex items-baseline gap-1">
          <span
            className={clsx(
              variant === 'number'
                ? 'tabular text-xl font-medium leading-8 tracking-[-0.01em]'
                : 'text-base font-medium leading-8',
              stale
                ? 'text-text-3'
                : tone === 'danger'
                  ? 'text-danger'
                  : 'text-text',
            )}
          >
            {value}
          </span>
          {unit ? (
            <span className="text-xs text-text-2">{unit}</span>
          ) : null}
        </div>
      )}

      {note ? (
        <div className="mt-0.5 truncate text-2xs text-text-3">{note}</div>
      ) : (
        <div className="mt-0.5 h-4" />
      )}
    </div>
  )
}

/** Metrics in a row, separated by hairlines rather than boxed in cards. */
export function MetricRow({ children }: { children: ReactNode }) {
  return (
    <div
      className="grid grid-cols-2 divide-x divide-border border-b
                 border-border sm:grid-cols-3 lg:grid-cols-5"
    >
      {children}
    </div>
  )
}

/** A compact counter, used for the alert severity summary. */
export function Counter({
  label,
  value,
  tone = 'default',
  active = false,
  onClick,
}: {
  label: string
  value: number
  tone?: 'default' | 'danger' | 'warn'
  active?: boolean
  onClick?: () => void
}) {
  const content = (
    <>
      <div
        className={clsx(
          'tabular text-lg font-medium leading-7',
          tone === 'danger' && 'text-danger',
          tone === 'warn' && 'text-warn',
          tone === 'default' && 'text-text',
        )}
      >
        {value.toLocaleString('en-US')}
      </div>
      <div className="label-field mt-0.5">{label}</div>
    </>
  )

  if (!onClick) {
    return <div className="px-4 py-2.5 first:pl-0">{content}</div>
  }

  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={clsx(
        'px-4 py-2.5 text-left transition-colors first:pl-0',
        active ? 'bg-surface-3' : 'hover:bg-surface-2',
      )}
    >
      {content}
    </button>
  )
}
