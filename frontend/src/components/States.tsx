import clsx from 'clsx'
import type { LucideIcon } from 'lucide-react'
import { AlertTriangle, RefreshCw } from 'lucide-react'
import type { ReactNode } from 'react'
import type { ApiError } from '@/lib/api'

/*
 * Empty, loading and error states.
 *
 * Small icon, one line of explanation, and — where it helps — one action.
 * No illustrations, no full-page spinners.
 */

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  compact = false,
}: {
  icon: LucideIcon
  title: string
  description?: string
  action?: ReactNode
  compact?: boolean
}) {
  return (
    <div
      className={clsx(
        'flex flex-col items-center justify-center px-6 text-center',
        compact ? 'py-8' : 'py-16',
      )}
    >
      <Icon
        className="h-4 w-4 text-text-3"
        strokeWidth={1.5}
        aria-hidden
      />
      <p className="mt-2.5 text-sm text-text">{title}</p>
      {description ? (
        <p className="mt-1 max-w-sm text-xs text-text-2">{description}</p>
      ) : null}
      {action ? <div className="mt-3">{action}</div> : null}
    </div>
  )
}

/**
 * An API failure, expressed for an operator.
 *
 * The headline says what is not working; the muted line carries the
 * technical detail. Raw stack traces are never rendered.
 */
export function ErrorState({
  error,
  title,
  onRetry,
  compact = false,
}: {
  error: ApiError | null
  title?: string
  onRetry?: () => void
  compact?: boolean
}) {
  const heading =
    title ?? (error?.status === 0 ? 'API unreachable' : 'Request failed')

  return (
    <div
      className={clsx(
        'flex flex-col items-center justify-center px-6 text-center',
        compact ? 'py-8' : 'py-14',
      )}
      role="alert"
    >
      <AlertTriangle
        className="h-4 w-4 text-warn"
        strokeWidth={1.5}
        aria-hidden
      />
      <p className="mt-2.5 text-sm text-text">{heading}</p>

      {error?.detail ? (
        <p className="mono mt-1.5 max-w-lg break-words text-text-3">
          {error.detail}
        </p>
      ) : null}

      {onRetry ? (
        <button type="button" className="btn mt-3.5" onClick={onRetry}>
          <RefreshCw className="h-3.5 w-3.5" aria-hidden />
          Retry
        </button>
      ) : null}
    </div>
  )
}

/** A live area that has connected but has not seen traffic yet. */
export function WaitingState({
  message,
  detail,
}: {
  message: string
  detail?: string
}) {
  return (
    <div className="flex flex-col items-center justify-center py-14 text-center">
      <span className="flex items-center gap-2 text-sm text-text-2">
        <span
          aria-hidden
          className="inline-block h-1.5 w-1.5 rounded-full bg-accent
                     motion-safe:animate-pulse"
        />
        {message}
      </span>
      {detail ? (
        <p className="mt-1 max-w-sm text-xs text-text-3">{detail}</p>
      ) : null}
    </div>
  )
}

/* --- Skeletons ------------------------------------------------------ */

export function SkeletonLine({ className }: { className?: string }) {
  return (
    <div
      className={clsx(
        'h-3 animate-pulse rounded-sm bg-surface-3',
        className,
      )}
    />
  )
}

export function SkeletonRows({
  rows = 8,
  columns = 6,
}: {
  rows?: number
  columns?: number
}) {
  return (
    <tbody>
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <tr key={rowIndex}>
          {Array.from({ length: columns }).map((__, cellIndex) => (
            <td key={cellIndex} className="border-b border-border px-3 py-2">
              <SkeletonLine
                className={cellIndex === 0 ? 'w-16' : 'w-full max-w-[92px]'}
              />
            </td>
          ))}
        </tr>
      ))}
    </tbody>
  )
}
