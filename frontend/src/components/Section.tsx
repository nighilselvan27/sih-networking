import clsx from 'clsx'
import type { ReactNode } from 'react'

/*
 * Section headings.
 *
 * The page title lives in the application header (see TopBar), so pages do
 * not repeat it. There is no hero type in this product — an operator
 * already knows which console they opened.
 */

export function SectionHeader({
  title,
  description,
  actions,
  className,
}: {
  title: string
  description?: ReactNode
  actions?: ReactNode
  className?: string
}) {
  return (
    <div
      className={clsx(
        'flex flex-wrap items-end justify-between gap-3 pb-3',
        className,
      )}
    >
      <div className="min-w-0">
        <h2 className="text-sm font-medium text-text">{title}</h2>
        {description ? (
          <p className="mt-0.5 text-xs text-text-2">{description}</p>
        ) : null}
      </div>
      {actions ? (
        <div className="flex flex-wrap items-center gap-2">{actions}</div>
      ) : null}
    </div>
  )
}

/** A section divided from the one above by a rule rather than a card. */
export function Section({
  children,
  className,
  first = false,
}: {
  children: ReactNode
  className?: string
  first?: boolean
}) {
  return (
    <section
      className={clsx(
        first ? 'pt-1' : 'mt-8 border-t border-border pt-6',
        className,
      )}
    >
      {children}
    </section>
  )
}

/**
 * Attribution for anything read from a file on disk. Every benchmark table
 * carries one, so a reader can check the number at its source.
 */
export function SourceNote({
  file,
  note,
}: {
  file: string
  note?: string
}) {
  return (
    <p className="mt-2 text-2xs text-text-3">
      Source <span className="font-mono">{file}</span>
      {note ? <span> · {note}</span> : null}
    </p>
  )
}

/** A labelled value pair, used throughout the detail views. */
export function Field({
  label,
  children,
  mono = false,
  className,
}: {
  label: string
  children: ReactNode
  mono?: boolean
  className?: string
}) {
  return (
    <div className={clsx('min-w-0', className)}>
      <dt className="label-field">{label}</dt>
      <dd
        className={clsx(
          'mt-0.5 break-words text-xs text-text',
          mono && 'font-mono tabular',
        )}
      >
        {children}
      </dd>
    </div>
  )
}

/**
 * Label on the left, value on the right, one per line.
 *
 * Used where the value is a long technical figure — a two-column grid in a
 * narrow card wraps a threshold like 0.25541964 across lines, which makes
 * it unreadable and, worse, easy to misread.
 */
export function StatRow({
  label,
  children,
  mono = true,
}: {
  label: string
  children: ReactNode
  mono?: boolean
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <dt className="text-2xs text-text-3">{label}</dt>
      <dd
        className={clsx(
          'shrink-0 text-xs text-text',
          mono && 'font-mono tabular',
        )}
      >
        {children}
      </dd>
    </div>
  )
}

export function StatList({ children }: { children: ReactNode }) {
  return <dl className="mt-2.5 divide-y divide-border">{children}</dl>
}

export function FieldGrid({
  children,
  columns = 2,
}: {
  children: ReactNode
  columns?: 2 | 3
}) {
  return (
    <dl
      className={clsx(
        'grid gap-x-6 gap-y-3.5',
        columns === 2 ? 'grid-cols-2' : 'grid-cols-2 sm:grid-cols-3',
      )}
    >
      {children}
    </dl>
  )
}
