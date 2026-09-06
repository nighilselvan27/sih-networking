import clsx from 'clsx'
import { Search, X } from 'lucide-react'
import type { ReactNode } from 'react'
import { useId } from 'react'
import type { TimeRange } from '@/lib/types'

/* Form controls. Small, quiet, keyboard-reachable. */

export function SearchField({
  value,
  onChange,
  placeholder = 'Search',
  className,
}: {
  value: string
  onChange: (next: string) => void
  placeholder?: string
  className?: string
}) {
  const id = useId()

  return (
    <div className={clsx('relative', className)}>
      <label htmlFor={id} className="sr-only">
        {placeholder}
      </label>
      <Search
        className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5
                   -translate-y-1/2 text-text-3"
        aria-hidden
      />
      <input
        id={id}
        type="search"
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
        className="field w-full pl-8 pr-7"
        autoComplete="off"
        spellCheck={false}
      />
      {value ? (
        <button
          type="button"
          onClick={() => onChange('')}
          className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded-sm
                     p-0.5 text-text-3 hover:text-text"
          aria-label="Clear search"
        >
          <X className="h-3.5 w-3.5" aria-hidden />
        </button>
      ) : null}
    </div>
  )
}

export function Select({
  label,
  value,
  onChange,
  options,
  className,
}: {
  label: string
  value: string
  onChange: (next: string) => void
  options: { value: string; label: string }[]
  className?: string
}) {
  const id = useId()

  return (
    <div className={clsx('flex items-center gap-2', className)}>
      <label htmlFor={id} className="sr-only">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="field cursor-pointer pr-6"
        aria-label={label}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  )
}

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
}: {
  value: T
  onChange: (next: T) => void
  options: { value: T; label: string }[]
  label: string
}) {
  return (
    <div className="segment" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className="segment-item"
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

const RANGES: { value: TimeRange; label: string }[] = [
  { value: '1m', label: '1m' },
  { value: '5m', label: '5m' },
  { value: '15m', label: '15m' },
  { value: '1h', label: '1h' },
]

export function RangePicker({
  value,
  onChange,
}: {
  value: TimeRange
  onChange: (next: TimeRange) => void
}) {
  return (
    <Segmented
      value={value}
      onChange={onChange}
      options={RANGES}
      label="Time range"
    />
  )
}

/** A row of filters above a table. */
export function FilterBar({
  children,
  right,
}: {
  children: ReactNode
  right?: ReactNode
}) {
  return (
    <div
      className="flex flex-wrap items-center gap-2 border-b border-border
                 py-2.5"
    >
      {children}
      {right ? (
        // Drops to its own line on narrow screens rather than being clipped.
        <div className="flex w-full items-center gap-2 sm:ml-auto sm:w-auto">
          {right}
        </div>
      ) : null}
    </div>
  )
}
