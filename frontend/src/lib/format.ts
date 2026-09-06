/*
 * Display formatting.
 *
 * Rule: never round a model score to the point where the operator cannot
 * compare it against a threshold. Scores keep enough precision to verify
 * the gate decision by eye.
 */

const NBSP = ' '

export function formatInt(value: number): string {
  if (!Number.isFinite(value)) return '—'
  return Math.round(value).toLocaleString('en-US')
}

function formatDecimal(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return '—'
  return value.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

/** Bits per second, in the unit a network engineer expects. */
export function formatBitrate(bitsPerSecond: number): {
  value: string
  unit: string
} {
  if (!Number.isFinite(bitsPerSecond) || bitsPerSecond <= 0) {
    return { value: '0', unit: 'bps' }
  }

  const units = ['bps', 'Kbps', 'Mbps', 'Gbps']
  let value = bitsPerSecond
  let index = 0

  while (value >= 1000 && index < units.length - 1) {
    value /= 1000
    index += 1
  }

  return {
    value: value >= 100 ? formatDecimal(value, 0) : formatDecimal(value, 1),
    unit: units[index] ?? 'bps',
  }
}

/** Bytes, base-1024, as used for volume totals. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'

  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = bytes
  let index = 0

  while (value >= 1024 && index < units.length - 1) {
    value /= 1024
    index += 1
  }

  const digits = index === 0 ? 0 : value >= 100 ? 0 : 1

  return `${formatDecimal(value, digits)}${NBSP}${units[index]}`
}

export function formatRate(value: number): string {
  if (!Number.isFinite(value)) return '—'
  if (value === 0) return '0'
  if (value >= 100) return formatInt(value)
  if (value >= 10) return formatDecimal(value, 1)
  return formatDecimal(value, 2)
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return '—'
  if (seconds < 1) return `${formatDecimal(seconds * 1000, 0)} ms`
  if (seconds < 60) return `${formatDecimal(seconds, 1)} s`

  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)

  if (minutes < 60) return `${minutes}m ${rest}s`

  const hours = Math.floor(minutes / 60)
  return `${hours}h ${minutes % 60}m`
}

export function formatPercent(fraction: number, digits = 1): string {
  if (!Number.isFinite(fraction)) return '—'
  return `${formatDecimal(fraction * 100, digits)}%`
}

/**
 * Model scores, kept at enough precision to check them against a threshold
 * by eye. Six decimals matches what the backend rounds probabilities to;
 * reconstruction error is passed 8, because it can be very small and must
 * stay comparable to the 0.25541964 threshold.
 */
export function formatScore(value: number, digits = 6): string {
  if (!Number.isFinite(value)) return '—'
  return value.toFixed(digits)
}

/**
 * Timestamps arrive either from the capture host (local, no zone) or from
 * the server (UTC, ISO). Parse both, and never silently render an invalid
 * date as a plausible-looking time.
 */
function parseTimestamp(value: string): Date | null {
  if (!value) return null

  const parsed = new Date(value)

  return Number.isNaN(parsed.getTime()) ? null : parsed
}

export function formatClock(value: string | number): string {
  const date =
    typeof value === 'number' ? new Date(value * 1000) : parseTimestamp(value)

  if (!date) return '—'

  return date.toLocaleTimeString('en-GB', { hour12: false })
}

export function formatClockSeconds(epochSeconds: number): string {
  const date = new Date(epochSeconds * 1000)

  return date.toLocaleTimeString('en-GB', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export function formatDateTime(value: string | number): string {
  const date =
    typeof value === 'number' ? new Date(value * 1000) : parseTimestamp(value)

  if (!date) return '—'

  return `${date.toLocaleDateString('en-CA')} ${date.toLocaleTimeString(
    'en-GB',
    { hour12: false },
  )}`
}

