import clsx from 'clsx'
import type { FlowRecord, Risk } from '@/lib/types'

/*
 * State indicators.
 *
 * A 6px dot plus a word. No pills, no badges, no emoji. Colour means:
 *   red    malicious / critical
 *   amber  elevated
 *   green  benign / healthy
 *   blue   active / informational
 *   grey   unknown or inactive
 */

export type Tone = 'danger' | 'warn' | 'ok' | 'accent' | 'muted'

const DOT: Record<Tone, string> = {
  danger: 'bg-danger',
  warn: 'bg-warn',
  ok: 'bg-ok',
  accent: 'bg-accent',
  muted: 'bg-text-3',
}

const TEXT: Record<Tone, string> = {
  danger: 'text-danger',
  warn: 'text-warn',
  ok: 'text-ok',
  accent: 'text-accent',
  muted: 'text-text-2',
}

export function StatusDot({
  tone,
  pulse = false,
  className,
}: {
  tone: Tone
  pulse?: boolean
  className?: string
}) {
  return (
    <span
      aria-hidden
      className={clsx(
        'inline-block h-1.5 w-1.5 shrink-0 rounded-full transition-colors',
        DOT[tone],
        pulse && 'motion-safe:animate-pulse',
        className,
      )}
    />
  )
}

export function Status({
  tone,
  label,
  pulse = false,
  muted = false,
  className,
}: {
  tone: Tone
  label: string
  pulse?: boolean
  /** Render the text in the secondary colour instead of the tone colour. */
  muted?: boolean
  className?: string
}) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 whitespace-nowrap',
        className,
      )}
    >
      <StatusDot tone={tone} pulse={pulse} />
      <span className={muted ? 'text-text-2' : TEXT[tone]}>{label}</span>
    </span>
  )
}

/* --- Risk ---------------------------------------------------------- */

/*
 * Three malicious bands, three distinct readings — red, amber, and plain
 * secondary text. Giving High and Medium the same colour would make the
 * severity column useless for scanning, and adding a fourth hue would
 * spend a colour on a distinction that weight already carries.
 */
function riskTone(risk: string): Tone {
  switch (risk.toUpperCase()) {
    case 'CRITICAL':
      return 'danger'
    case 'HIGH':
      return 'warn'
    case 'MEDIUM':
      return 'muted'
    case 'LOW':
      return 'muted'
    case 'SAFE':
      return 'ok'
    default:
      return 'muted'
  }
}

/** Risk bands come from inference.py:928-950. */
export function RiskLabel({ risk }: { risk: Risk | string }) {
  if (!risk) return <span className="text-text-3">—</span>

  const tone = riskTone(risk)

  return (
    <span
      className={clsx(
        'text-xs',
        tone === 'danger' && 'font-medium text-danger',
        tone === 'warn' && 'text-warn',
        (tone === 'muted' || tone === 'ok') && 'text-text-2',
      )}
    >
      {risk.charAt(0) + risk.slice(1).toLowerCase()}
    </span>
  )
}

/*
 * The verdict.
 *
 * The model is binary: MALICIOUS or BENIGN. There is no attack family and
 * none is shown. Risk is the secondary qualifier, and it is the only extra
 * information the detector produces.
 */
export function Verdict({
  flow,
  showRisk = true,
}: {
  flow: Pick<FlowRecord, 'prediction' | 'label' | 'risk'>
  showRisk?: boolean
}) {
  const malicious = flow.prediction === 1

  return (
    <span className="inline-flex items-center gap-1.5">
      <StatusDot tone={malicious ? 'danger' : 'ok'} />
      <span
        className={clsx(
          'text-xs',
          malicious ? 'font-medium text-danger' : 'text-text-2',
        )}
      >
        {malicious ? 'Malicious' : 'Benign'}
      </span>
      {showRisk && malicious && flow.risk ? (
        <span className="text-2xs text-text-3">{flow.risk}</span>
      ) : null}
    </span>
  )
}

/**
 * A score against its threshold. The bar is a comparison aid, not
 * decoration: it is only ever drawn where a threshold genuinely exists.
 */
export function ScoreBar({
  value,
  threshold,
  tone = 'accent',
}: {
  /** 0..1 */
  value: number
  /** 0..1, drawn as a tick */
  threshold?: number
  tone?: Tone
}) {
  const clamped = Math.max(0, Math.min(value, 1))

  return (
    <div className="relative h-1 w-full overflow-hidden rounded-sm bg-surface-3">
      <div
        className={clsx('h-full rounded-sm', DOT[tone])}
        style={{ width: `${clamped * 100}%` }}
      />
      {threshold !== undefined && threshold > 0 && threshold <= 1 ? (
        <span
          aria-hidden
          className="absolute top-0 h-full w-px bg-text-3"
          style={{ left: `${threshold * 100}%` }}
        />
      ) : null}
    </div>
  )
}
