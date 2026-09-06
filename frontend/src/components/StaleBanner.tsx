import { RefreshCw } from 'lucide-react'
import { useEffect, useState } from 'react'
import { StatusDot } from './Indicators'
import { flowStream } from '@/lib/stream'
import { useSystem } from '@/state/SystemContext'

/*
 * Connection-loss notice.
 *
 * When the API stops answering, the last figures fetched are still on
 * screen. In a monitoring console that is the most dangerous failure there
 * is: an operator reads a healthy-looking number and concludes the network
 * is fine, when in fact nothing has been measured for minutes. This says
 * plainly that the values are frozen and how old they are.
 */

function useTicker(active: boolean): number {
  const [, setTick] = useState(0)

  useEffect(() => {
    if (!active) return

    const timer = window.setInterval(() => setTick((n) => n + 1), 1000)
    return () => window.clearInterval(timer)
  }, [active])

  return 0
}

export function StaleBanner() {
  const { error, lastSuccessAt, reload } = useSystem()

  useTicker(Boolean(error))

  if (!error) return null

  const age =
    lastSuccessAt === null
      ? null
      : Math.max(0, Math.round((Date.now() - lastSuccessAt) / 1000))

  return (
    <div
      role="alert"
      className="mb-4 flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-md
                 border border-danger bg-danger-weak px-3 py-2"
    >
      <StatusDot tone="danger" />

      <span className="text-xs font-medium text-text">API unreachable</span>

      <span className="text-xs text-text-2">
        {age === null
          ? 'No data has been received since this console opened.'
          : `Figures on this page are frozen as of ${age}s ago and are not live.`}
      </span>

      <button
        type="button"
        className="btn ml-auto"
        onClick={() => {
          reload()
          // Also clear the stream's backoff, which can have grown to half
          // a minute; otherwise Retry appears to do nothing to the live
          // table.
          flowStream.reconnect()
        }}
      >
        <RefreshCw className="h-3.5 w-3.5" aria-hidden />
        Retry
      </button>
    </div>
  )
}
