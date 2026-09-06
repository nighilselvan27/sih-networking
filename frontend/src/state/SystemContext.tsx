import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react'
import type { ReactNode } from 'react'
import { api, ApiError } from '@/lib/api'
import type { Health, Stats } from '@/lib/types'

/*
 * Shared system status.
 *
 * The header, the sidebar footer and several pages all need "is the API up,
 * is traffic arriving". One poller answers for all of them, so opening more
 * pages does not multiply requests against the capture host.
 */

interface SystemState {
  stats: Stats | null
  health: Health | null
  error: ApiError | null
  loading: boolean
  /**
   * Wall-clock ms of the last successful poll. While `error` is set, any
   * numbers on screen are from this moment, not from now — and a
   * monitoring console must never let a stale figure pass for a live one.
   */
  lastSuccessAt: number | null
  /** Poll interval in ms; configurable from Settings. */
  intervalMs: number
  setIntervalMs: (next: number) => void
  reload: () => void
}

const SystemContext = createContext<SystemState | null>(null)

const INTERVAL_KEY = 'unindr.pollInterval'
const DEFAULT_INTERVAL = 2000

function readInterval(): number {
  try {
    const stored = Number(localStorage.getItem(INTERVAL_KEY))
    if (Number.isFinite(stored) && stored >= 1000 && stored <= 60_000) {
      return stored
    }
  } catch {
    /* storage unavailable */
  }
  return DEFAULT_INTERVAL
}

export function SystemProvider({ children }: { children: ReactNode }) {
  const [stats, setStats] = useState<Stats | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(true)
  const [lastSuccessAt, setLastSuccessAt] = useState<number | null>(null)
  const [intervalMs, setIntervalState] = useState<number>(readInterval)
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    let alive = true
    let timer: number | null = null
    let healthCountdown = 0

    const poll = async () => {
      if (document.visibilityState !== 'visible') return

      try {
        const next = await api.stats()
        if (!alive) return

        setStats(next)
        setError(null)
        setLastSuccessAt(Date.now())
      } catch (caught) {
        if (!alive) return

        setError(
          caught instanceof ApiError
            ? caught
            : new ApiError('Unexpected error.', 0, String(caught)),
        )
      } finally {
        if (alive) setLoading(false)
      }

      // Health changes rarely; check it roughly every ten stats polls.
      if (healthCountdown <= 0) {
        healthCountdown = 10

        try {
          const next = await api.health()
          if (alive) setHealth(next)
        } catch {
          if (alive) setHealth(null)
        }
      }

      healthCountdown -= 1
    }

    void poll()
    timer = window.setInterval(() => void poll(), intervalMs)

    const onVisible = () => {
      if (document.visibilityState === 'visible') void poll()
    }

    document.addEventListener('visibilitychange', onVisible)

    return () => {
      alive = false
      if (timer !== null) window.clearInterval(timer)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [intervalMs, nonce])

  const value = useMemo<SystemState>(
    () => ({
      stats,
      health,
      error,
      loading,
      lastSuccessAt,
      intervalMs,
      setIntervalMs: (next: number) => {
        setIntervalState(next)
        try {
          localStorage.setItem(INTERVAL_KEY, String(next))
        } catch {
          /* storage unavailable */
        }
      },
      reload: () => setNonce((value) => value + 1),
    }),
    [stats, health, error, loading, lastSuccessAt, intervalMs],
  )

  return (
    <SystemContext.Provider value={value}>{children}</SystemContext.Provider>
  )
}

export function useSystem(): SystemState {
  const context = useContext(SystemContext)

  if (!context) {
    throw new Error('useSystem must be used inside SystemProvider')
  }

  return context
}

/**
 * How to describe capture state honestly.
 *
 * The API cannot see whether the Scapy process is running — it only knows
 * when a flow last arrived. So that is exactly what is reported.
 */
export function captureDescription(
  stats: Stats | null,
  error: ApiError | null,
): { tone: 'ok' | 'warn' | 'danger' | 'muted'; label: string; detail: string } {
  if (error) {
    return {
      tone: 'danger',
      label: 'API unreachable',
      detail: error.detail,
    }
  }

  if (!stats) {
    return {
      tone: 'muted',
      label: 'Connecting',
      detail: 'Contacting the detection API.',
    }
  }

  const since = stats.seconds_since_last_flow

  if (since === null) {
    return {
      tone: 'muted',
      label: 'No flows yet',
      detail:
        'The API is up but has not scored a flow since it started. ' +
        'Start scripts/live_capture.py to feed it.',
    }
  }

  if (stats.capture_state === 'receiving') {
    return {
      tone: 'ok',
      label: 'Receiving flows',
      detail: `Last flow ${since < 1 ? 'under a second' : `${Math.round(since)}s`} ago.`,
    }
  }

  return {
    tone: 'warn',
    label: 'Idle',
    detail: `No flow received for ${Math.round(since)}s.`,
  }
}
