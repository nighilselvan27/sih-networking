import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '@/lib/api'

export interface Async<T> {
  data: T | null
  error: ApiError | null
  /** True only before the first successful load, so refreshes don't flash. */
  loading: boolean
  /** True while a refresh is in flight over existing data. */
  refreshing: boolean
  reload: () => void
}

/**
 * Fetch once, then re-fetch on an interval.
 *
 * Keeps the previous value while refreshing so live numbers update in place
 * instead of collapsing to a skeleton every few seconds. Polling pauses
 * while the tab is hidden — there is no operator watching, and a background
 * tab hammering the capture host is not free.
 */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number | null,
  deps: unknown[] = [],
): Async<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  const alive = useRef(true)
  const inFlight = useRef(false)

  const run = useCallback(async () => {
    if (inFlight.current) return
    inFlight.current = true

    setRefreshing(true)

    try {
      const next = await fetcherRef.current()

      if (!alive.current) return

      setData(next)
      setError(null)
    } catch (caught) {
      if (!alive.current) return

      setError(
        caught instanceof ApiError
          ? caught
          : new ApiError('Unexpected error.', 0, String(caught)),
      )
    } finally {
      if (alive.current) {
        setLoading(false)
        setRefreshing(false)
      }
      inFlight.current = false
    }
  }, [])

  useEffect(() => {
    alive.current = true
    setLoading(true)
    void run()

    return () => {
      alive.current = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => {
    if (intervalMs === null) return

    let timer: number | null = null

    const tick = () => {
      if (document.visibilityState === 'visible') void run()
    }

    timer = window.setInterval(tick, intervalMs)

    const onVisible = () => {
      if (document.visibilityState === 'visible') void run()
    }

    document.addEventListener('visibilitychange', onVisible)

    return () => {
      if (timer !== null) window.clearInterval(timer)
      document.removeEventListener('visibilitychange', onVisible)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, ...deps])

  return { data, error, loading, refreshing, reload: run }
}
