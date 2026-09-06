import { url } from './api'
import type { FlowRecord } from './types'

/*
 * Live flow stream.
 *
 * One EventSource for the whole application, shared through
 * useSyncExternalStore. Pages that mount and unmount do not each open a
 * connection, and a page transition does not drop the stream.
 *
 * Two invariants keep this correct and fast:
 *
 *  1. `state` is only ever replaced inside `emit`, which then notifies.
 *     useSyncExternalStore reads the snapshot on every render and will
 *     re-render whenever its identity changed without a notification, so a
 *     silent mutation turns into a render loop.
 *
 *  2. Arriving flows are coalesced and applied on a fixed cadence. The
 *     capture layer can deliver flows far faster than a browser can lay out
 *     a table; one render per event lets the stream rate dictate the frame
 *     rate.
 *
 * Reconnection is on exponential backoff rather than the browser's built-in
 * EventSource retry, which gives no way to tell the operator what is
 * happening — and "am I actually connected?" is a question this product has
 * to answer honestly.
 */

export type StreamStatus = 'idle' | 'connecting' | 'open' | 'error'

export interface StreamState {
  status: StreamStatus
  /** Newest first. Capped at MAX_FLOWS. */
  flows: FlowRecord[]
  /** Wall-clock ms of the last flow or heartbeat received. */
  lastEventAt: number | null
  /** Flows that arrived while paused, so the UI can say how far behind it is. */
  bufferedWhilePaused: number
  paused: boolean
  error: string | null
  attempt: number
}

/*
 * Depth of the live tail. This is a moving window, not the history — the
 * Flows page pages through the API's own buffer for anything older.
 */
const MAX_FLOWS = 300

const FLUSH_INTERVAL = 250
const BASE_DELAY = 1000
const MAX_DELAY = 30_000

let state: StreamState = {
  status: 'idle',
  flows: [],
  lastEventAt: null,
  bufferedWhilePaused: 0,
  paused: false,
  error: null,
  attempt: 0,
}

const listeners = new Set<() => void>()

let source: EventSource | null = null
let retryTimer: number | null = null
let consumers = 0

/* Coalesced between flushes. Never read by a component. */
let pendingFlows: FlowRecord[] = []
let pendingPaused = 0
let pendingLastEventAt: number | null = null
let flushTimer: number | null = null

function emit(next: Partial<StreamState>) {
  state = { ...state, ...next }
  listeners.forEach((listener) => listener())
}

function flush() {
  flushTimer = null

  const hasFlows = pendingFlows.length > 0
  const hasPaused = pendingPaused > 0
  const hasClock = pendingLastEventAt !== null

  if (!hasFlows && !hasPaused && !hasClock) return

  const next: Partial<StreamState> = {}

  if (hasClock) {
    next.lastEventAt = pendingLastEventAt
    pendingLastEventAt = null
  }

  if (hasFlows) {
    next.status = 'open'
    next.flows = [...pendingFlows, ...state.flows].slice(0, MAX_FLOWS)
    pendingFlows = []
  }

  if (hasPaused) {
    next.bufferedWhilePaused = state.bufferedWhilePaused + pendingPaused
    pendingPaused = 0
  }

  emit(next)
}

function scheduleFlush() {
  if (flushTimer !== null) return
  flushTimer = window.setTimeout(flush, FLUSH_INTERVAL)
}

function clearPending() {
  if (flushTimer !== null) {
    window.clearTimeout(flushTimer)
    flushTimer = null
  }

  pendingFlows = []
  pendingPaused = 0
  pendingLastEventAt = null
}

function clearRetry() {
  if (retryTimer !== null) {
    window.clearTimeout(retryTimer)
    retryTimer = null
  }
}

function scheduleRetry() {
  clearRetry()

  const attempt = state.attempt + 1
  const delay = Math.min(BASE_DELAY * 2 ** (attempt - 1), MAX_DELAY)

  emit({ attempt })

  retryTimer = window.setTimeout(() => {
    if (consumers > 0) open()
  }, delay)
}

function open() {
  close()

  emit({ status: 'connecting', error: null })

  let es: EventSource

  try {
    es = new EventSource(url('/api/stream'))
  } catch {
    emit({ status: 'error', error: 'Could not open the event stream.' })
    scheduleRetry()
    return
  }

  source = es

  es.addEventListener('ready', () => {
    emit({
      status: 'open',
      error: null,
      attempt: 0,
      lastEventAt: Date.now(),
    })
  })

  es.addEventListener('heartbeat', () => {
    pendingLastEventAt = Date.now()
    scheduleFlush()
  })

  es.addEventListener('flow', (event) => {
    let record: FlowRecord

    try {
      record = JSON.parse((event as MessageEvent<string>).data) as FlowRecord
    } catch {
      return
    }

    pendingLastEventAt = Date.now()

    if (state.paused) {
      // Count only; the table must not move under the operator's cursor.
      pendingPaused += 1
    } else {
      pendingFlows = [record, ...pendingFlows].slice(0, MAX_FLOWS)
    }

    scheduleFlush()
  })

  es.onerror = () => {
    // EventSource does not distinguish "server gone" from "network blip";
    // treat both as a disconnect and back off.
    close()

    emit({ status: 'error', error: 'The event stream disconnected.' })

    scheduleRetry()
  }
}

function close() {
  clearPending()

  if (source) {
    source.onerror = null
    source.close()
    source = null
  }
}

export const flowStream = {
  subscribe(listener: () => void): () => void {
    listeners.add(listener)
    consumers += 1

    if (consumers === 1 && source === null && retryTimer === null) open()

    return () => {
      listeners.delete(listener)
      consumers -= 1

      // The stream stays open across page changes; it is torn down only
      // when nothing in the app is listening any more.
      if (consumers === 0) {
        clearRetry()
        close()
        emit({ status: 'idle', attempt: 0 })
      }
    }
  },

  getSnapshot(): StreamState {
    return state
  },

  /** Fill the table from /api/flows before the first live event arrives. */
  seed(records: FlowRecord[]) {
    if (state.flows.length > 0) return

    emit({ flows: records.slice(0, MAX_FLOWS) })
  },

  pause() {
    emit({ paused: true, bufferedWhilePaused: 0 })
  },

  resume() {
    emit({ paused: false, bufferedWhilePaused: 0 })
  },

  clear() {
    clearPending()
    emit({ flows: [], bufferedWhilePaused: 0 })
  },

  reconnect() {
    clearRetry()
    emit({ attempt: 0 })
    open()
  },
}
