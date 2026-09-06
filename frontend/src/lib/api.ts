import type {
  AlertPage,
  Benchmarks,
  Distribution,
  DemoStatus,
  FeatureInventory,
  FlowPage,
  FlowRecord,
  Health,
  ModelInfo,
  Stats,
  TimeRange,
  Timeseries,
} from './types'

/*
 * Typed client for the IDS API.
 *
 * The base URL is empty by default so requests go through the Vite dev
 * proxy (development) or the same origin (when the built console is served
 * by the API itself). Settings can point it elsewhere.
 */

const BASE_KEY = 'unindr.apiBase'

export function getApiBase(): string {
  try {
    return localStorage.getItem(BASE_KEY) ?? ''
  } catch {
    return ''
  }
}

export function setApiBase(value: string): void {
  try {
    const cleaned = value.trim().replace(/\/$/, '')
    if (cleaned) localStorage.setItem(BASE_KEY, cleaned)
    else localStorage.removeItem(BASE_KEY)
  } catch {
    /* storage unavailable; fall back to same-origin for this session */
  }
}

export function url(path: string): string {
  return `${getApiBase()}${path}`
}

/** An API failure carrying enough detail for the error state to be useful. */
export class ApiError extends Error {
  readonly status: number
  readonly detail: string

  constructor(message: string, status: number, detail: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function request<T>(
  path: string,
  init?: RequestInit & { timeoutMs?: number },
): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(
    () => controller.abort(),
    init?.timeoutMs ?? 15_000,
  )

  let response: Response

  try {
    response = await fetch(url(path), {
      ...init,
      signal: controller.signal,
      headers: {
        Accept: 'application/json',
        ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
        ...init?.headers,
      },
    })
  } catch (error) {
    const aborted = error instanceof DOMException && error.name === 'AbortError'

    throw new ApiError(
      aborted ? 'The request timed out.' : 'The API is not reachable.',
      0,
      aborted
        ? `No response from ${url(path)} within the timeout.`
        : `Could not connect to ${url(path) || window.location.origin}.`,
    )
  } finally {
    window.clearTimeout(timeout)
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`

    try {
      const body = (await response.json()) as { detail?: unknown }
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* non-JSON error body; the status line is the best we have */
    }

    throw new ApiError(
      `Request failed (${response.status}).`,
      response.status,
      detail,
    )
  }

  return (await response.json()) as T
}

function query(params: Record<string, string | number | boolean | undefined>) {
  const search = new URLSearchParams()

  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === '') continue
    search.set(key, String(value))
  }

  const text = search.toString()
  return text ? `?${text}` : ''
}

export const api = {
  health: () => request<Health>('/health', { timeoutMs: 6000 }),

  modelInfo: () => request<ModelInfo>('/model-info'),

  stats: (window?: number) =>
    request<Stats>(`/api/stats${query({ window })}`, { timeoutMs: 8000 }),

  timeseries: (range: TimeRange) =>
    request<Timeseries>(`/api/timeseries${query({ range })}`),

  flows: (params: {
    limit?: number
    before_seq?: number
    protocol?: string
    verdict?: string
    risk?: string
    q?: string
  }) => request<FlowPage>(`/api/flows${query(params)}`),

  flow: (seq: number) => request<FlowRecord>(`/api/flows/${seq}`),

  alerts: (params: {
    limit?: number
    risk?: string
    acknowledged?: boolean
    q?: string
  }) => request<AlertPage>(`/api/alerts${query(params)}`),

  acknowledge: (seq: number) =>
    request<FlowRecord>(`/api/alerts/${seq}/ack`, { method: 'POST' }),

  distribution: (limit?: number) =>
    request<Distribution>(`/api/distribution${query({ limit })}`),

  benchmarks: () => request<Benchmarks>('/api/benchmarks', { timeoutMs: 20_000 }),

  features: () => request<FeatureInventory>('/api/features'),

  demoStatus: () => request<DemoStatus>('/api/demo/status'),

  demoRun: (preset: string) =>
    request<{ started: boolean }>('/api/demo/run', {
      method: 'POST',
      body: JSON.stringify({ preset }),
    }),

  demoStop: () =>
    request<{ stopped: boolean }>('/api/demo/stop', { method: 'POST' }),

  resetBuffer: () =>
    request<{ cleared: boolean }>('/api/buffer/reset', { method: 'POST' }),
}
