import { useSyncExternalStore } from 'react'
import { flowStream, type StreamState } from '@/lib/stream'

/** Subscribes this component to the shared live flow stream. */
export function useStream(): StreamState {
  return useSyncExternalStore(
    flowStream.subscribe,
    flowStream.getSnapshot,
    flowStream.getSnapshot,
  )
}
