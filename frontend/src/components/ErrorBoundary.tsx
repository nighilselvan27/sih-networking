import { AlertTriangle } from 'lucide-react'
import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'

/*
 * Page-level error boundary.
 *
 * Without one, a single render failure unmounts the whole tree and the
 * operator is left staring at a blank window with no indication that
 * anything is wrong — the worst possible failure mode for a monitoring
 * console. This keeps the shell and navigation alive so the rest of the
 * console stays usable, and states plainly that this page failed.
 */

interface Props {
  children: ReactNode
  /** Remounts the boundary when the route changes. */
  resetKey?: string
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // The detail belongs in the console for whoever is debugging, not in
    // the operator's face.
    console.error('Console page failed to render:', error, info.componentStack)
  }

  componentDidUpdate(previous: Props): void {
    if (previous.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null })
    }
  }

  render(): ReactNode {
    const { error } = this.state

    if (!error) return this.props.children

    return (
      <div
        className="flex flex-col items-center justify-center py-16 text-center"
        role="alert"
      >
        <AlertTriangle
          className="h-4 w-4 text-warn"
          strokeWidth={1.5}
          aria-hidden
        />
        <p className="mt-2.5 text-sm text-text">This page failed to render</p>
        <p className="mt-1 max-w-md text-xs text-text-2">
          Navigation and the rest of the console are unaffected. The technical
          detail is in the browser console.
        </p>
        <p className="mono mt-2 max-w-lg break-words text-text-3">
          {error.message}
        </p>
        <button
          type="button"
          className="btn mt-3.5"
          onClick={() => this.setState({ error: null })}
        >
          Try again
        </button>
      </div>
    )
  }
}
