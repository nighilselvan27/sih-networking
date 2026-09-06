import { Play, Square, Terminal } from 'lucide-react'
import { useState } from 'react'
import { SectionHeader } from './Section'
import { Status } from './Indicators'
import { usePolling } from '@/hooks/usePolling'
import { api, ApiError } from '@/lib/api'

/*
 * Controlled traffic generation.
 *
 * The backend only accepts a preset name from a fixed server-side list, and
 * only from loopback with IDS_DEMO_CONTROLS=1 set on the API process. This
 * panel is hidden entirely when the API reports the controls as disabled,
 * and it shows the equivalent terminal command either way so the same
 * demonstration can be driven by hand.
 */

export function DemoControls() {
  const status = usePolling(() => api.demoStatus(), 2000, [])
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const data = status.data

  // Nothing is rendered until the API has answered, so the panel never
  // flashes into view and then disappears.
  if (!data) return null

  const running = data.running

  const start = async (preset: string) => {
    setBusy(preset)
    setError(null)

    try {
      await api.demoRun(preset)
      status.reload()
    } catch (caught) {
      setError(
        caught instanceof ApiError ? caught.detail : 'Could not start the run.',
      )
    } finally {
      setBusy(null)
    }
  }

  const stop = async () => {
    setBusy('stop')
    setError(null)

    try {
      await api.demoStop()
      status.reload()
    } catch (caught) {
      setError(
        caught instanceof ApiError ? caught.detail : 'Could not stop the run.',
      )
    } finally {
      setBusy(null)
    }
  }

  if (!data.enabled) {
    return (
      <>
        <SectionHeader
          title="Traffic generation"
          description="Controlled, local-only test traffic"
        />
        <p className="text-xs text-text-2">
          {data.reason ??
            `Disabled. Set ${data.env_flag}=1 on the API process to run the
             generator from here.`}
        </p>
        <p className="mt-2 text-2xs text-text-3">
          The same demonstration runs from a terminal with{' '}
          <span className="font-mono">scripts/test_traffic.py</span>, as
          documented in the project README.
        </p>
      </>
    )
  }

  return (
    <>
      <SectionHeader
        title="Traffic generation"
        description="Controlled, local-only test traffic"
        actions={
          running ? (
            <button
              type="button"
              className="btn"
              onClick={stop}
              disabled={busy !== null}
            >
              <Square className="h-3.5 w-3.5" aria-hidden />
              Stop
            </button>
          ) : null
        }
      />

      {running ? (
        <div className="mb-3 flex items-center gap-2 text-xs">
          <Status tone="accent" label={`Running ${running.label}`} pulse />
          <span className="mono text-text-3">{running.command}</span>
        </div>
      ) : null}

      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {data.presets.map((preset) => (
          <div
            key={preset.name}
            className="panel flex flex-col p-3"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="text-xs font-medium text-text">
                  {preset.label}
                </div>
                <p className="mt-0.5 text-2xs text-text-2">
                  {preset.description}
                </p>
              </div>
              <button
                type="button"
                className="btn shrink-0"
                onClick={() => start(preset.name)}
                disabled={busy !== null || running !== null}
                aria-label={`Run ${preset.label}`}
              >
                <Play className="h-3.5 w-3.5" aria-hidden />
                Run
              </button>
            </div>

            <p className="mt-2 text-2xs text-text-3">{preset.expectation}</p>

            <div
              className="mono mt-2.5 flex items-start gap-1.5 border-t
                         border-border pt-2 text-text-3"
            >
              <Terminal className="mt-0.5 h-3 w-3 shrink-0" aria-hidden />
              <span className="break-all">{preset.command}</span>
            </div>
          </div>
        ))}
      </div>

      {error ? (
        <p className="mt-3 text-xs text-danger" role="alert">
          {error}
        </p>
      ) : null}

      <p className="mt-3 text-2xs text-text-3">
        Presets are fixed on the server; destination, rate and duration are
        not accepted from this page. The generator refuses any target that is
        not loopback or private, and each run is capped at{' '}
        {data.max_run_seconds}s. Verdicts always come from the trained models.
      </p>
    </>
  )
}
