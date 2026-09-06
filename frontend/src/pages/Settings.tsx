import { Check, X } from 'lucide-react'
import { useState } from 'react'
import { Segmented } from '@/components/Controls'
import { Status } from '@/components/Indicators'
import { Field, FieldGrid, Section, SectionHeader } from '@/components/Section'
import { useTheme } from '@/hooks/useTheme'
import type { Theme } from '@/hooks/useTheme'
import { getApiBase, setApiBase } from '@/lib/api'
import { api } from '@/lib/api'
import { formatDuration, formatInt } from '@/lib/format'
import { captureDescription, useSystem } from '@/state/SystemContext'

/*
 * Settings and diagnostics.
 *
 * The diagnostics block is where a broken deployment explains itself: which
 * model artifacts the API can see, whether it loaded them, and whether the
 * console is talking to a live detector at all.
 */

const INTERVALS = [
  { value: '1000', label: '1s' },
  { value: '2000', label: '2s' },
  { value: '5000', label: '5s' },
  { value: '10000', label: '10s' },
]

const THEMES: { value: Theme; label: string }[] = [
  { value: 'system', label: 'System' },
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
]

export function Settings() {
  const { stats, health, error, intervalMs, setIntervalMs, reload } =
    useSystem()
  const [theme, setTheme] = useTheme()
  const [base, setBase] = useState(getApiBase())
  const [saved, setSaved] = useState(false)
  const [clearing, setClearing] = useState(false)

  const capture = captureDescription(stats, error)

  const saveBase = () => {
    setApiBase(base)
    setSaved(true)
    reload()
    window.setTimeout(() => setSaved(false), 2000)
  }

  const clearBuffer = async () => {
    setClearing(true)

    try {
      await api.resetBuffer()
      reload()
    } catch {
      /* the diagnostics block already reports connection state */
    } finally {
      setClearing(false)
    }
  }

  return (
    <>
      <Section first>
        <SectionHeader
          title="Connection"
          description="Where the console reads detections from"
        />

        <div className="max-w-xl space-y-4">
          <div>
            <label
              htmlFor="api-base"
              className="label-field mb-1.5 block"
            >
              API base URL
            </label>
            <div className="flex gap-2">
              <input
                id="api-base"
                type="url"
                value={base}
                onChange={(event) => setBase(event.target.value)}
                placeholder="Same origin"
                className="field flex-1 font-mono"
                spellCheck={false}
              />
              <button type="button" className="btn" onClick={saveBase}>
                {saved ? (
                  <>
                    <Check className="h-3.5 w-3.5" aria-hidden />
                    Saved
                  </>
                ) : (
                  'Apply'
                )}
              </button>
            </div>
            <p className="mt-1.5 text-2xs text-text-3">
              Leave empty to use the same origin, which is correct when the
              console is served by the API or through the development proxy.
              Set it to e.g.{' '}
              <span className="font-mono">http://127.0.0.1:8000</span> to
              point at a detector on another host.
            </p>
          </div>

          <div>
            <span className="label-field mb-1.5 block">Refresh interval</span>
            <Segmented
              value={String(intervalMs)}
              onChange={(value) => setIntervalMs(Number(value))}
              options={INTERVALS}
              label="Refresh interval"
            />
            <p className="mt-1.5 text-2xs text-text-3">
              How often the console polls for counters. The live flow table
              uses a persistent event stream and is not affected. Polling
              pauses while the tab is in the background.
            </p>
          </div>
        </div>
      </Section>

      <Section>
        <SectionHeader title="Display" />

        <div>
          <span className="label-field mb-1.5 block">Theme</span>
          <Segmented
            value={theme}
            onChange={setTheme}
            options={THEMES}
            label="Theme"
          />
        </div>
      </Section>

      <Section>
        <SectionHeader
          title="Diagnostics"
          description="Live state of the detection API"
        />

        {health?.replay ? (
          <div className="panel mb-5 border-warn p-3">
            <Status tone="warn" label="Replay server" />
            <p className="mt-1.5 text-xs text-text-2">
              This console is connected to{' '}
              <span className="font-mono">scripts/replay_api.py</span>, not to
              the live detector. No model is loaded and no packet is being
              captured
              {health.source ? (
                <>
                  {' '}
                  — the source is{' '}
                  <span className="font-mono">{health.source}</span>
                </>
              ) : null}
              .
            </p>
          </div>
        ) : null}

        <FieldGrid columns={3}>
          <Field label="API">
            {error ? (
              <Status tone="danger" label="Unreachable" />
            ) : (
              <Status tone="ok" label={health?.status ?? 'Reachable'} />
            )}
          </Field>
          <Field label="Models">
            <Status
              tone={health?.models_loaded ? 'ok' : 'warn'}
              label={health?.models ?? 'Unknown'}
            />
          </Field>
          <Field label="Capture">
            <Status tone={capture.tone} label={capture.label} />
          </Field>

          <Field label="Buffer" mono>
            {stats
              ? `${formatInt(stats.buffer_size)} / ${formatInt(
                  stats.buffer_capacity,
                )}`
              : '—'}
          </Field>
          <Field label="Stream subscribers" mono>
            {stats ? formatInt(stats.subscribers) : '—'}
          </Field>
          <Field label="API uptime" mono>
            {stats ? formatDuration(stats.uptime_seconds) : '—'}
          </Field>

          <Field label="Flows scored" mono>
            {stats ? formatInt(stats.totals.flows) : '—'}
          </Field>
          <Field label="Detections" mono>
            {stats ? formatInt(stats.totals.threats) : '—'}
          </Field>
          <Field label="Prediction errors" mono>
            {stats ? formatInt(stats.totals.errors) : '—'}
          </Field>
        </FieldGrid>

        {error ? (
          <p className="mono mt-4 text-danger">{error.detail}</p>
        ) : null}

        {health?.artifacts && health.artifacts.length > 0 ? (
          <div className="mt-6">
            <h3 className="mb-2.5 text-xs font-medium text-text">
              Model artifacts
            </h3>
            <div className="scroll-x max-w-2xl">
              <table className="tbl">
                <thead>
                  <tr>
                    <th scope="col">Artifact</th>
                    <th scope="col">File</th>
                    <th scope="col">Present</th>
                  </tr>
                </thead>
                <tbody>
                  {health.artifacts.map((artifact) => (
                    <tr key={artifact.name}>
                      <td>{artifact.name}</td>
                      <td className="font-mono text-text-2">
                        {artifact.file}
                      </td>
                      <td>
                        {artifact.present ? (
                          <span className="inline-flex items-center gap-1 text-ok">
                            <Check className="h-3 w-3" aria-hidden />
                            Yes
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-danger">
                            <X className="h-3 w-3" aria-hidden />
                            Missing
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {health.missing_artifacts &&
            health.missing_artifacts.length > 0 ? (
              <p className="mt-2.5 max-w-2xl text-xs text-text-2">
                The API cannot load its models while an artifact is missing.
                Regenerate the scaler with{' '}
                <span className="font-mono">
                  scripts/rebuild_autoencoder_scaler.py
                </span>
                .
              </p>
            ) : null}
          </div>
        ) : null}
      </Section>

      <Section>
        <SectionHeader
          title="Buffer"
          description="In-memory history of scored flows"
        />

        <p className="max-w-xl text-xs text-text-2">
          The API retains the most recent{' '}
          {stats ? formatInt(stats.buffer_capacity) : ''} scored flows in
          memory. Clearing discards that history, including alert
          acknowledgements. It does not affect capture, the models or any file
          on disk.
        </p>

        <button
          type="button"
          className="btn mt-3"
          onClick={clearBuffer}
          disabled={clearing || Boolean(error)}
        >
          {clearing ? 'Clearing…' : 'Clear buffer'}
        </button>
      </Section>
    </>
  )
}
