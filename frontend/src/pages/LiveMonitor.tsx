import { Pause, Play, Radio, RotateCw } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { FilterBar, SearchField, Select } from '@/components/Controls'
import { DemoControls } from '@/components/DemoControls'
import { Drawer } from '@/components/Drawer'
import { FlowDetail } from '@/components/FlowDetail'
import { FlowTable } from '@/components/FlowTable'
import { Status } from '@/components/Indicators'
import { Metric, MetricRow } from '@/components/Metric'
import { Section, SectionHeader } from '@/components/Section'
import { EmptyState, ErrorState, WaitingState } from '@/components/States'
import { useStream } from '@/hooks/useStream'
import { api } from '@/lib/api'
import { formatBitrate, formatInt, formatRate } from '@/lib/format'
import { flowStream } from '@/lib/stream'
import type { FlowRecord } from '@/lib/types'
import { captureDescription, useSystem } from '@/state/SystemContext'

/*
 * Live Monitor.
 *
 * The operational view: every flow the detector scores, as it is scored.
 * Rows arrive over the event stream; the table is seeded from the API's
 * buffer so the page is useful the moment it opens.
 */

const PROTOCOLS = [
  { value: '', label: 'All protocols' },
  { value: 'tcp', label: 'TCP' },
  { value: 'udp', label: 'UDP' },
]

const VERDICTS = [
  { value: '', label: 'All detections' },
  { value: 'malicious', label: 'Malicious' },
  { value: 'benign', label: 'Benign' },
]

export function LiveMonitor() {
  const { stats, error } = useSystem()
  const stream = useStream()

  const [search, setSearch] = useState('')
  const [protocol, setProtocol] = useState('')
  const [verdict, setVerdict] = useState('')
  const [selected, setSelected] = useState<FlowRecord | null>(null)

  // Seed from the API buffer so the table is not empty while waiting for
  // the first live event.
  useEffect(() => {
    let alive = true

    void api
      .flows({ limit: 100 })
      .then((page) => {
        if (alive) flowStream.seed(page.flows)
      })
      .catch(() => {
        /* the stream indicator already reports connection trouble */
      })

    return () => {
      alive = false
    }
  }, [])

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase()

    return stream.flows.filter((flow) => {
      if (protocol && flow.protocol !== protocol) return false

      if (verdict === 'malicious' && flow.prediction !== 1) return false
      if (verdict === 'benign' && flow.prediction !== 0) return false

      if (!needle) return true

      return (
        `${flow.src_ip}:${flow.src_port} ${flow.dst_ip}:${flow.dst_port} ` +
        `${flow.protocol} ${flow.state} ${flow.label} ${flow.risk}`
      )
        .toLowerCase()
        .includes(needle)
    })
  }, [stream.flows, search, protocol, verdict])

  const stale = Boolean(error)
  const capture = captureDescription(stats, error)
  const bitrate = formatBitrate(stats?.bits_per_second ?? 0)

  const filtersActive = Boolean(search || protocol || verdict)

  return (
    <>
      <MetricRow>
        <Metric
          label="Capture"
          variant="status"
          value={
            <Status
              tone={capture.tone}
              label={capture.label}
              pulse={capture.tone === 'ok'}
            />
          }
          note={capture.detail}

          stale={stale}
        />
        <Metric
          label="Flows / sec"
          value={formatRate(stats?.flows_per_second ?? 0)}
          note={`Last ${stats?.window_seconds ?? 60}s`}
          stale={stale}
        />
        <Metric
          label="Packets / sec"
          value={formatRate(stats?.packets_per_second ?? 0)}
          note={`${formatInt(stats?.packets ?? 0)} in window`}
          stale={stale}
        />
        <Metric
          label="Throughput"
          value={bitrate.value}
          unit={bitrate.unit}
          note="Observed at the API"
          stale={stale}
        />
        <Metric
          label="Detections"
          value={formatInt(stats?.threats ?? 0)}
          tone={(stats?.threats ?? 0) > 0 ? 'danger' : 'default'}
          note={`${formatInt(stats?.totals.threats ?? 0)} since start`}
          stale={stale}
        />
      </MetricRow>

      <Section first className="pt-6">
        <SectionHeader
          title="Flow stream"
          description={
            stream.paused
              ? `Paused. ${stream.bufferedWhilePaused} flow${
                  stream.bufferedWhilePaused === 1 ? '' : 's'
                } arrived while paused.`
              : 'Scored flows, newest first'
          }
          actions={
            <>
              <button
                type="button"
                className="btn"
                onClick={() =>
                  stream.paused ? flowStream.resume() : flowStream.pause()
                }
              >
                {stream.paused ? (
                  <>
                    <Play className="h-3.5 w-3.5" aria-hidden />
                    Resume
                  </>
                ) : (
                  <>
                    <Pause className="h-3.5 w-3.5" aria-hidden />
                    Pause
                  </>
                )}
              </button>
              {stream.status === 'error' ? (
                <button
                  type="button"
                  className="btn"
                  onClick={() => flowStream.reconnect()}
                >
                  <RotateCw className="h-3.5 w-3.5" aria-hidden />
                  Reconnect
                </button>
              ) : null}
            </>
          }
        />

        <FilterBar
          right={
            <span className="text-2xs text-text-3">
              {filtered.length === stream.flows.length
                ? `${filtered.length} flows`
                : `${filtered.length} of ${stream.flows.length} flows`}
            </span>
          }
        >
          <SearchField
            value={search}
            onChange={setSearch}
            placeholder="Search address, port or state"
            className="w-full sm:w-72"
          />
          <Select
            label="Protocol"
            value={protocol}
            onChange={setProtocol}
            options={PROTOCOLS}
          />
          <Select
            label="Detection"
            value={verdict}
            onChange={setVerdict}
            options={VERDICTS}
          />
        </FilterBar>

        {stream.status === 'error' && stream.flows.length === 0 ? (
          <ErrorState
            error={null}
            title="Stream disconnected"
            onRetry={() => flowStream.reconnect()}
            compact
          />
        ) : stream.flows.length === 0 && stream.status === 'connecting' ? (
          <WaitingState
            message="Connecting to the flow stream…"
            detail="The console subscribes to detections as they are scored."
          />
        ) : (
          <FlowTable
            flows={filtered}
            columns={[
              'time',
              'source',
              'destination',
              'protocol',
              'state',
              'packets',
              'bytes',
              'duration',
              'verdict',
              'xgboost',
            ]}
            onSelect={setSelected}
            selectedSeq={selected?.seq ?? null}
            animateNew={!stream.paused}
            caption="Live scored flows"
            emptyState={
              filtersActive ? (
                <EmptyState
                  icon={Radio}
                  title="No flows match these filters"
                  description="Clear or widen the filters to see live traffic."
                  compact
                />
              ) : (
                <WaitingState
                  message="Waiting for traffic…"
                  detail={
                    'The API is connected but no flow has been scored yet. ' +
                    'Start scripts/live_capture.py, then generate traffic.'
                  }
                />
              )
            }
          />
        )}
      </Section>

      <Section>
        <DemoControls />
      </Section>

      <Drawer
        open={selected !== null}
        onClose={() => setSelected(null)}
        title={selected ? `${selected.src_ip} → ${selected.dst_ip}` : 'Flow'}
        subtitle={
          selected
            ? `${selected.protocol.toUpperCase()} · flow #${selected.seq}`
            : undefined
        }
      >
        {selected ? <FlowDetail flow={selected} /> : null}
      </Drawer>
    </>
  )
}
