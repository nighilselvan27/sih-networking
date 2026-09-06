import { ShieldCheck } from 'lucide-react'
import { useState } from 'react'
import { Legend, TimeseriesChart } from '@/components/Charts'
import type { Series } from '@/components/Charts'
import { RangePicker } from '@/components/Controls'
import { Drawer } from '@/components/Drawer'
import { FlowDetail } from '@/components/FlowDetail'
import { FlowTable } from '@/components/FlowTable'
import { Metric, MetricRow } from '@/components/Metric'
import { Section, SectionHeader } from '@/components/Section'
import { EmptyState, ErrorState, SkeletonRows } from '@/components/States'
import { usePolling } from '@/hooks/usePolling'
import { api } from '@/lib/api'
import {
  formatBitrate,
  formatInt,
  formatPercent,
  formatRate,
} from '@/lib/format'
import type { FlowRecord, TimeRange } from '@/lib/types'
import { useSystem } from '@/state/SystemContext'

/*
 * Overview.
 *
 * Answers, in order: is traffic arriving, how much, is anything being
 * detected, and what were the most recent detections.
 */

/*
 * Two series, two axes. Packets/sec is deliberately not plotted here: it is
 * three orders of magnitude larger than the flow rate, so sharing an axis
 * would flatten the flow line onto the baseline. It has its own chart on
 * the Traffic page, and its current value sits in the metric row above.
 */
const SERIES: Series[] = [
  { key: 'flows_per_second', label: 'Flows/sec', color: 'series-1' },
  {
    key: 'threats',
    label: 'Detections',
    color: 'series-3',
    axis: 'right',
    format: (value) => formatInt(value),
  },
]

export function Overview() {
  const { stats, error, loading } = useSystem()
  const [range, setRange] = useState<TimeRange>('5m')
  const [selected, setSelected] = useState<FlowRecord | null>(null)

  const series = usePolling(() => api.timeseries(range), 2000, [range])

  const recent = usePolling(() => api.alerts({ limit: 8 }), 3000, [])

  const stale = Boolean(error)
  const bitrate = formatBitrate(stats?.bits_per_second ?? 0)

  if (error && !stats) {
    return (
      <ErrorState
        error={error}
        title="Detection API unreachable"
        onRetry={() => window.location.reload()}
      />
    )
  }

  const windowLabel = `Last ${stats?.window_seconds ?? 60}s`

  return (
    <>
      <MetricRow>
        <Metric
          label="Traffic"
          value={bitrate.value}
          unit={bitrate.unit}
          note={`${windowLabel}, observed at the API`}
          loading={loading}
          stale={stale}
        />
        <Metric
          label="Flows"
          value={formatInt(stats?.flows ?? 0)}
          note={windowLabel}
          loading={loading}
          stale={stale}
        />
        <Metric
          label="Flows / sec"
          value={formatRate(stats?.flows_per_second ?? 0)}
          note={`${formatRate(stats?.packets_per_second ?? 0)} packets/sec`}
          loading={loading}
          stale={stale}
        />
        <Metric
          label="Detections"
          value={formatInt(stats?.threats ?? 0)}
          tone={(stats?.threats ?? 0) > 0 ? 'danger' : 'default'}
          note={windowLabel}
          loading={loading}
          stale={stale}
        />
        <Metric
          label="Detection share"
          value={formatPercent(stats?.threat_share ?? 0)}
          note="Of flows scored in the window"
          loading={loading}
          stale={stale}
        />
      </MetricRow>

      <Section first className="pt-6">
        <SectionHeader
          title="Network activity"
          description="Live flow activity observed at the detection API"
          actions={<RangePicker value={range} onChange={setRange} />}
        />

        {series.error && !series.data ? (
          <ErrorState error={series.error} onRetry={series.reload} compact />
        ) : (
          <>
            <TimeseriesChart
              points={series.data?.points ?? []}
              series={SERIES}
              height={220}
              rightTickFormat={formatInt}
            />
            <div className="mt-2 flex flex-wrap items-center justify-between gap-3">
              <Legend series={SERIES} />
              <span className="text-2xs text-text-3">
                {series.data
                  ? `${series.data.bucket_seconds}s buckets`
                  : 'Loading'}
              </span>
            </div>
          </>
        )}
      </Section>

      <Section>
        <SectionHeader
          title="Recent detections"
          description="Flows the hybrid detector scored as malicious"
        />

        {recent.error && !recent.data ? (
          <ErrorState error={recent.error} onRetry={recent.reload} compact />
        ) : (
          <FlowTable
            flows={recent.data?.alerts ?? []}
            columns={[
              'time',
              'source',
              'destination',
              'protocol',
              'flow',
              'verdict',
              'confidence',
            ]}
            onSelect={setSelected}
            selectedSeq={selected?.seq ?? null}
            animateNew
            caption="Recent malicious detections"
            loading={
              recent.loading ? <SkeletonRows rows={5} columns={7} /> : undefined
            }
            emptyState={
              <EmptyState
                icon={ShieldCheck}
                title="No detections"
                description={
                  'Traffic is being scored. Flows classified as malicious ' +
                  'will appear here.'
                }
                compact
              />
            }
          />
        )}
      </Section>

      <Drawer
        open={selected !== null}
        onClose={() => setSelected(null)}
        title={
          selected ? `${selected.src_ip} → ${selected.dst_ip}` : 'Detection'
        }
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
