import { useState } from 'react'
import { BarList, Legend, TimeseriesChart } from '@/components/Charts'
import type { Series } from '@/components/Charts'
import { RangePicker } from '@/components/Controls'
import { Metric, MetricRow } from '@/components/Metric'
import { Section, SectionHeader } from '@/components/Section'
import { ErrorState } from '@/components/States'
import { usePolling } from '@/hooks/usePolling'
import { api } from '@/lib/api'
import {
  formatBitrate,
  formatBytes,
  formatInt,
  formatRate,
} from '@/lib/format'
import type { TimeRange } from '@/lib/types'
import { useSystem } from '@/state/SystemContext'

/*
 * Traffic.
 *
 * One primary chart, then the two breakdowns that actually inform it:
 * protocol mix and who is talking. Deliberately not six charts.
 */

function bitrateTick(value: number): string {
  const rate = formatBitrate(value)
  return `${rate.value} ${rate.unit}`
}

/* Throughput on the left axis, packet rate on its own axis on the right. */
const VOLUME: Series[] = [
  {
    key: 'bits_per_second',
    label: 'Throughput',
    color: 'series-1',
    format: bitrateTick,
  },
  {
    key: 'packets_per_second',
    label: 'Packets/sec',
    color: 'series-2',
    axis: 'right',
  },
]

export function Traffic() {
  const { stats, error } = useSystem()
  const [range, setRange] = useState<TimeRange>('15m')

  const series = usePolling(() => api.timeseries(range), 3000, [range])
  const distribution = usePolling(() => api.distribution(10), 5000, [])

  const stale = Boolean(error)
  const bitrate = formatBitrate(stats?.bits_per_second ?? 0)
  const data = distribution.data

  const protocolTotal =
    data?.protocols.reduce((sum, item) => sum + item.flows, 0) ?? 0

  return (
    <>
      <MetricRow>
        <Metric
          label="Throughput"
          value={bitrate.value}
          unit={bitrate.unit}
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
          label="Flows / sec"
          value={formatRate(stats?.flows_per_second ?? 0)}
          note={`${formatInt(stats?.flows ?? 0)} in window`}
          stale={stale}
        />
        <Metric
          label="Sources"
          value={formatInt(stats?.unique_sources ?? 0)}
          note="Distinct source addresses"
          stale={stale}
        />
        <Metric
          label="Destinations"
          value={formatInt(stats?.unique_destinations ?? 0)}
          note="Distinct destination addresses"
          stale={stale}
        />
      </MetricRow>

      <Section first className="pt-6">
        <SectionHeader
          title="Traffic volume"
          description="Throughput and packet rate observed at the detection API"
          actions={<RangePicker value={range} onChange={setRange} />}
        />

        {series.error && !series.data ? (
          <ErrorState error={series.error} onRetry={series.reload} compact />
        ) : (
          <>
            <TimeseriesChart
              points={series.data?.points ?? []}
              series={VOLUME}
              height={240}
              leftTickFormat={bitrateTick}
              leftAxisWidth={68}
            />
            <div className="mt-2 flex flex-wrap items-center justify-between gap-3">
              <Legend series={VOLUME} />
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
        <div className="grid gap-8 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
          <div>
            <SectionHeader
              title="Protocol mix"
              description={
                data
                  ? `Across ${formatInt(data.sample_size)} retained flows`
                  : undefined
              }
            />
            {distribution.error && !data ? (
              <ErrorState
                error={distribution.error}
                onRetry={distribution.reload}
                compact
              />
            ) : (
              <BarList
                items={(data?.protocols ?? []).map((item) => ({
                  label: item.protocol.toUpperCase(),
                  value: item.flows,
                }))}
                total={protocolTotal}
                emptyLabel="No flows scored yet"
                formatValue={(value) => `${formatInt(value)} flows`}
              />
            )}
          </div>

          <div className="grid gap-8 sm:grid-cols-2">
            <div>
              <SectionHeader title="Top sources" />
              <TalkerTable
                rows={data?.top_sources ?? []}
                empty={!data || data.top_sources.length === 0}
              />
            </div>
            <div>
              <SectionHeader title="Top destinations" />
              <TalkerTable
                rows={data?.top_destinations ?? []}
                empty={!data || data.top_destinations.length === 0}
              />
            </div>
          </div>
        </div>

        <p className="mt-4 text-2xs text-text-3">
          Breakdowns cover the flows currently held in the API buffer
          {stats
            ? ` (${formatInt(stats.buffer_size)} of ${formatInt(
                stats.buffer_capacity,
              )})`
            : ''}
          , not the full capture history.
        </p>
      </Section>
    </>
  )
}

function TalkerTable({
  rows,
  empty,
}: {
  rows: { address: string; flows: number; bytes: number }[]
  empty: boolean
}) {
  if (empty) {
    return <p className="py-6 text-xs text-text-3">No flows scored yet.</p>
  }

  return (
    <div className="scroll-x">
      <table className="tbl">
        <thead>
          <tr>
            <th scope="col">Address</th>
            <th scope="col" className="text-right">
              Flows
            </th>
            <th scope="col" className="text-right">
              Bytes
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.address}>
              <td className="font-mono tabular">{row.address}</td>
              <td className="num">{formatInt(row.flows)}</td>
              <td className="num">{formatBytes(row.bytes)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
