import clsx from 'clsx'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useChartColors } from '@/hooks/useChartColors'
import { formatClockSeconds, formatRate } from '@/lib/format'
import type { TimeseriesPoint } from '@/lib/types'

/*
 * Charts are analytical instruments here.
 *
 * Thin strokes, horizontal gridlines only, no area fills, no gradients, no
 * point markers, no animation on update — a moving line that also fades in
 * is harder to read, not easier.
 */

export interface Series {
  key: keyof TimeseriesPoint
  label: string
  color: 'series-1' | 'series-2' | 'series-3'
  /** Right-hand axis, for a series on a different scale. */
  axis?: 'left' | 'right'
  format?: (value: number) => string
}

function ChartTooltip({
  active,
  payload,
  label,
  series,
}: {
  active?: boolean
  payload?: { dataKey?: string | number; value?: number | string }[]
  label?: number | string
  series: Series[]
}) {
  if (!active || !payload?.length) return null

  return (
    <div
      className="panel px-2.5 py-2 shadow-overlay"
      role="tooltip"
    >
      <div className="mono mb-1 text-text-2">
        {formatClockSeconds(Number(label))}
      </div>
      <ul className="space-y-0.5">
        {series.map((item) => {
          const entry = payload.find((row) => row.dataKey === item.key)
          if (!entry) return null

          const value = Number(entry.value ?? 0)

          return (
            <li
              key={String(item.key)}
              className="flex items-center gap-2 text-2xs"
            >
              <span
                aria-hidden
                className="h-px w-3"
                style={{ background: `var(--${item.color})` }}
              />
              <span className="text-text-2">{item.label}</span>
              <span className="tabular ml-auto font-mono text-text">
                {(item.format ?? formatRate)(value)}
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export function TimeseriesChart({
  points,
  series,
  height = 200,
  leftTickFormat = formatRate,
  leftAxisWidth = 52,
  rightTickFormat = formatRate,
}: {
  points: TimeseriesPoint[]
  series: Series[]
  height?: number
  /*
   * Axis formatters. A bitrate and a flow count do not read the same, and
   * two series of very different magnitude must not share an axis — the
   * smaller one flattens to the baseline and stops carrying information.
   */
  leftTickFormat?: (value: number) => string
  /* Widened when the left tick carries a unit, e.g. "80.0 Mbps". */
  leftAxisWidth?: number
  rightTickFormat?: (value: number) => string
}) {
  const colors = useChartColors()

  const hasRightAxis = series.some((item) => item.axis === 'right')

  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={points}
          margin={{ top: 4, right: hasRightAxis ? 4 : 8, bottom: 0, left: 0 }}
        >
          <CartesianGrid
            stroke={colors.grid}
            strokeWidth={1}
            vertical={false}
          />
          <XAxis
            dataKey="ts"
            type="number"
            scale="time"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(value: number) => formatClockSeconds(value)}
            tick={{ fill: colors['text-3'], fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: colors.border }}
            minTickGap={48}
            height={20}
          />
          <YAxis
            yAxisId="left"
            tick={{ fill: colors['text-3'], fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={leftAxisWidth}
            tickFormatter={leftTickFormat}
            allowDecimals
          />
          {hasRightAxis ? (
            <YAxis
              yAxisId="right"
              orientation="right"
              tick={{ fill: colors['text-3'], fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={44}
              allowDecimals={false}
              tickFormatter={rightTickFormat}
            />
          ) : null}
          <Tooltip
            content={<ChartTooltip series={series} />}
            cursor={{ stroke: colors.border, strokeWidth: 1 }}
            isAnimationActive={false}
          />
          {series.map((item) => (
            <Line
              key={String(item.key)}
              yAxisId={item.axis ?? 'left'}
              type="monotone"
              dataKey={item.key}
              stroke={colors[item.color]}
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 2.5, strokeWidth: 0 }}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

/** A compact inline legend. Not a boxed panel of its own. */
export function Legend({ series }: { series: Series[] }) {
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1">
      {series.map((item) => (
        <li
          key={String(item.key)}
          className="flex items-center gap-1.5 text-2xs text-text-2"
        >
          <span
            aria-hidden
            className="h-px w-3"
            style={{ background: `var(--${item.color})` }}
          />
          {item.label}
        </li>
      ))}
    </ul>
  )
}

/*
 * Horizontal proportion bars.
 *
 * Chosen over a pie chart because comparing lengths against a shared
 * baseline is accurate and comparing angles is not.
 */
export function BarList({
  items,
  total,
  emptyLabel = 'No data',
  formatValue,
  tone = 'accent',
}: {
  items: { label: string; value: number; note?: string }[]
  total?: number
  emptyLabel?: string
  formatValue?: (value: number) => string
  tone?: 'accent' | 'danger' | 'muted'
}) {
  const sum = total ?? items.reduce((acc, item) => acc + item.value, 0)
  const max = Math.max(...items.map((item) => item.value), 1)

  if (items.length === 0) {
    return <p className="py-6 text-center text-xs text-text-3">{emptyLabel}</p>
  }

  return (
    <ul className="space-y-2">
      {items.map((item) => {
        const share = sum > 0 ? item.value / sum : 0

        return (
          <li key={item.label}>
            <div className="flex items-baseline justify-between gap-3">
              <span className="mono truncate text-text">{item.label}</span>
              <span className="tabular shrink-0 font-mono text-2xs text-text-2">
                {(formatValue ?? ((value: number) => value.toLocaleString('en-US')))(
                  item.value,
                )}
                <span className="ml-2 text-text-3">
                  {(share * 100).toFixed(1)}%
                </span>
              </span>
            </div>
            <div className="mt-1 h-1 w-full overflow-hidden rounded-sm bg-surface-3">
              <div
                className={clsx(
                  'h-full rounded-sm',
                  tone === 'accent' && 'bg-accent',
                  tone === 'danger' && 'bg-danger',
                  tone === 'muted' && 'bg-text-3',
                )}
                style={{ width: `${(item.value / max) * 100}%` }}
              />
            </div>
            {item.note ? (
              <div className="mt-0.5 text-2xs text-text-3">{item.note}</div>
            ) : null}
          </li>
        )
      })}
    </ul>
  )
}
