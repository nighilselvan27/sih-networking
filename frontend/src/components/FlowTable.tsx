import clsx from 'clsx'
import { memo, useEffect, useMemo, useRef } from 'react'
import type { ReactNode } from 'react'
import { RiskLabel, Verdict } from './Indicators'
import {
  formatBytes,
  formatClock,
  formatDuration,
  formatInt,
  formatScore,
} from '@/lib/format'
import type { FlowRecord } from '@/lib/types'

/*
 * The flow table.
 *
 * Compact rows, hairline separators, monospace for every technical
 * identifier, numerics right-aligned and tabular so columns of digits line
 * up. Shared by Live Monitor, Flows, Alerts and the Overview summary so
 * that a flow looks identical wherever it appears.
 */

export type ColumnId =
  | 'time'
  | 'source'
  | 'destination'
  | 'protocol'
  | 'state'
  | 'flow'
  | 'packets'
  | 'bytes'
  | 'duration'
  | 'verdict'
  | 'verdict_plain'
  | 'risk'
  | 'xgboost'
  | 'autoencoder'
  | 'confidence'
  | 'status'

interface Column {
  id: ColumnId
  header: string
  /** Right-aligned numeric column. */
  numeric?: boolean
  width?: string
  render: (flow: FlowRecord) => ReactNode
}

const COLUMNS: Record<ColumnId, Column> = {
  time: {
    id: 'time',
    header: 'Time',
    width: '86px',
    render: (flow) => (
      <span className="font-mono tabular text-text-2">
        {formatClock(flow.timestamp || flow.received_at)}
      </span>
    ),
  },
  source: {
    id: 'source',
    header: 'Source',
    render: (flow) => (
      <span className="font-mono tabular">
        {flow.src_ip}
        <span className="text-text-3">:{flow.src_port}</span>
      </span>
    ),
  },
  destination: {
    id: 'destination',
    header: 'Destination',
    render: (flow) => (
      <span className="font-mono tabular">
        {flow.dst_ip}
        <span className="text-text-3">:{flow.dst_port}</span>
      </span>
    ),
  },
  protocol: {
    id: 'protocol',
    header: 'Proto',
    width: '60px',
    render: (flow) => (
      <span className="font-mono uppercase text-text-2">{flow.protocol}</span>
    ),
  },
  state: {
    id: 'state',
    header: 'State',
    width: '76px',
    render: (flow) => (
      <span className="font-mono text-text-2">{flow.state || '—'}</span>
    ),
  },
  flow: {
    id: 'flow',
    header: 'Flow',
    width: '84px',
    render: (flow) => (
      <span className="font-mono tabular text-text-2">#{flow.seq}</span>
    ),
  },
  packets: {
    id: 'packets',
    header: 'Packets',
    numeric: true,
    width: '84px',
    render: (flow) => formatInt(flow.packets),
  },
  bytes: {
    id: 'bytes',
    header: 'Bytes',
    numeric: true,
    width: '88px',
    render: (flow) => formatBytes(flow.bytes),
  },
  duration: {
    id: 'duration',
    header: 'Duration',
    numeric: true,
    width: '84px',
    render: (flow) => formatDuration(flow.duration),
  },
  verdict: {
    id: 'verdict',
    header: 'Detection',
    width: '132px',
    render: (flow) => <Verdict flow={flow} />,
  },
  // For tables that already carry a Risk column; repeating the band in the
  // detection cell says the same thing twice.
  verdict_plain: {
    id: 'verdict_plain',
    header: 'Detection',
    width: '104px',
    render: (flow) => <Verdict flow={flow} showRisk={false} />,
  },
  risk: {
    id: 'risk',
    header: 'Severity',
    width: '82px',
    render: (flow) => <RiskLabel risk={flow.risk} />,
  },
  xgboost: {
    id: 'xgboost',
    header: 'XGBoost',
    numeric: true,
    width: '96px',
    render: (flow) => (
      <span
        className={clsx(
          flow.xgboost_malicious ? 'text-danger' : 'text-text-2',
        )}
      >
        {formatScore(flow.xgboost_score)}
      </span>
    ),
  },
  autoencoder: {
    id: 'autoencoder',
    header: 'AE error',
    numeric: true,
    width: '104px',
    render: (flow) => (
      <span
        className={clsx(
          flow.autoencoder_anomalous ? 'text-warn' : 'text-text-2',
        )}
      >
        {formatScore(flow.autoencoder_score, 6)}
      </span>
    ),
  },
  confidence: {
    id: 'confidence',
    header: 'Confidence',
    numeric: true,
    width: '92px',
    render: (flow) => `${(flow.confidence * 100).toFixed(1)}%`,
  },
  status: {
    id: 'status',
    header: 'Status',
    width: '110px',
    render: (flow) =>
      flow.acknowledged ? (
        <span className="text-text-2">Acknowledged</span>
      ) : (
        <span className="text-text">Open</span>
      ),
  },
}

export function FlowTable({
  flows,
  columns,
  onSelect,
  selectedSeq,
  animateNew = false,
  emptyState,
  loading,
  caption,
}: {
  flows: FlowRecord[]
  columns: ColumnId[]
  onSelect?: (flow: FlowRecord) => void
  selectedSeq?: number | null
  /** Fade newly arrived rows in. Only used on live views. */
  animateNew?: boolean
  emptyState?: ReactNode
  loading?: ReactNode
  caption?: string
}) {
  // Memoised on the column ids so the row components below can be skipped
  // by React.memo — a new array literal every render would defeat it.
  const definitions = useMemo(
    () =>
      columns
        .map((id) => COLUMNS[id])
        .filter((column): column is Column => Boolean(column)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [columns.join(',')],
  )

  // High-water mark of rows already rendered, so only genuinely new rows
  // animate in — not every row on a re-render.
  const seen = useRef<number>(0)
  const previousMax = useRef<number>(0)

  useEffect(() => {
    const max = flows.reduce((acc, flow) => Math.max(acc, flow.seq), 0)
    previousMax.current = seen.current
    seen.current = Math.max(seen.current, max)
  }, [flows])

  const threshold = previousMax.current

  if (loading) {
    return (
      <div className="scroll-x">
        <table className="tbl">
          {caption ? <caption className="sr-only">{caption}</caption> : null}
          <Head definitions={definitions} />
          {loading}
        </table>
      </div>
    )
  }

  if (flows.length === 0 && emptyState) {
    return <>{emptyState}</>
  }

  return (
    <div className="scroll-x">
      <table className="tbl">
        {caption ? <caption className="sr-only">{caption}</caption> : null}
        <Head definitions={definitions} />
        <tbody>
          {flows.map((flow) => (
            <Row
              key={flow.seq}
              flow={flow}
              definitions={definitions}
              selected={selectedSeq === flow.seq}
              isNew={Boolean(animateNew) && flow.seq > threshold}
              onSelect={onSelect}
            />
          ))}
        </tbody>
      </table>
    </div>
  )
}

/*
 * Rows are memoised because the live table re-renders on every stream
 * flush. Flow records are immutable once scored, so an existing row's
 * output cannot change — only newly inserted rows need to render.
 */
const Row = memo(function Row({
  flow,
  definitions,
  selected,
  isNew,
  onSelect,
}: {
  flow: FlowRecord
  definitions: Column[]
  selected: boolean
  isNew: boolean
  onSelect?: (flow: FlowRecord) => void
}) {
  const interactive = Boolean(onSelect)

  return (
    <tr
      aria-selected={selected}
      tabIndex={interactive ? 0 : undefined}
      role={interactive ? 'button' : undefined}
      onClick={interactive ? () => onSelect?.(flow) : undefined}
      onKeyDown={
        interactive
          ? (event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                onSelect?.(flow)
              }
            }
          : undefined
      }
      className={clsx(
        interactive && 'cursor-pointer',
        isNew && 'animate-row-in motion-reduce:animate-none',
      )}
    >
      {definitions.map((column) => (
        <td key={column.id} className={clsx(column.numeric && 'num')}>
          {column.render(flow)}
        </td>
      ))}
    </tr>
  )
})

function Head({ definitions }: { definitions: Column[] }) {
  return (
    <thead>
      <tr>
        {definitions.map((column) => (
          <th
            key={column.id}
            scope="col"
            style={column.width ? { width: column.width } : undefined}
            className={clsx(column.numeric && 'text-right')}
          >
            {column.header}
          </th>
        ))}
      </tr>
    </thead>
  )
}
