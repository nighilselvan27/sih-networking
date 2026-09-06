import { Check, ShieldCheck } from 'lucide-react'
import { useState } from 'react'
import { FilterBar, SearchField, Select } from '@/components/Controls'
import { Counter } from '@/components/Metric'
import { Drawer } from '@/components/Drawer'
import { FlowDetail } from '@/components/FlowDetail'
import { FlowTable } from '@/components/FlowTable'
import { Section, SectionHeader } from '@/components/Section'
import { EmptyState, ErrorState, SkeletonRows } from '@/components/States'
import { usePolling } from '@/hooks/usePolling'
import { api } from '@/lib/api'
import type { FlowRecord } from '@/lib/types'

/*
 * Alerts.
 *
 * An alert is a flow the pipeline scored as malicious. There is no second
 * alerting rule layered on top, and severity is the model's own risk band —
 * nothing is re-scored here.
 */

const STATUS = [
  { value: '', label: 'All statuses' },
  { value: 'open', label: 'Open' },
  { value: 'acknowledged', label: 'Acknowledged' },
]

export function Alerts() {
  const [risk, setRisk] = useState<string>('')
  const [status, setStatus] = useState<string>('')
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<FlowRecord | null>(null)
  const [acknowledging, setAcknowledging] = useState(false)

  const acknowledged =
    status === 'acknowledged' ? true : status === 'open' ? false : undefined

  const alerts = usePolling(
    () =>
      api.alerts({
        limit: 200,
        risk: risk || undefined,
        acknowledged,
        q: search || undefined,
      }),
    4000,
    [risk, status, search],
  )

  const counts = alerts.data?.counts

  const acknowledge = async (flow: FlowRecord) => {
    setAcknowledging(true)

    try {
      const updated = await api.acknowledge(flow.seq)
      setSelected(updated)
      alerts.reload()
    } catch {
      /* the row stays open; the API refused or the record was evicted */
    } finally {
      setAcknowledging(false)
    }
  }

  const toggleRisk = (next: string) => setRisk(risk === next ? '' : next)

  return (
    <>
      <div
        className="grid grid-cols-2 divide-x divide-y divide-border border-b
                   border-border sm:grid-cols-4 sm:divide-y-0"
      >
        <Counter
          label="Critical"
          value={counts?.critical ?? 0}
          tone="danger"
          active={risk === 'CRITICAL'}
          onClick={() => toggleRisk('CRITICAL')}
        />
        <Counter
          label="High"
          value={counts?.high ?? 0}
          tone="warn"
          active={risk === 'HIGH'}
          onClick={() => toggleRisk('HIGH')}
        />
        <Counter
          label="Medium"
          value={counts?.medium ?? 0}
          active={risk === 'MEDIUM'}
          onClick={() => toggleRisk('MEDIUM')}
        />
        <Counter
          label="Acknowledged"
          value={counts?.acknowledged ?? 0}
          active={status === 'acknowledged'}
          onClick={() =>
            setStatus(status === 'acknowledged' ? '' : 'acknowledged')
          }
        />
      </div>

      <Section first className="pt-6">
        <SectionHeader
          title="Alerts"
          description="Flows scored as malicious by the gated hybrid detector"
        />

        <FilterBar
          right={
            <span className="text-2xs text-text-3">
              {alerts.data
                ? `${alerts.data.alerts.length} shown`
                : 'Loading'}
            </span>
          }
        >
          <SearchField
            value={search}
            onChange={setSearch}
            placeholder="Search address or port"
            className="w-full sm:w-72"
          />
          <Select
            label="Severity"
            value={risk}
            onChange={setRisk}
            options={[
              { value: '', label: 'All severities' },
              { value: 'CRITICAL', label: 'Critical' },
              { value: 'HIGH', label: 'High' },
              { value: 'MEDIUM', label: 'Medium' },
            ]}
          />
          <Select
            label="Status"
            value={status}
            onChange={setStatus}
            options={STATUS}
          />
        </FilterBar>

        {alerts.error && !alerts.data ? (
          <ErrorState error={alerts.error} onRetry={alerts.reload} />
        ) : (
          <FlowTable
            flows={alerts.data?.alerts ?? []}
            columns={[
              'risk',
              'time',
              'source',
              'destination',
              'protocol',
              'verdict_plain',
              'xgboost',
              'confidence',
              'status',
            ]}
            onSelect={setSelected}
            selectedSeq={selected?.seq ?? null}
            animateNew
            caption="Security alerts"
            loading={
              alerts.loading ? <SkeletonRows rows={10} columns={9} /> : undefined
            }
            emptyState={
              <EmptyState
                icon={ShieldCheck}
                title={
                  risk || status || search
                    ? 'No alerts match these filters'
                    : 'No active alerts'
                }
                description={
                  risk || status || search
                    ? 'Clear the filters to see every alert in the buffer.'
                    : 'Traffic is being monitored. Detected anomalies will appear here.'
                }
              />
            }
          />
        )}

        <p className="mt-3 text-2xs text-text-3">
          Alerts are held in the API&rsquo;s in-memory buffer and are cleared
          when it restarts. Acknowledgement is not persisted to disk.
        </p>
      </Section>

      <Drawer
        open={selected !== null}
        onClose={() => setSelected(null)}
        title={selected ? `${selected.src_ip} → ${selected.dst_ip}` : 'Alert'}
        subtitle={
          selected
            ? `${selected.protocol.toUpperCase()} · flow #${selected.seq}`
            : undefined
        }
        footer={
          selected && !selected.acknowledged ? (
            <button
              type="button"
              className="btn"
              onClick={() => acknowledge(selected)}
              disabled={acknowledging}
            >
              <Check className="h-3.5 w-3.5" aria-hidden />
              {acknowledging ? 'Acknowledging…' : 'Acknowledge'}
            </button>
          ) : selected ? (
            <span className="text-xs text-text-2">
              Acknowledged in this API session.
            </span>
          ) : null
        }
      >
        {selected ? <FlowDetail flow={selected} /> : null}
      </Drawer>
    </>
  )
}
