import { Network } from 'lucide-react'
import { useEffect, useState } from 'react'
import { FilterBar, SearchField, Select } from '@/components/Controls'
import { Drawer } from '@/components/Drawer'
import { FlowDetail } from '@/components/FlowDetail'
import { FlowTable } from '@/components/FlowTable'
import { Section, SectionHeader } from '@/components/Section'
import { EmptyState, ErrorState, SkeletonRows } from '@/components/States'
import { usePolling } from '@/hooks/usePolling'
import { api } from '@/lib/api'
import { formatInt } from '@/lib/format'
import type { FlowRecord } from '@/lib/types'

/*
 * Flows.
 *
 * The explorer over everything the API has scored and still holds. Live
 * updates are paused while the operator is paging back through history —
 * a list that reorders under the cursor is not an explorer.
 */

const PAGE_SIZE = 100

export function Flows() {
  const [search, setSearch] = useState('')
  const [protocol, setProtocol] = useState('')
  const [verdict, setVerdict] = useState('')
  const [risk, setRisk] = useState('')
  const [beforeSeq, setBeforeSeq] = useState<number | null>(null)
  const [selected, setSelected] = useState<FlowRecord | null>(null)

  // Changing a filter always returns to the newest page.
  useEffect(() => {
    setBeforeSeq(null)
  }, [search, protocol, verdict, risk])

  const paging = beforeSeq !== null

  const page = usePolling(
    () =>
      api.flows({
        limit: PAGE_SIZE,
        before_seq: beforeSeq ?? undefined,
        protocol: protocol || undefined,
        verdict: verdict || undefined,
        risk: risk || undefined,
        q: search || undefined,
      }),
    // Only the newest page refreshes on its own.
    paging ? null : 4000,
    [search, protocol, verdict, risk, beforeSeq],
  )

  const data = page.data

  return (
    <>
      <Section first>
        <SectionHeader
          title="Flow explorer"
          description="Every flow held in the API buffer, newest first"
          actions={
            data ? (
              <span className="text-2xs text-text-3">
                Buffer {formatInt(data.buffer_size)} /{' '}
                {formatInt(data.buffer_capacity)}
              </span>
            ) : null
          }
        />

        <FilterBar
          right={
            paging ? (
              <button
                type="button"
                className="btn"
                onClick={() => setBeforeSeq(null)}
              >
                Back to newest
              </button>
            ) : null
          }
        >
          <SearchField
            value={search}
            onChange={setSearch}
            placeholder="Search address, port, flow id or state"
            className="w-full sm:w-80"
          />
          <Select
            label="Protocol"
            value={protocol}
            onChange={setProtocol}
            options={[
              { value: '', label: 'All protocols' },
              { value: 'tcp', label: 'TCP' },
              { value: 'udp', label: 'UDP' },
            ]}
          />
          <Select
            label="Detection"
            value={verdict}
            onChange={setVerdict}
            options={[
              { value: '', label: 'All detections' },
              { value: 'malicious', label: 'Malicious' },
              { value: 'benign', label: 'Benign' },
            ]}
          />
          <Select
            label="Risk"
            value={risk}
            onChange={setRisk}
            options={[
              { value: '', label: 'All risk levels' },
              { value: 'CRITICAL', label: 'Critical' },
              { value: 'HIGH', label: 'High' },
              { value: 'MEDIUM', label: 'Medium' },
              { value: 'LOW', label: 'Low' },
              { value: 'SAFE', label: 'Safe' },
            ]}
          />
        </FilterBar>

        {page.error && !data ? (
          <ErrorState error={page.error} onRetry={page.reload} />
        ) : (
          <FlowTable
            flows={data?.flows ?? []}
            columns={[
              'flow',
              'time',
              'source',
              'destination',
              'protocol',
              'packets',
              'bytes',
              'duration',
              'verdict',
              'xgboost',
              'autoencoder',
            ]}
            onSelect={setSelected}
            selectedSeq={selected?.seq ?? null}
            animateNew={!paging}
            caption="Scored flows"
            loading={
              page.loading ? <SkeletonRows rows={12} columns={11} /> : undefined
            }
            emptyState={
              <EmptyState
                icon={Network}
                title={
                  search || protocol || verdict || risk
                    ? 'No flows match these filters'
                    : 'No flows scored yet'
                }
                description={
                  search || protocol || verdict || risk
                    ? 'Clear the filters to see the full buffer.'
                    : 'Flows appear here once the capture layer submits them for scoring.'
                }
              />
            }
          />
        )}

        {data ? (
          <div className="flex items-center justify-between gap-3 py-3">
            <span className="text-2xs text-text-3">
              {data.flows.length > 0
                ? `Showing ${formatInt(data.flows.length)} flows${
                    paging ? ' (paged)' : ''
                  }`
                : ''}
            </span>
            <button
              type="button"
              className="btn"
              disabled={!data.has_more || data.next_before_seq === null}
              onClick={() => setBeforeSeq(data.next_before_seq)}
            >
              Older flows
            </button>
          </div>
        ) : null}
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
