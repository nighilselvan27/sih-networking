import clsx from 'clsx'
import { Check, Info } from 'lucide-react'
import { Field, FieldGrid } from './Section'
import { RiskLabel, ScoreBar, StatusDot, Verdict } from './Indicators'
import {
  formatBytes,
  formatDateTime,
  formatDuration,
  formatInt,
  formatPercent,
  formatRate,
  formatScore,
} from '@/lib/format'
import type { FlowRecord } from '@/lib/types'

/*
 * Flow investigation view.
 *
 * The rule this component follows: show the evidence the detector actually
 * produced, and show the arithmetic that turned it into a verdict. Nothing
 * is inferred, ranked or attributed beyond what backend/inference.py
 * returns.
 *
 * In particular there is no attack family here, because the model does not
 * emit one — it is a binary classifier. Saying "UDP flood" would be a
 * guess dressed up as a finding.
 */

/** Uncertainty floor from the gate in inference.py:864. */
const UNCERTAINTY_LOW = 0.05

function Group({
  title,
  children,
  note,
}: {
  title: string
  children: React.ReactNode
  note?: string
}) {
  return (
    <section className="mt-6 border-t border-border pt-4 first:mt-0 first:border-t-0 first:pt-0">
      <h3 className="mb-3 text-xs font-medium text-text">{title}</h3>
      {children}
      {note ? <p className="mt-2.5 text-2xs text-text-3">{note}</p> : null}
    </section>
  )
}

/** One model's score against its own threshold, with the comparison shown. */
function ModelSignal({
  name,
  value,
  threshold,
  triggered,
  /** Position on the 0..1 bar. AE error is unbounded, so it is normalised. */
  barValue,
  barThreshold,
  unit,
  description,
}: {
  name: string
  value: string
  threshold: string
  triggered: boolean
  barValue: number
  barThreshold?: number
  unit?: string
  description: string
}) {
  return (
    <div className="py-3 first:pt-0 last:pb-0">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-xs font-medium text-text">{name}</span>
        <span className="flex items-center gap-1.5">
          <StatusDot tone={triggered ? 'danger' : 'ok'} />
          <span
            className={clsx(
              'text-2xs',
              triggered ? 'text-danger' : 'text-text-2',
            )}
          >
            {triggered ? 'Above threshold' : 'Below threshold'}
          </span>
        </span>
      </div>

      <div className="mt-1.5">
        <ScoreBar
          value={barValue}
          threshold={barThreshold}
          tone={triggered ? 'danger' : 'muted'}
        />
      </div>

      <div className="mt-1.5 flex items-baseline justify-between gap-3">
        <span className="mono text-text">
          {value}
          {unit ? <span className="text-text-3"> {unit}</span> : null}
        </span>
        <span className="mono text-text-3">threshold {threshold}</span>
      </div>

      <p className="mt-1 text-2xs text-text-3">{description}</p>
    </div>
  )
}

export function FlowDetail({ flow }: { flow: FlowRecord }) {
  const xgbThreshold =
    flow.evidence.xgboost_threshold ?? flow.details.xgboost_threshold ?? 0.2

  const aeThreshold =
    flow.evidence.autoencoder_threshold ??
    flow.details.autoencoder_threshold ??
    0.25541964

  const xgbWeight =
    flow.evidence.xgboost_weight ?? flow.details.xgboost_weight ?? 0.9

  const aeWeight =
    flow.evidence.autoencoder_weight ?? flow.details.autoencoder_weight ?? 0.1

  const malicious = flow.prediction === 1

  // Which branch of the gate produced this verdict.
  const branch = flow.xgboost_malicious
    ? 'xgboost'
    : flow.autoencoder_anomalous && flow.xgboost_score >= UNCERTAINTY_LOW
      ? 'autoencoder'
      : 'none'

  const supporting = Object.entries(flow.supporting_features ?? {})
  const evidenceFeatures = flow.details.evidence_features ?? []
  const resolution = flow.details.categorical_resolution ?? []

  return (
    <div>
      {/* --- Verdict ------------------------------------------------ */}
      <Group title="Detection">
        <FieldGrid columns={3}>
          <Field label="Verdict">
            <Verdict flow={flow} showRisk={false} />
          </Field>
          <Field label="Threat class">
            <span className="font-mono">{flow.threat_class || '—'}</span>
          </Field>
          <Field label="Risk">
            <RiskLabel risk={flow.risk} />
          </Field>
          <Field label="Confidence" mono>
            {formatPercent(flow.confidence, 1)}
          </Field>
          <Field label="Hybrid score" mono>
            {formatScore(flow.hybrid_score)}
          </Field>
          <Field label="Observed" mono>
            {formatDateTime(flow.timestamp || flow.received_at)}
          </Field>
        </FieldGrid>
      </Group>

      {/* --- Why -------------------------------------------------- */}
      <Group title="Why was this detected?">
        <div className="divide-y divide-border">
          <ModelSignal
            name="XGBoost"
            value={formatScore(flow.xgboost_score)}
            threshold={formatScore(xgbThreshold, 2)}
            triggered={flow.xgboost_malicious}
            barValue={flow.xgboost_score}
            barThreshold={xgbThreshold}
            description="Attack probability for the positive class."
          />
          <ModelSignal
            name="Autoencoder"
            value={formatScore(flow.autoencoder_score, 8)}
            threshold={formatScore(aeThreshold, 8)}
            triggered={flow.autoencoder_anomalous}
            barValue={flow.autoencoder_normalized}
            barThreshold={1}
            unit="MSE"
            description={
              'Mean squared reconstruction error over the ' +
              `${flow.details.autoencoder_encoded_features ?? 257}-dimension ` +
              'encoded input. Not a probability; the bar shows the error ' +
              'relative to the threshold.'
            }
          />
        </div>

        {/* The gate, written out. */}
        <div className="mt-4 rounded border border-border bg-surface-2 p-3">
          <div className="label-field mb-2">Gate</div>

          <ol className="space-y-1.5 text-2xs">
            <li className="flex items-start gap-2">
              <StatusDot
                tone={branch === 'xgboost' ? 'danger' : 'muted'}
                className="mt-1"
              />
              <span className="font-mono text-text-2">
                xgboost {formatScore(flow.xgboost_score)}{' '}
                {flow.xgboost_malicious ? '≥' : '<'}{' '}
                {formatScore(xgbThreshold, 2)}
                <span className="ml-2 font-sans">
                  {branch === 'xgboost'
                    ? '→ malicious'
                    : 'not met'}
                </span>
              </span>
            </li>

            <li className="flex items-start gap-2">
              <StatusDot
                tone={branch === 'autoencoder' ? 'danger' : 'muted'}
                className="mt-1"
              />
              <span className="font-mono text-text-2">
                autoencoder {flow.autoencoder_anomalous ? '≥' : '<'}{' '}
                {formatScore(aeThreshold, 8)} and xgboost{' '}
                {flow.xgboost_score >= UNCERTAINTY_LOW ? '≥' : '<'}{' '}
                {UNCERTAINTY_LOW.toFixed(2)}
                <span className="ml-2 font-sans">
                  {branch === 'autoencoder' ? '→ malicious' : 'not met'}
                </span>
              </span>
            </li>

            <li className="flex items-start gap-2">
              <StatusDot
                tone={branch === 'none' ? 'ok' : 'muted'}
                className="mt-1"
              />
              <span className="font-mono text-text-2">
                otherwise
                <span className="ml-2 font-sans">
                  {branch === 'none' ? '→ benign' : 'not reached'}
                </span>
              </span>
            </li>
          </ol>

          <div className="mt-3 border-t border-border pt-2.5">
            <div className="mono text-text-2">
              hybrid = {xgbWeight} × {formatScore(flow.xgboost_score)} +{' '}
              {aeWeight} × {formatScore(flow.autoencoder_normalized)} ={' '}
              <span className="text-text">
                {formatScore(flow.hybrid_score)}
              </span>
            </div>
            <p className="mt-1 text-2xs text-text-3">
              The hybrid score sets confidence and risk. It does not decide
              the verdict — the gate above does.
            </p>
          </div>
        </div>

        {flow.explanation ? (
          <p className="mt-3 text-xs text-text-2">{flow.explanation}</p>
        ) : null}

        <p className="mt-3 flex items-start gap-1.5 text-2xs text-text-3">
          <Info className="mt-0.5 h-3 w-3 shrink-0" aria-hidden />
          <span>
            The detector is a binary classifier. It reports malicious or
            benign with a risk band, and does not identify an attack family.
          </span>
        </p>
      </Group>

      {/* --- Flow --------------------------------------------------- */}
      <Group title="Flow">
        <FieldGrid columns={2}>
          <Field label="Source" mono>
            {flow.src_ip}
            <span className="text-text-3">:{flow.src_port}</span>
          </Field>
          <Field label="Destination" mono>
            {flow.dst_ip}
            <span className="text-text-3">:{flow.dst_port}</span>
          </Field>
          <Field label="Protocol" mono>
            {flow.protocol.toUpperCase()}
          </Field>
          <Field label="Direction" mono>
            {flow.direction || '—'}
          </Field>
          <Field label="State" mono>
            {flow.state || '—'}
          </Field>
          <Field label="Flow ID" mono className="col-span-2">
            {flow.flow_id || '—'}
          </Field>
        </FieldGrid>
      </Group>

      <Group
        title="Volume and timing"
        note={
          'Values submitted with this flow by the capture layer. ' +
          'Unidirectional capture observes one direction only, so ' +
          'destination bytes are zero by construction.'
        }
      >
        <FieldGrid columns={3}>
          <Field label="Packets" mono>
            {formatInt(flow.packets)}
          </Field>
          <Field label="Bytes" mono>
            {formatBytes(flow.bytes)}
          </Field>
          <Field label="Duration" mono>
            {formatDuration(flow.duration)}
          </Field>
          <Field label="Packets/sec" mono>
            {formatRate(flow.packets_per_second)}
          </Field>
          <Field label="Bytes/sec" mono>
            {formatRate(flow.bytes_per_second)}
          </Field>
          <Field label="Avg packet" mono>
            {formatRate(flow.avg_packet_size)} B
          </Field>
        </FieldGrid>
      </Group>

      {/* --- Model inputs -------------------------------------------- */}
      {supporting.length > 0 ? (
        <Group
          title="Model inputs"
          note={
            evidenceFeatures.length > 0
              ? 'These are this flow’s own values for the features the ' +
                'trained booster weights most by gain. They are the model’s ' +
                'inputs, not a per-feature attribution for this decision.'
              : undefined
          }
        >
          <table className="tbl">
            <thead>
              <tr>
                <th scope="col">Feature</th>
                <th scope="col" className="text-right">
                  Value
                </th>
              </tr>
            </thead>
            <tbody>
              {supporting.map(([name, value]) => (
                <tr key={name}>
                  <td className="font-mono">{name}</td>
                  <td className="num">
                    {typeof value === 'number'
                      ? Number.isInteger(value)
                        ? formatInt(value)
                        : formatRate(value)
                      : String(value)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Group>
      ) : null}

      {resolution.length > 0 ? (
        <Group
          title="Categorical encoding"
          note={
            'How this flow’s categorical values mapped onto the ' +
            'autoencoder’s one-hot columns.'
          }
        >
          <ul className="space-y-1">
            {resolution.map((line) => (
              <li key={line} className="mono text-text-2">
                {line}
              </li>
            ))}
          </ul>
        </Group>
      ) : null}

      <Group title="Record">
        <FieldGrid columns={3}>
          <Field label="Sequence" mono>
            #{flow.seq}
          </Field>
          <Field label="Timestamp source" mono>
            {flow.details.timestamp_source ?? '—'}
          </Field>
          <Field label="Encoded inputs" mono>
            {flow.details.autoencoder_encoded_features ?? '—'}
          </Field>
          {flow.details.autoencoder_non_zero_features !== undefined ? (
            <Field label="Non-zero inputs" mono>
              {flow.details.autoencoder_non_zero_features}
            </Field>
          ) : null}
          <Field label="Acknowledged">
            {flow.acknowledged ? (
              <span className="inline-flex items-center gap-1 text-ok">
                <Check className="h-3 w-3" aria-hidden />
                Yes
              </span>
            ) : (
              <span className="text-text-2">No</span>
            )}
          </Field>
        </FieldGrid>
      </Group>

      {!malicious ? null : (
        <p className="mt-6 text-2xs text-text-3">
          Acknowledgement is held in the API&rsquo;s memory and is cleared
          when it restarts.
        </p>
      )}
    </div>
  )
}
