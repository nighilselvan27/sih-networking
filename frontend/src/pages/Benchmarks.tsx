import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { BenchmarkTable } from '@/components/BenchmarkTable'
import { Field, FieldGrid, Section, SectionHeader, SourceNote } from '@/components/Section'
import { ErrorState, SkeletonLine } from '@/components/States'
import { useChartColors } from '@/hooks/useChartColors'
import { usePolling } from '@/hooks/usePolling'
import { api } from '@/lib/api'
import { formatInt } from '@/lib/format'
import type { CsvSection, StreamingSummary } from '@/lib/types'

/*
 * Benchmarks.
 *
 * Recorded evaluation results, read from the artifacts in models/ and
 * outputs/. Nothing on this page is measured live and nothing is
 * recomputed. Where a result is weak it is shown as it is — the
 * scenario-level breakdown is the honest part of this page.
 */

const DEPLOYED = 'Hybrid XGBoost + Autoencoder'

export function Benchmarks() {
  const benchmarks = usePolling(() => api.benchmarks(), null, [])

  if (benchmarks.error && !benchmarks.data) {
    return <ErrorState error={benchmarks.error} onRetry={benchmarks.reload} />
  }

  if (!benchmarks.data) {
    return (
      <div className="space-y-3 py-8">
        <SkeletonLine className="w-64" />
        <SkeletonLine className="w-full max-w-2xl" />
        <SkeletonLine className="w-full max-w-xl" />
      </div>
    )
  }

  const { sections, missing } = benchmarks.data
  const summary = sections.streaming_summary.data as StreamingSummary | null

  // Read from the recorded configuration rather than asserted in prose, so
  // the page cannot claim an operating point the artifacts contradict.
  const deployedThreshold = (
    sections.hybrid_config.data as { xgboost_threshold?: number } | null
  )?.xgboost_threshold

  return (
    <>
      <Section first>
        <SectionHeader
          title="Deployed detector"
          description="Trained on CTU-13 scenarios 1–10, evaluated on the held-out scenarios 11–13"
        />

        <BenchmarkTable
          section={sections.hybrid}
          highlightRow={DEPLOYED}
          note="547,490 held-out flows"
        />

        <p className="mt-4 max-w-3xl text-xs text-text-2">
          The gated hybrid trades a fraction of recall for precision against
          XGBoost alone: it removes 22 false positives and adds 10 false
          negatives. That is the design intent — the autoencoder is a
          precision aid, not a second detector — but it also means the
          headline numbers are XGBoost&rsquo;s.
        </p>
      </Section>

      <Section>
        <SectionHeader
          title="Per-scenario results"
          description="The same three models, split by held-out scenario"
        />

        <BenchmarkTable section={sections.hybrid_per_scenario} />

        <p className="mt-4 max-w-3xl text-xs text-text-2">
          Performance is not uniform across scenarios, and the aggregate
          hides that. Scenario 11 is detected almost perfectly; scenario 12
          is largely missed. The botnet behaviour in scenario 12 does not
          resemble the training scenarios in these features.
        </p>
      </Section>

      <Section>
        <SectionHeader
          title="Operating point"
          description="XGBoost decision threshold sweep over the held-out set"
        />

        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_260px]">
          <ThresholdChart section={sections.threshold_sweep} />

          <div>
            <h3 className="mb-3 text-xs font-medium text-text">
              Recorded selection
            </h3>
            {sections.threshold_recommendation.available ? (
              <FieldGrid columns={2}>
                {Object.entries(
                  sections.threshold_recommendation.data ?? {},
                ).map(([key, value]) => (
                  <Field key={key} label={key.replace(/_/g, ' ')} mono>
                    {typeof value === 'number'
                      ? value.toFixed(value < 1 && value > 0 ? 4 : 2)
                      : String(value)}
                  </Field>
                ))}
              </FieldGrid>
            ) : (
              <p className="text-xs text-text-2">
                No recommendation artifact present.
              </p>
            )}
            <SourceNote file={sections.threshold_recommendation.file} />
          </div>
        </div>

        <p className="mt-4 max-w-3xl text-xs text-text-2">
          The deployed threshold is{' '}
          <span className="font-mono">{deployedThreshold ?? '—'}</span>. It is
          not the F1 optimum in this sweep — a higher threshold scores
          marginally better — but it holds the false positive rate low while
          keeping the uncertainty band that the autoencoder gate depends on.
        </p>
      </Section>

      <Section>
        <SectionHeader
          title="Model comparison"
          description="Candidate classifiers, trained on scenarios 1–10 and tested on 11–13"
        />

        <BenchmarkTable
          section={sections.multiscenario}
          highlightRow="XGBoost"
        />

        <div className="mt-8">
          <SectionHeader
            title="Single-scenario comparison"
            description="Earlier evaluation with training and test data from the same scenario"
          />
          <BenchmarkTable
            section={sections.model_comparison}
            highlightRow="XGBoost"
            note="not directly comparable with the cross-scenario results above"
          />
        </div>
      </Section>

      <Section>
        <SectionHeader
          title="Autoencoder"
          description="Reconstruction error against the 99th-percentile benign threshold"
        />

        <BenchmarkTable
          section={sections.autoencoder}
          columns={[
            'Model',
            'Scenarios',
            'Rows',
            'Threshold',
            'Accuracy',
            'Precision',
            'Recall',
            'F1',
            'ROC_AUC',
            'FPR',
            'TN',
            'FP',
            'FN',
            'TP',
          ]}
        />

        <div className="mt-6 grid gap-8 lg:grid-cols-2">
          <div>
            <h3 className="mb-2.5 text-xs font-medium text-text">
              Per scenario
            </h3>
            <BenchmarkTable
              section={sections.autoencoder_per_scenario}
              columns={[
                'Scenario',
                'Rows',
                'Precision',
                'Recall',
                'F1',
                'ROC_AUC',
                'Anomalies',
                'Mean_Error',
                'Median_Error',
                'Max_Error',
              ]}
            />
          </div>

          <div>
            <h3 className="mb-2.5 text-xs font-medium text-text">
              Training history
            </h3>
            <TrainingChart section={sections.autoencoder_training_history} />
          </div>
        </div>

        <p className="mt-4 max-w-3xl text-xs text-text-2">
          On its own the autoencoder is a weak detector across the held-out
          set: it separates scenario 11 well and the other two barely at all.
          That is why it is gated behind XGBoost rather than allowed to raise
          alerts independently.
        </p>
      </Section>

      <Section>
        <SectionHeader
          title="Synthetic attack detection"
          description="Detection rate per generated attack class, at the recorded threshold"
        />

        <BenchmarkTable
          section={sections.synthetic_per_attack}
          columns={[
            'AttackType',
            'Rows',
            'Detected',
            'Missed',
            'DetectionRate',
            'MeanProbability',
            'MedianProbability',
          ]}
        />

        <p className="mt-4 max-w-3xl text-xs text-text-2">
          These are synthetic flow records, not captured traffic. The model
          was trained on CTU-13 botnet behaviour rather than on volumetric
          floods, and the detection rates reflect that.
        </p>
      </Section>

      {summary ? (
        <Section>
          <SectionHeader
            title="Streaming replay"
            description="Chunked replay of the held-out scenarios through the detector"
          />

          <div className="grid gap-4 sm:grid-cols-4">
            <Field label="Flows replayed" mono>
              {formatInt(summary.total_rows)}
            </Field>
            <Field label="Benign" mono>
              {formatInt(summary.benign)}
            </Field>
            <Field label="Malicious" mono>
              {formatInt(summary.malicious)}
            </Field>
            <Field label="XGBoost alerts" mono>
              {formatInt(summary.xgboost_alerts)}
            </Field>
          </div>

          <div className="scroll-x mt-5">
            <table className="tbl">
              <thead>
                <tr>
                  <th scope="col">Scenario</th>
                  <th scope="col" className="text-right">
                    Rows
                  </th>
                  <th scope="col" className="text-right">
                    Benign
                  </th>
                  <th scope="col" className="text-right">
                    Suspicious
                  </th>
                  <th scope="col" className="text-right">
                    Malicious
                  </th>
                  <th scope="col" className="text-right">
                    XGBoost alerts
                  </th>
                  <th scope="col" className="text-right">
                    AE alerts
                  </th>
                </tr>
              </thead>
              <tbody>
                {summary.scenario_results.map((row) => (
                  <tr key={row.scenario}>
                    <td className="font-mono">{row.scenario}</td>
                    <td className="num">{formatInt(row.rows)}</td>
                    <td className="num">{formatInt(row.benign)}</td>
                    <td className="num">{formatInt(row.suspicious)}</td>
                    <td className="num">{formatInt(row.malicious)}</td>
                    <td className="num">{formatInt(row.xgboost_alerts)}</td>
                    <td className="num">{formatInt(row.autoencoder_alerts)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <SourceNote
            file={sections.streaming_summary.file}
            note={
              'the replay harness uses a three-way benign / suspicious / ' +
              'malicious label, which is not the binary vocabulary the live ' +
              'API returns'
            }
          />
        </Section>
      ) : null}

      {missing.length > 0 ? (
        <Section>
          <SectionHeader title="Missing artifacts" />
          <ul className="mono space-y-1 text-text-2">
            {missing.map((file) => (
              <li key={file}>{file}</li>
            ))}
          </ul>
        </Section>
      ) : null}
    </>
  )
}

/* --- Charts -------------------------------------------------------- */

function ThresholdChart({ section }: { section: CsvSection }) {
  const colors = useChartColors()

  if (!section.available) {
    return (
      <p className="text-xs text-text-2">
        Threshold sweep artifact is not present.
      </p>
    )
  }

  const data = section.rows.map((row) => ({
    threshold: Number(row.Threshold ?? 0),
    precision: Number(row.Precision ?? 0),
    recall: Number(row.Recall ?? 0),
    f1: Number(row.F1 ?? 0),
  }))

  return (
    <div>
      <div style={{ height: 240 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={colors.grid} vertical={false} />
            <XAxis
              dataKey="threshold"
              type="number"
              domain={[0, 1]}
              ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
              tick={{ fill: colors['text-3'], fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: colors.border }}
              height={20}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fill: colors['text-3'], fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={36}
              tickFormatter={(value: number) => value.toFixed(1)}
            />
            <Tooltip
              isAnimationActive={false}
              contentStyle={{
                background: colors.surface,
                border: `1px solid ${colors.border}`,
                borderRadius: 6,
                fontSize: 11,
              }}
              labelFormatter={(value) => `Threshold ${Number(value).toFixed(2)}`}
              formatter={(value: number, name: string) => [
                value.toFixed(4),
                name,
              ]}
            />
            <Line
              dataKey="precision"
              name="Precision"
              stroke={colors['series-1']}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              dataKey="recall"
              name="Recall"
              stroke={colors['series-3']}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              dataKey="f1"
              name="F1"
              stroke={colors['series-2']}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        {[
          { label: 'Precision', color: 'series-1' },
          { label: 'Recall', color: 'series-3' },
          { label: 'F1', color: 'series-2' },
        ].map((item) => (
          <li
            key={item.label}
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

      <SourceNote file={section.file} />
    </div>
  )
}

function TrainingChart({ section }: { section: CsvSection }) {
  const colors = useChartColors()

  if (!section.available) {
    return (
      <p className="text-xs text-text-2">
        Training history artifact is not present.
      </p>
    )
  }

  const data = section.rows.map((row) => ({
    epoch: Number(row.epoch ?? 0),
    loss: Number(row.loss ?? 0),
    val_loss: Number(row.val_loss ?? 0),
  }))

  return (
    <div>
      <div style={{ height: 200 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={colors.grid} vertical={false} />
            <XAxis
              dataKey="epoch"
              tick={{ fill: colors['text-3'], fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: colors.border }}
              height={20}
            />
            <YAxis
              tick={{ fill: colors['text-3'], fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={36}
              tickFormatter={(value: number) => value.toFixed(1)}
            />
            <Tooltip
              isAnimationActive={false}
              contentStyle={{
                background: colors.surface,
                border: `1px solid ${colors.border}`,
                borderRadius: 6,
                fontSize: 11,
              }}
              labelFormatter={(value) => `Epoch ${value}`}
              formatter={(value: number, name: string) => [
                value.toFixed(4),
                name === 'loss' ? 'Training loss' : 'Validation loss',
              ]}
            />
            <Line
              dataKey="loss"
              stroke={colors['series-1']}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              dataKey="val_loss"
              stroke={colors['series-2']}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        {[
          { label: 'Training loss', color: 'series-1' },
          { label: 'Validation loss', color: 'series-2' },
        ].map((item) => (
          <li
            key={item.label}
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

      <SourceNote file={section.file} />
    </div>
  )
}
