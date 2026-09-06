import { ModelPipeline } from '@/components/ModelPipeline'
import {
  Section,
  SectionHeader,
  SourceNote,
  StatList,
  StatRow,
} from '@/components/Section'
import { ErrorState, SkeletonLine } from '@/components/States'
import { usePolling } from '@/hooks/usePolling'
import { api } from '@/lib/api'
import { formatInt } from '@/lib/format'

/*
 * Models.
 *
 * How detection actually works and at what operating point. Thresholds and
 * weights are read from the running API, and the feature inventory from the
 * artifacts the training run wrote — nothing here is transcribed by hand.
 */

export function Models() {
  const info = usePolling(() => api.modelInfo(), null, [])
  const features = usePolling(() => api.features(), null, [])

  if (info.error && !info.data) {
    return <ErrorState error={info.error} onRetry={info.reload} />
  }

  const model = info.data
  const inventory = features.data

  return (
    <>
      <Section first>
        <SectionHeader
          title="Detection pipeline"
          description="Gated hybrid over unidirectional IP flows"
        />

        {model ? (
          <ModelPipeline
            xgbFeatures={model.xgboost.features}
            xgbThreshold={model.xgboost.threshold}
            aeFeatures={model.autoencoder.features}
            aeThreshold={model.autoencoder.threshold}
            xgbWeight={model.hybrid.xgboost_weight}
            aeWeight={model.hybrid.autoencoder_weight}
          />
        ) : (
          <div className="space-y-2 py-6">
            <SkeletonLine className="w-72" />
            <SkeletonLine className="w-96" />
            <SkeletonLine className="w-64" />
          </div>
        )}

        <p className="mt-4 max-w-2xl text-xs text-text-2">
          XGBoost is the primary classifier. The autoencoder cannot raise an
          alert on its own: it escalates a flow only when XGBoost has already
          scored it in the uncertain band between 0.05 and the decision
          threshold. The hybrid score sets confidence and risk, not the
          verdict.
        </p>
      </Section>

      <Section>
        <SectionHeader
          title="Operating point"
          description="Read from the running API"
        />

        <div className="grid gap-4 sm:grid-cols-3">
          <div className="panel p-4">
            <h3 className="text-xs font-medium text-text">XGBoost</h3>
            <StatList>
              <StatRow label="Threshold">
                {model?.xgboost.threshold ?? '—'}
              </StatRow>
              <StatRow label="Weight">{model?.xgboost.weight ?? '—'}</StatRow>
              <StatRow label="Inputs">
                {model ? formatInt(model.xgboost.features) : '—'}
              </StatRow>
              <StatRow label="Role" mono={false}>
                Primary
              </StatRow>
            </StatList>
          </div>

          <div className="panel p-4">
            <h3 className="text-xs font-medium text-text">Autoencoder</h3>
            <StatList>
              <StatRow label="Threshold">
                {model?.autoencoder.threshold ?? '—'}
              </StatRow>
              <StatRow label="Weight">
                {model?.autoencoder.weight ?? '—'}
              </StatRow>
              <StatRow label="Inputs">
                {model ? formatInt(model.autoencoder.features) : '—'}
              </StatRow>
              <StatRow label="Role" mono={false}>
                Anomaly support
              </StatRow>
            </StatList>
          </div>

          <div className="panel p-4">
            <h3 className="text-xs font-medium text-text">Hybrid</h3>
            <StatList>
              <StatRow label="Mode">{model?.hybrid.mode ?? '—'}</StatRow>
              <StatRow label="XGBoost weight">
                {model?.hybrid.xgboost_weight ?? '—'}
              </StatRow>
              <StatRow label="AE weight">
                {model?.hybrid.autoencoder_weight ?? '—'}
              </StatRow>
              <StatRow label="Status">{model?.status ?? '—'}</StatRow>
            </StatList>
          </div>
        </div>

        <p className="mt-3 text-2xs text-text-3">
          The autoencoder threshold is the 99th percentile of reconstruction
          error over benign training traffic. Reconstruction error is
          unbounded and is not a probability.
        </p>
      </Section>

      <Section>
        <SectionHeader
          title="Risk bands"
          description="How a verdict becomes a severity"
        />

        <div className="scroll-x max-w-2xl">
          <table className="tbl">
            <thead>
              <tr>
                <th scope="col">Verdict</th>
                <th scope="col">Condition</th>
                <th scope="col">Risk</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Malicious</td>
                <td className="font-mono">confidence ≥ 0.85</td>
                <td className="font-medium text-danger">Critical</td>
              </tr>
              <tr>
                <td>Malicious</td>
                <td className="font-mono">confidence ≥ 0.65</td>
                <td className="text-warn">High</td>
              </tr>
              <tr>
                <td>Malicious</td>
                <td className="font-mono">otherwise</td>
                <td className="text-text-2">Medium</td>
              </tr>
              <tr>
                <td>Benign</td>
                <td className="font-mono">hybrid ≥ 0.10</td>
                <td className="text-text-2">Low</td>
              </tr>
              <tr>
                <td>Benign</td>
                <td className="font-mono">otherwise</td>
                <td className="text-text-2">Safe</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section>
        <SectionHeader
          title="Model inputs"
          description="Feature schema read from the trained artifacts"
        />

        {features.error && !inventory ? (
          <ErrorState
            error={features.error}
            onRetry={features.reload}
            compact
          />
        ) : (
          <div className="grid gap-8 lg:grid-cols-2">
            <div>
              <h3 className="mb-2.5 text-xs font-medium text-text">
                XGBoost — {inventory?.xgboost.features.length ?? 0} numeric
                features
              </h3>
              <ul className="mono grid grid-cols-2 gap-x-4 gap-y-1 text-text-2">
                {(inventory?.xgboost.features ?? []).map((name, index) => (
                  <li key={name} className="flex gap-2">
                    <span className="w-5 shrink-0 text-right text-text-3">
                      {index + 1}
                    </span>
                    <span className="truncate">{name}</span>
                  </li>
                ))}
              </ul>
              {inventory?.xgboost.available ? (
                <SourceNote file={inventory.xgboost.file} />
              ) : null}
            </div>

            <div>
              <h3 className="mb-2.5 text-xs font-medium text-text">
                Autoencoder —{' '}
                {inventory?.autoencoder.input_dimension ?? 0} encoded
                dimensions
              </h3>

              <div className="scroll-x">
                <table className="tbl">
                  <thead>
                    <tr>
                      <th scope="col">Block</th>
                      <th scope="col" className="text-right">
                        Columns
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Numeric features</td>
                      <td className="num">
                        {inventory?.autoencoder.numeric_features.length ?? 0}
                      </td>
                    </tr>
                    {(inventory?.autoencoder.categorical_breakdown ?? []).map(
                      (item) => (
                        <tr key={item.feature}>
                          <td>
                            <span className="font-mono">{item.feature}</span>{' '}
                            <span className="text-text-3">one-hot</span>
                          </td>
                          <td className="num">{item.columns}</td>
                        </tr>
                      ),
                    )}
                    <tr>
                      <td className="font-medium">Total</td>
                      <td className="num font-medium">
                        {inventory?.autoencoder.input_dimension ?? 0}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <p className="mt-2.5 text-2xs text-text-3">
                A categorical value the training data barely contained is
                resolved to an all-zero block rather than a one-hot column.
                Without that, a rare category&rsquo;s tiny scaler variance
                turns a live value into hundreds of standard deviations and
                dominates the reconstruction error.
              </p>

              {inventory?.autoencoder.available ? (
                <SourceNote file={inventory.autoencoder.file} />
              ) : null}
            </div>
          </div>
        )}
      </Section>
    </>
  )
}
