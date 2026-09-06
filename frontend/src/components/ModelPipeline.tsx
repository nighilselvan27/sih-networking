/*
 * The detection pipeline, drawn from the real decision logic.
 *
 * Every threshold and weight shown is passed in from /model-info, so the
 * diagram cannot drift from the deployed configuration. The gate is
 * inference.py:858-874 stated in full, because "gated hybrid" on its own
 * tells an operator nothing about when the autoencoder can actually
 * escalate a flow.
 */

function Box({
  x,
  y,
  width,
  height,
  title,
  lines,
  emphasis = false,
  titleAnchor,
}: {
  x: number
  y: number
  width: number
  height: number
  title: string
  lines?: string[]
  emphasis?: boolean
  /** 'top' for boxes whose body is drawn by the caller, e.g. the gate. */
  titleAnchor?: 'center' | 'top'
}) {
  const anchor = titleAnchor ?? (lines?.length ? 'top' : 'center')

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={6}
        className={
          emphasis
            ? 'fill-surface-3 stroke-border-strong'
            : 'fill-surface stroke-border'
        }
        strokeWidth={1}
      />
      <text
        x={x + width / 2}
        y={y + (anchor === 'top' ? 21 : height / 2 + 4)}
        textAnchor="middle"
        className="fill-text"
        style={{ fontSize: 12, fontWeight: 500 }}
      >
        {title}
      </text>
      {lines?.map((line, index) => (
        <text
          key={line}
          x={x + width / 2}
          y={y + 38 + index * 14}
          textAnchor="middle"
          className="fill-text-2"
          style={{ fontSize: 11, fontFamily: 'ui-monospace, monospace' }}
        >
          {line}
        </text>
      ))}
    </g>
  )
}

function Arrow({
  d,
  label,
}: {
  d: string
  label?: { x: number; y: number; text: string }
}) {
  return (
    <>
      <path
        d={d}
        className="stroke-border-strong"
        strokeWidth={1}
        fill="none"
        markerEnd="url(#pipeline-arrow)"
      />
      {label ? (
        <text
          x={label.x}
          y={label.y}
          textAnchor="middle"
          className="fill-text-3"
          style={{ fontSize: 10 }}
        >
          {label.text}
        </text>
      ) : null}
    </>
  )
}

export function ModelPipeline({
  xgbFeatures,
  xgbThreshold,
  aeFeatures,
  aeThreshold,
  xgbWeight,
  aeWeight,
}: {
  xgbFeatures: number
  xgbThreshold: number
  aeFeatures: number
  aeThreshold: number
  xgbWeight: number
  aeWeight: number
}) {
  return (
    <div className="scroll-x">
      <svg
        viewBox="0 0 560 460"
        className="h-auto w-full min-w-[520px] max-w-[560px]"
        role="img"
        aria-label={
          `Detection pipeline: a unidirectional flow is reduced to ` +
          `${xgbFeatures} numeric features, scored by XGBoost against a ` +
          `threshold of ${xgbThreshold} and by an autoencoder against a ` +
          `reconstruction error threshold of ${aeThreshold}, then combined ` +
          `by a gate that lets the autoencoder escalate only when the ` +
          `XGBoost score is uncertain.`
        }
      >
        <defs>
          <marker
            id="pipeline-arrow"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M0 1 L7 4 L0 7 z" className="fill-border-strong" />
          </marker>
        </defs>

        <Box
          x={180}
          y={0}
          width={200}
          height={34}
          title="Unidirectional flow"
        />
        <Arrow d="M280 34 L280 58" />

        <Box
          x={150}
          y={58}
          width={260}
          height={52}
          title="Feature extraction"
          lines={[`${xgbFeatures} numeric features`]}
        />

        {/* Split to the two models */}
        <Arrow d="M280 110 L280 128 M280 128 L120 128 M120 128 L120 152" />
        <Arrow d="M280 128 L440 128 M440 128 L440 152" />

        <Box
          x={20}
          y={152}
          width={200}
          height={72}
          title="XGBoost"
          lines={[
            `threshold ${xgbThreshold}`,
            `${xgbFeatures} inputs`,
          ]}
        />
        <Box
          x={340}
          y={152}
          width={200}
          height={72}
          title="Autoencoder"
          lines={[
            `threshold ${aeThreshold}`,
            `${aeFeatures} encoded inputs`,
          ]}
        />

        {/* Converge on the gate */}
        <Arrow
          d="M120 224 L120 254 M120 254 L280 254 M280 254 L280 276"
          label={{ x: 170, y: 246, text: 'probability' }}
        />
        <Arrow
          d="M440 224 L440 254 M440 254 L280 254"
          label={{ x: 392, y: 246, text: 'recon. error' }}
        />

        <Box
          x={70}
          y={276}
          width={420}
          height={100}
          title="Gate"
          titleAnchor="top"
          emphasis
        />
        {[
          `if xgboost ≥ ${xgbThreshold} → malicious`,
          `else if ae ≥ ${aeThreshold} and xgboost ≥ 0.05 → malicious`,
          'else → benign',
        ].map((line, index) => (
          <text
            key={line}
            x={92}
            y={322 + index * 16}
            className="fill-text-2"
            style={{ fontSize: 11, fontFamily: 'ui-monospace, monospace' }}
          >
            {line}
          </text>
        ))}

        <Arrow d="M280 376 L280 400" />

        <Box
          x={130}
          y={400}
          width={300}
          height={56}
          title="Verdict, risk and confidence"
          lines={[
            `hybrid = ${xgbWeight}·xgb + ${aeWeight}·ae_norm`,
          ]}
        />
      </svg>
    </div>
  )
}
