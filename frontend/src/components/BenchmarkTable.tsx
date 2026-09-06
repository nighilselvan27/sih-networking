import clsx from 'clsx'
import { SourceNote } from './Section'
import type { CsvCell, CsvSection } from '@/lib/types'

/*
 * Renders a benchmark CSV exactly as it exists on disk.
 *
 * Columns are not renamed, reordered or dropped, and no value is recomputed
 * — the point of this page is that every number can be checked against its
 * source file, which is named beneath each table.
 */

/** Metrics in [0,1] keep four decimals: enough to compare models. */
const RATE_COLUMNS = new Set([
  'Accuracy',
  'Precision',
  'Recall',
  'F1',
  'ROC-AUC',
  'ROC_AUC',
  'PR-AUC',
  'PR_AUC',
  'FPR',
  'Detection_Rate',
  'DetectionRate',
  'Anomaly_Rate',
  'MeanProbability',
  'MedianProbability',
  'Threshold',
  'loss',
  'val_loss',
])

const COUNT_COLUMNS = new Set([
  'Rows',
  'TN',
  'FP',
  'FN',
  'TP',
  'True_Negatives',
  'False_Positives',
  'False_Negatives',
  'True_Positives',
  'Anomalies',
  'Detected',
  'Missed',
  'Scenario',
  'epoch',
])

const SECONDS_COLUMNS = new Set([
  'Train Time (sec)',
  'Train_Time_Sec',
  'Prediction_Time_Sec',
])

function isNumericColumn(column: string, rows: Record<string, CsvCell>[]) {
  return rows.some((row) => typeof row[column] === 'number')
}

function formatCell(column: string, value: CsvCell): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'string') return value

  if (COUNT_COLUMNS.has(column)) {
    return value.toLocaleString('en-US')
  }

  if (SECONDS_COLUMNS.has(column)) {
    return `${value.toFixed(1)} s`
  }

  if (RATE_COLUMNS.has(column)) {
    // Very small or very large values would be misleading at 4 dp.
    if (value !== 0 && Math.abs(value) < 0.0001) return value.toExponential(2)
    if (Math.abs(value) >= 1000) return value.toExponential(3)
    return value.toFixed(4)
  }

  if (Number.isInteger(value)) return value.toLocaleString('en-US')
  if (Math.abs(value) >= 1000) return value.toExponential(3)

  return value.toFixed(4)
}

export function BenchmarkTable({
  section,
  columns,
  highlightRow,
  note,
  emptyLabel = 'This artifact is not present in the repository.',
}: {
  section: CsvSection
  /** Restrict and order columns. Defaults to the file's own order. */
  columns?: string[]
  /** Emphasise the row whose first cell matches, e.g. the deployed model. */
  highlightRow?: string
  note?: string
  emptyLabel?: string
}) {
  if (!section.available) {
    return (
      <div>
        <p className="py-4 text-xs text-text-2">{emptyLabel}</p>
        <SourceNote file={section.file} note="not found" />
      </div>
    )
  }

  const shown = (columns ?? section.columns).filter((column) =>
    section.columns.includes(column),
  )

  const first = shown[0]

  return (
    <div>
      <div className="scroll-x">
        <table className="tbl">
          <thead>
            <tr>
              {shown.map((column) => (
                <th
                  key={column}
                  scope="col"
                  className={clsx(
                    isNumericColumn(column, section.rows) && 'text-right',
                  )}
                >
                  {column.replace(/_/g, ' ')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {section.rows.map((row, index) => {
              const key = first ? String(row[first] ?? index) : String(index)
              const highlighted =
                highlightRow !== undefined &&
                first !== undefined &&
                String(row[first]) === highlightRow

              return (
                <tr
                  key={`${key}-${index}`}
                  className={clsx(highlighted && 'bg-surface-2')}
                >
                  {shown.map((column) => (
                    <td
                      key={column}
                      className={clsx(
                        isNumericColumn(column, section.rows) && 'num',
                        highlighted && 'font-medium',
                      )}
                    >
                      {formatCell(column, row[column] ?? null)}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <SourceNote file={section.file} note={note} />
    </div>
  )
}
