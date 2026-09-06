/*
 * Mirrors the backend response shapes exactly.
 *
 * These types are the contract boundary. If a field is not declared here,
 * the backend does not return it — the UI must not display it.
 *
 * Sources:
 *   backend/telemetry.py  (_build_record, stats, timeseries, distribution)
 *   backend/inference.py  (predict result keys)
 *   backend/main.py       (route response shapes)
 *   backend/benchmarks.py (collect, feature_list, autoencoder_schema)
 */

/** The only verdict vocabulary the model produces. */
export type Label = 'MALICIOUS' | 'BENIGN'

/** CTU-13 ThreatClass vocabulary. Binary — there is no attack family. */
export type ThreatClass = 'BOTNET' | 'BENIGN'

/** Risk bands from inference.py:928-950. */
export type Risk = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'SAFE'

export type TimeRange = '1m' | '5m' | '15m' | '1h'

export interface FlowRecord {
  seq: number
  received_at: number

  /* Identity — taken from the request the capture layer submitted. */
  flow_id: string
  src_ip: string
  dst_ip: string
  src_port: number
  dst_port: number
  protocol: string
  direction: string
  state: string

  /* Volume and timing — submitted feature values, not re-derived. */
  packets: number
  bytes: number
  src_bytes: number
  dst_bytes: number
  duration: number
  packets_per_second: number
  bytes_per_second: number
  avg_packet_size: number

  /* Verdict — verbatim from inference.predict(). */
  timestamp: string
  prediction: 0 | 1
  label: Label | ''
  threat_class: ThreatClass | ''
  risk: Risk | ''
  confidence: number
  xgboost_score: number
  /** Mean squared reconstruction error. Unbounded; NOT a probability. */
  autoencoder_score: number
  autoencoder_normalized: number
  hybrid_score: number
  gated: boolean
  xgboost_malicious: boolean
  autoencoder_anomalous: boolean
  explanation: string
  supporting_features: Record<string, string | number>
  evidence: {
    xgboost_threshold?: number
    autoencoder_threshold?: number
    xgboost_weight?: number
    autoencoder_weight?: number
  }
  details: {
    xgboost_threshold?: number
    autoencoder_threshold?: number
    xgboost_weight?: number
    autoencoder_weight?: number
    autoencoder_non_zero_features?: number
    autoencoder_encoded_features?: number
    categorical_resolution?: string[]
    timestamp_source?: string
    evidence_features?: string[]
    replay_source?: string
  }

  /* Console-side only. In-memory, cleared when the API restarts. */
  acknowledged: boolean
}

export interface Stats {
  window_seconds: number

  flows: number
  flows_per_second: number
  packets: number
  packets_per_second: number
  bytes: number
  bytes_per_second: number
  bits_per_second: number

  threats: number
  threat_share: number

  unique_sources: number
  unique_destinations: number

  capture_state: 'receiving' | 'idle'
  seconds_since_last_flow: number | null

  buffer_size: number
  buffer_capacity: number
  subscribers: number

  totals: {
    flows: number
    threats: number
    packets: number
    bytes: number
    xgb_alerts: number
    ae_alerts: number
    errors: number
  }
  uptime_seconds: number
}

export interface TimeseriesPoint {
  ts: number
  flows: number
  packets: number
  bytes: number
  threats: number
  xgb_alerts: number
  ae_alerts: number
  flows_per_second: number
  packets_per_second: number
  bits_per_second: number
}

export interface Timeseries {
  range: TimeRange
  window_seconds: number
  bucket_seconds: number
  points: TimeseriesPoint[]
}

export interface FlowPage {
  flows: FlowRecord[]
  has_more: boolean
  next_before_seq: number | null
  buffer_size: number
  buffer_capacity: number
}

export interface AlertPage {
  alerts: FlowRecord[]
  counts: {
    critical: number
    high: number
    medium: number
    acknowledged: number
    total: number
  }
}

export interface Talker {
  address: string
  flows: number
  bytes: number
}

export interface Distribution {
  sample_size: number
  protocols: { protocol: string; flows: number }[]
  risks: { risk: string; flows: number }[]
  top_sources: Talker[]
  top_destinations: Talker[]
}

export interface ModelInfo {
  status: string
  xgboost: { features: number; threshold: number; weight: number }
  autoencoder: { features: number; threshold: number; weight: number }
  hybrid: {
    mode: string
    xgboost_weight: number
    autoencoder_weight: number
  }
}

export interface HealthArtifact {
  name: string
  file: string
  present: boolean
}

export interface Health {
  status: string
  models: string
  timestamp: string
  models_loaded?: boolean
  artifacts?: HealthArtifact[]
  missing_artifacts?: string[]
  /** Present only when served by scripts/replay_api.py. */
  replay?: boolean
  source?: string
}

/* --- Benchmarks ---------------------------------------------------- */

export type CsvCell = string | number | null

export interface CsvSection {
  file: string
  path: string
  available: boolean
  error?: string
  columns: string[]
  rows: Record<string, CsvCell>[]
}

export interface JsonSection<T = Record<string, unknown>> {
  file: string
  path: string
  available: boolean
  error?: string
  data: T | null
}

export interface Benchmarks {
  sections: {
    model_comparison: CsvSection
    multiscenario: CsvSection
    multiscenario_per_scenario: CsvSection
    hybrid: CsvSection
    hybrid_per_scenario: CsvSection
    autoencoder: CsvSection
    autoencoder_per_scenario: CsvSection
    autoencoder_training_history: CsvSection
    threshold_sweep: CsvSection
    threshold_recommendation: JsonSection
    best_threshold: JsonSection
    synthetic_per_attack: CsvSection
    hybrid_config: JsonSection
    streaming_summary: JsonSection<StreamingSummary>
  }
  missing: string[]
  models_dir: string
  outputs_dir: string
}

export interface StreamingSummary {
  project: string
  mode: string
  testing_scenarios: number[]
  total_rows: number
  benign: number
  suspicious: number
  malicious: number
  xgboost_alerts: number
  autoencoder_alerts: number
  configuration: Record<string, number>
  scenario_results: {
    scenario: number
    rows: number
    benign: number
    suspicious: number
    malicious: number
    xgboost_alerts: number
    autoencoder_alerts: number
  }[]
}

export interface FeatureInventory {
  xgboost: {
    file: string
    available: boolean
    features: string[]
    error?: string
  }
  autoencoder: {
    file: string
    available: boolean
    numeric_features: string[]
    categorical_features: string[]
    input_dimension: number | null
    encoded_feature_count?: number
    categorical_breakdown: { feature: string; columns: number }[]
    error?: string
  }
}

/* --- Demo controls -------------------------------------------------- */

export interface DemoRun {
  preset: string
  label: string
  command: string
  started_at: number
  finished_at: number | null
  exit_code: number | null
  outcome: string
}

export interface DemoPreset {
  name: string
  label: string
  description: string
  expectation: string
  command: string
}

export interface DemoStatus {
  enabled: boolean
  loopback: boolean
  script_available: boolean
  env_flag: string
  max_run_seconds: number
  running: DemoRun | null
  last_run: DemoRun | null
  presets: DemoPreset[]
  reason?: string
}
