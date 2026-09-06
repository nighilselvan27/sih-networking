# sih-networking

CTU-13 Hybrid IDS — real-time intrusion detection using a gated XGBoost +
Autoencoder hybrid over unidirectional IP flows.

Pipeline: Scapy/Npcap live capture → unidirectional flow aggregation →
30 numeric features → FastAPI `/predict` → XGBoost (threshold 0.20) +
Autoencoder (threshold 0.2554, 257-dim input) gated hybrid (weights 0.9 / 0.1)
→ live detection output.

## LIVE ATTACK TESTING

A controlled, **local-only** harness for demonstrating the IDS in real time.
All traffic targets loopback / private addresses; the generator refuses public
destinations. Predictions always come from the existing trained models — the
harness never fakes a verdict.

Use three terminals.

**Terminal 1 — backend API**

```bash
python backend/main.py
```

(Set `AE_DEBUG=0` in the environment first for peak throughput; the default
`AE_DEBUG=1` prints a per-flow diagnostic table.)

**Terminal 2 — live capture** (needs Npcap; run the shell as Administrator if
capture does not start)

```bash
python scripts/live_capture.py
```

**Terminal 3 — controlled traffic generator**

```bash
# Benign UDP (TEST 1)
python scripts/test_traffic.py --type benign-udp --duration 10 --rate 200 --port 9999

# High-rate UDP flood (TEST 2)
python scripts/test_traffic.py --type udp --duration 10 --rate 3000 --port 9999

# TCP SYN toward a closed local port (TEST 3)
python scripts/test_traffic.py --type syn --duration 10 --rate 1500 --port 9998

# High-rate benign multi-socket burst
python scripts/test_traffic.py --type burst --duration 10 --rate 2000

# Benign TCP conversation against a local echo listener
python scripts/test_traffic.py --type benign-tcp --count 200 --size 512
```

Running `python scripts/test_traffic.py` with no arguments sends the original
default: benign UDP, 5000 packets to `127.0.0.1:9999`.

`test_traffic.py` safety: only loopback / private targets are allowed; rate,
count, size and duration are capped (`--rate` ≤ 20000 pps, `--duration` ≤ 300 s).
The TCP SYN test uses ordinary OS sockets (real SYNs, no source-IP spoofing).

## THROUGHPUT BENCHMARK

Demonstrates the traffic rate the IDS was tested against.

```bash
# Measure the real capture→predict pipeline (generate UDP in the background):
python scripts/throughput_test.py --mode live --duration 30 --generate udp

# Or measure while you drive traffic yourself from Terminal 3:
python scripts/throughput_test.py --mode live --duration 30

# Pure prediction throughput / latency, no capture (fallback):
python scripts/throughput_test.py --mode api --duration 20 --rate 500 --workers 8
```

Reports test duration, packets captured/sec, flows/sec, predictions/sec,
approximate Mbps, average and P95 API latency, successful predictions and API
errors. `throughput_test.py` reuses the aggregation code from
`scripts/live_capture.py` and the existing `/predict` endpoint — it does not
modify the backend or any model artifact.

## OPERATOR CONSOLE (UniNDR)

A web console for the detection pipeline, in `frontend/`. It is a presentation
layer: it reads what the API has already scored and changes nothing about
capture, feature extraction or the models.

### Running it

```bash
# Terminal 4 — console (development)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The dev server proxies `/api`, `/predict`,
`/model-info` and `/health` to `http://127.0.0.1:8000`; set `IDS_API_URL` to
point somewhere else.

To serve the console from the API itself instead of a second process:

```bash
cd frontend && npm run build
```

`backend/main.py` then serves it at http://127.0.0.1:8000/console.

### What the console shows

Overview, Live Monitor, Alerts, Traffic, Flows, Models, Benchmarks, Settings.

Every number is either a value the API returned or a value read from a file in
`models/` or `outputs/`, and each benchmark table names its source file. The
console shows no attack-family label, because the detector is binary — it
returns `MALICIOUS`/`BENIGN` with a risk band and does not identify an attack
type. Alert detail shows the gate arithmetic itself: both model scores against
their thresholds, which branch fired, and the hybrid score.

Capture status is derived from how long ago the API last scored a flow. The
API cannot see the Scapy process, so the console never claims capture is
connected — it reports "Receiving flows" or "Idle" and says how long it has
been.

### Read-side API

These endpoints were added for the console and are read-only over an in-memory
ring buffer of predictions the API has already made. Nothing is persisted; the
buffer and any alert acknowledgements are cleared when the API restarts.

```
GET  /api/stats             rolling counters and capture state
GET  /api/timeseries        1m | 5m | 15m | 1h, bucketed
GET  /api/flows             newest-first page, with filters
GET  /api/flows/{seq}       one scored flow
GET  /api/alerts            flows scored malicious, with severity counts
POST /api/alerts/{seq}/ack  acknowledge (in memory only)
GET  /api/distribution      protocol mix and top talkers
GET  /api/stream            server-sent events, one per scored flow
GET  /api/benchmarks        parsed benchmark artifacts from models/ and outputs/
GET  /api/features          model input schema
POST /api/buffer/reset      clear the buffer
```

`GET /health` now reports the real loader state and which model artifacts are
present, instead of a fixed string.

### Demo controls

The Live Monitor can start the presets from `scripts/test_traffic.py` directly.
This is **off by default**. To enable it:

```bash
set IDS_DEMO_CONTROLS=1        # Windows
export IDS_DEMO_CONTROLS=1     # POSIX
```

The API then accepts a preset **name** only, from a fixed server-side list, from
loopback callers only, with `shell=False` and a hard run cap. Host, port, rate,
size and duration are never taken from the request. `test_traffic.py`'s own
refusal of non-private targets and its rate/count/size caps still apply. With
the flag unset the panel shows the equivalent terminal command instead.

### Running the console without the models

`scripts/replay_api.py` serves the same read-side contract without loading
xgboost, tensorflow or the scaler — useful for working on the console, or on a
machine that cannot run the detector.

```bash
# Real recorded model scores from outputs/streaming_results.csv,
# re-evaluated through the same gate as backend/inference.py.
python scripts/replay_api.py --source ctu13 --start 130000 --rate 14

# Generated fixtures for UI work only. Not real traffic, not real output.
python scripts/replay_api.py --source synthetic --rate 20
```

It runs no model and invents no verdict. In `ctu13` mode the verdicts come from
real XGBoost probabilities and reconstruction errors recorded during the
scenario 11-13 evaluation; that file has no packet or byte counts, so
throughput and packets/sec read zero. `--start` skips into a busier part of the
capture — the early rows are almost entirely background traffic — and the
startup banner reports the detection density you should expect.

`/health` reports `replay: true` in this mode and the console shows a warning
on the Settings page, so replayed data can never be mistaken for live capture.
