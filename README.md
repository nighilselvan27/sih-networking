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
