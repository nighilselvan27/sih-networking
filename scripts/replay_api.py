"""
Offline replay server for the operator console.

WHAT THIS IS FOR

The real API (backend/main.py) needs xgboost, tensorflow and the fitted
scaler at models/autoencoder_scaler.joblib. This server needs none of them.
It serves the same read-side JSON contract so the console can be developed,
reviewed and demonstrated on a machine that cannot load the models.

It is NOT a second detector. It runs no model and invents no verdict.

WHERE THE NUMBERS COME FROM

  --source ctu13   (default)
      Replays outputs/streaming_results.csv, which holds the real
      per-flow XGBoost probability and autoencoder reconstruction error
      recorded when scenarios 11-13 were scored by the trained models.
      Each row's verdict is recomputed by applying the SAME gate and the
      SAME constants that backend/inference.py uses (XGB_THRESHOLD 0.20,
      AE_THRESHOLD 0.25541964, weights 0.9/0.1). Identical inputs, identical
      rule, identical verdict.

      Caveat, stated plainly: that file records scores and identities but
      no packet or byte counts. Throughput and packets/sec will therefore
      read zero in this mode. Flow counts, detections, risk levels and every
      score are real.

  --source synthetic
      Generated flows for UI development only. Not real traffic and not
      real model output. /health reports source="synthetic" so the console
      shows a warning banner.

Every response includes a "replay" marker in /health so the console can
never silently present replayed data as live capture.

Stdlib only.

    py scripts/replay_api.py --source ctu13 --rate 8
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs
import argparse
import csv
import json
import queue
import random
import sys
import threading
import time


BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR / "backend"))

import benchmarks  # noqa: E402
import telemetry  # noqa: E402


# The constants below are read from backend/inference.py rather than
# restated, so this server cannot drift from the real decision rule.
def _load_inference_constants() -> Dict[str, float]:

    text = (BASE_DIR / "backend" / "inference.py").read_text(
        encoding="utf-8"
    )

    wanted = {
        "XGB_THRESHOLD": None,
        "AE_THRESHOLD": None,
        "XGB_WEIGHT": None,
        "AE_WEIGHT": None,
    }

    for line in text.splitlines():

        for name in wanted:

            if wanted[name] is None and line.startswith(f"{name} ="):
                wanted[name] = float(line.split("=", 1)[1].strip())

    missing = [name for name, value in wanted.items() if value is None]

    if missing:
        raise SystemExit(
            f"Could not read {', '.join(missing)} from backend/inference.py"
        )

    return wanted


CONSTANTS = _load_inference_constants()

XGB_THRESHOLD = CONSTANTS["XGB_THRESHOLD"]
AE_THRESHOLD = CONSTANTS["AE_THRESHOLD"]
XGB_WEIGHT = CONSTANTS["XGB_WEIGHT"]
AE_WEIGHT = CONSTANTS["AE_WEIGHT"]

RESULTS_CSV = BASE_DIR / "outputs" / "streaming_results.csv"

_source_mode = "ctu13"


# ============================================================
# VERDICT
#
# A transcription of backend/inference.py:825-950. Any change there
# must be mirrored here or the replay stops matching the detector.
# ============================================================

def build_verdict(
    xgb_score: float,
    ae_error: float,
    flow_id: str,
    timestamp: str,
) -> Dict[str, Any]:

    xgb_malicious = xgb_score >= XGB_THRESHOLD
    ae_anomalous = ae_error >= AE_THRESHOLD

    ae_normalized = (
        min(ae_error / AE_THRESHOLD, 1.0) if AE_THRESHOLD > 0 else 0.0
    )

    if xgb_malicious:
        prediction, gated = 1, True

    elif ae_anomalous and xgb_score >= 0.05:
        prediction, gated = 1, True

    else:
        prediction, gated = 0, False

    hybrid_score = min(
        max(XGB_WEIGHT * xgb_score + AE_WEIGHT * ae_normalized, 0.0), 1.0
    )

    confidence = hybrid_score if prediction == 1 else 1.0 - hybrid_score

    if prediction == 1:
        risk = (
            "CRITICAL" if confidence >= 0.85
            else "HIGH" if confidence >= 0.65
            else "MEDIUM"
        )

    else:
        risk = "LOW" if hybrid_score >= 0.10 else "SAFE"

    if prediction == 1:

        if xgb_malicious and ae_anomalous:
            explanation = (
                "Both models agree: XGBoost detected an attack pattern "
                "and the autoencoder flagged anomalous behaviour."
            )

        elif xgb_malicious:
            explanation = (
                "XGBoost detected elevated attack probability."
            )

        else:
            explanation = (
                "The autoencoder flagged anomalous behaviour on a flow "
                "XGBoost scored as uncertain."
            )

    else:
        explanation = "No attack pattern detected by either model."

    return {
        "timestamp": timestamp,
        "flow_id": flow_id,
        "threat_class": "BOTNET" if prediction == 1 else "BENIGN",
        "prediction": prediction,
        "label": "MALICIOUS" if prediction == 1 else "BENIGN",
        "confidence": round(confidence, 6),
        "xgboost_score": round(xgb_score, 6),
        "autoencoder_score": round(ae_error, 8),
        "autoencoder_normalized": round(ae_normalized, 6),
        "hybrid_score": round(hybrid_score, 6),
        "risk": risk,
        "gated": gated,
        "xgboost_malicious": xgb_malicious,
        "autoencoder_anomalous": ae_anomalous,
        "explanation": explanation,
        "evidence": {
            "xgboost_threshold": XGB_THRESHOLD,
            "autoencoder_threshold": AE_THRESHOLD,
            "xgboost_weight": XGB_WEIGHT,
            "autoencoder_weight": AE_WEIGHT,
        },
        "xgboost_probability": round(xgb_score, 6),
        "risk_level": risk,
    }


# ============================================================
# SOURCES
# ============================================================

def ctu13_rows(start: int = 0):
    """
    Stream real recorded scores from the CTU-13 replay output.

    `start` skips rows so a demonstration can begin inside a stretch of the
    capture that actually contains botnet activity. The file is in capture
    order and the early rows of scenario 11 are almost all background
    traffic, so replaying from row 0 shows a correct — but very quiet —
    picture.
    """

    if not RESULTS_CSV.exists():
        raise SystemExit(
            f"Source file not found: {RESULTS_CSV}\n"
            f"Run scripts/streaming_detector.py, or use --source synthetic."
        )

    while True:

        with RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:

            reader = csv.DictReader(fh)

            for _ in range(start):
                if next(reader, None) is None:
                    break

            for row in reader:

                yield {
                    "src_ip": row["source_ip"],
                    "dst_ip": row["destination_ip"],
                    "src_port": int(float(row["source_port"] or 0)),
                    "dst_port": int(float(row["destination_port"] or 0)),
                    "protocol": (row["protocol"] or "").lower(),
                    "xgb": float(row["xgboost_probability"] or 0.0),
                    "ae": float(row["reconstruction_error"] or 0.0),
                    "scenario": row["scenario"],
                }


def ctu13_density(start: int, sample: int = 20000) -> float:
    """Share of the upcoming rows the gate will score malicious."""

    malicious = 0
    seen = 0

    for row in ctu13_rows(start):

        if seen >= sample:
            break

        seen += 1

        if row["xgb"] >= XGB_THRESHOLD:
            malicious += 1
        elif row["ae"] >= AE_THRESHOLD and row["xgb"] >= 0.05:
            malicious += 1

    return (malicious / seen) if seen else 0.0


def synthetic_rows():
    """UI-development fixtures. Not real traffic, not real model output."""

    rng = random.Random(42)

    sources = [
        "10.0.0.24", "10.0.0.31", "192.168.1.7",
        "192.168.1.44", "172.16.0.9",
    ]
    destinations = ["10.0.0.5", "10.0.0.8", "192.168.1.1"]

    while True:

        attack = rng.random() < 0.12

        if attack:
            xgb = rng.uniform(0.22, 0.99)
            ae = rng.uniform(0.01, 0.9)
            packets = rng.randint(1500, 9000)

        else:
            xgb = rng.uniform(0.0, 0.06)
            ae = rng.uniform(0.001, 0.05)
            packets = rng.randint(4, 400)

        yield {
            "src_ip": rng.choice(sources),
            "dst_ip": rng.choice(destinations),
            "src_port": rng.randint(1024, 65535),
            "dst_port": rng.choice([53, 80, 443, 9999, 9998, 25]),
            "protocol": rng.choice(["tcp", "udp", "udp"]),
            "xgb": xgb,
            "ae": ae,
            "packets": packets,
            "bytes": packets * rng.randint(40, 700),
        }


def feed(
    source: str, rate: float, stop: threading.Event, start: int = 0
) -> None:

    rows = ctu13_rows(start) if source == "ctu13" else synthetic_rows()

    interval = 1.0 / rate if rate > 0 else 0.0

    for row in rows:

        if stop.is_set():
            return

        flow_id = (
            f"{row['src_ip']}:{row['src_port']}-"
            f"{row['dst_ip']}:{row['dst_port']}-"
            f"{row['protocol'].upper()}"
        )

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        duration = 5.0
        packets = float(row.get("packets", 0.0))
        total_bytes = float(row.get("bytes", 0.0))

        request = {
            "Dur": duration,
            "Sport": row["src_port"],
            "Dport": row["dst_port"],
            "TotPkts": packets,
            "TotBytes": total_bytes,
            "SrcBytes": total_bytes,
            "PacketsPerSecond": packets / duration,
            "BytesPerSecond": total_bytes / duration,
            "AvgPacketSize": (total_bytes / packets) if packets else 0.0,
            "Proto": row["protocol"],
            "Dir": "->",
            "State": "INT",
            "timestamp": timestamp,
            "flow_id": flow_id,
            "metadata": {
                "src_ip": row["src_ip"],
                "dst_ip": row["dst_ip"],
                "src_port": row["src_port"],
                "dst_port": row["dst_port"],
                "protocol": row["protocol"],
            },
        }

        verdict = build_verdict(row["xgb"], row["ae"], flow_id, timestamp)

        # Supporting features mirror what inference.py reports: this flow's
        # own submitted values for a set of model input features.
        verdict["supporting_features"] = {
            key: request[key]
            for key in (
                "Sport", "Dport", "Dur", "TotPkts",
                "TotBytes", "PacketsPerSecond",
            )
        }
        verdict["supporting_features"]["Proto"] = request["Proto"]
        verdict["supporting_features"]["State"] = request["State"]

        verdict["details"] = {
            "xgboost_threshold": XGB_THRESHOLD,
            "autoencoder_threshold": AE_THRESHOLD,
            "xgboost_weight": XGB_WEIGHT,
            "autoencoder_weight": AE_WEIGHT,
            "autoencoder_encoded_features": 257,
            "timestamp_source": "replay",
            "evidence_features": [
                "Sport", "Dport", "Dur", "TotPkts",
                "TotBytes", "PacketsPerSecond",
            ],
            "categorical_resolution": [
                f"Proto='{request['Proto']}' -> Proto_{request['Proto']}",
                "State='INT' -> State_INT",
            ],
            "replay_source": _source_mode,
        }

        telemetry.record(request, verdict)

        if interval:
            time.sleep(interval)


# ============================================================
# HTTP
# ============================================================

class Handler(BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter console
        pass

    # -- helpers ------------------------------------------------

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _json(self, payload: Any, status: int = 200):

        body = json.dumps(payload, default=str).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()

        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _query(self) -> Dict[str, list]:
        return parse_qs(urlparse(self.path).query)

    def _param(self, name: str, default=None):
        values = self._query().get(name)
        return values[0] if values else default

    def _int(self, name: str, default: int) -> int:
        try:
            return int(self._param(name, default))
        except (TypeError, ValueError):
            return default

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    # -- routes -------------------------------------------------

    def do_GET(self):

        path = urlparse(self.path).path

        if path == "/":
            return self._json({
                "name": "CTU-13 Hybrid IDS API (replay)",
                "status": "running",
                "replay": True,
            })

        if path == "/health":
            return self._json({
                "status": "healthy",
                "models": "replay",
                "models_loaded": False,
                "replay": True,
                "source": _source_mode,
                "artifacts": [],
                "missing_artifacts": [],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

        if path == "/model-info":
            schema = benchmarks.autoencoder_schema()

            return self._json({
                "status": "ready",
                "xgboost": {
                    "features": len(
                        benchmarks.feature_list()["features"]
                    ),
                    "threshold": XGB_THRESHOLD,
                    "weight": XGB_WEIGHT,
                },
                "autoencoder": {
                    "features": schema.get("encoded_feature_count") or 257,
                    "threshold": AE_THRESHOLD,
                    "weight": AE_WEIGHT,
                },
                "hybrid": {
                    "mode": "GATED",
                    "xgboost_weight": XGB_WEIGHT,
                    "autoencoder_weight": AE_WEIGHT,
                },
            })

        if path == "/api/stats":
            return self._json(
                telemetry.stats(window_seconds=self._int("window", 60))
            )

        if path == "/api/timeseries":
            return self._json(
                telemetry.timeseries(self._param("range", "5m"))
            )

        if path == "/api/flows":
            before = self._param("before_seq")

            return self._json(telemetry.list_flows(
                limit=self._int("limit", 100),
                before_seq=int(before) if before else None,
                protocol=self._param("protocol"),
                verdict=self._param("verdict"),
                risk=self._param("risk"),
                search=self._param("q"),
            ))

        if path.startswith("/api/flows/"):
            try:
                seq = int(path.rsplit("/", 1)[1])
            except ValueError:
                return self._json({"detail": "Bad flow id"}, 400)

            entry = telemetry.get_flow(seq)

            if entry is None:
                return self._json({"detail": "Not in buffer"}, 404)

            return self._json(entry)

        if path == "/api/alerts":
            ack = self._param("acknowledged")

            return self._json(telemetry.list_alerts(
                limit=self._int("limit", 100),
                risk=self._param("risk"),
                acknowledged=(
                    None if ack is None else ack.lower() == "true"
                ),
                search=self._param("q"),
            ))

        if path == "/api/distribution":
            return self._json(
                telemetry.distribution(limit=self._int("limit", 10))
            )

        if path == "/api/benchmarks":
            return self._json(benchmarks.collect())

        if path == "/api/features":
            return self._json({
                "xgboost": benchmarks.feature_list(),
                "autoencoder": benchmarks.autoencoder_schema(),
            })

        if path == "/api/demo/status":
            return self._json({
                "enabled": False,
                "loopback": True,
                "script_available": False,
                "env_flag": "IDS_DEMO_CONTROLS",
                "max_run_seconds": 45,
                "running": None,
                "last_run": None,
                "presets": [],
                "reason": "Demo controls are unavailable in replay mode.",
            })

        if path == "/api/stream":
            return self._stream()

        return self._json({"detail": "Not found"}, 404)

    def do_POST(self):

        path = urlparse(self.path).path

        length = int(self.headers.get("Content-Length") or 0)

        if length:
            self.rfile.read(length)

        if path.startswith("/api/alerts/") and path.endswith("/ack"):
            try:
                seq = int(path.split("/")[3])
            except (ValueError, IndexError):
                return self._json({"detail": "Bad alert id"}, 400)

            entry = telemetry.acknowledge(seq)

            if entry is None:
                return self._json({"detail": "Not in buffer"}, 404)

            return self._json(entry)

        if path == "/api/buffer/reset":
            telemetry.reset()
            return self._json({"cleared": True})

        if path.startswith("/api/demo/"):
            return self._json(
                {"detail": "Demo controls are unavailable in replay mode."},
                403,
            )

        return self._json({"detail": "Not found"}, 404)

    # -- SSE ----------------------------------------------------

    def _stream(self):

        channel = telemetry.subscribe()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self._cors()
        self.end_headers()

        last_heartbeat = time.time()

        try:
            self.wfile.write(b"event: ready\ndata: {}\n\n")
            self.wfile.flush()

            while True:

                sent = 0

                while sent < 100:

                    try:
                        entry = channel.get_nowait()

                    except queue.Empty:
                        break

                    sent += 1

                    payload = json.dumps(entry, default=str)

                    self.wfile.write(
                        f"event: flow\ndata: {payload}\n\n".encode("utf-8")
                    )

                if sent:
                    self.wfile.flush()

                now = time.time()

                if now - last_heartbeat >= 15:
                    last_heartbeat = now

                    self.wfile.write(
                        f'event: heartbeat\ndata: {{"ts": {now:.3f}}}\n\n'
                        .encode("utf-8")
                    )
                    self.wfile.flush()

                if not sent:
                    time.sleep(0.25)

        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

        finally:
            telemetry.unsubscribe(channel)


# ============================================================
# MAIN
# ============================================================

def main():

    global _source_mode

    parser = argparse.ArgumentParser(
        description=(
            "Offline replay server for the UniNDR console. Serves the "
            "read-side API contract without loading any model."
        )
    )
    parser.add_argument(
        "--source",
        choices=["ctu13", "synthetic"],
        default="ctu13",
        help=(
            "ctu13: real recorded model scores from outputs/ (default). "
            "synthetic: generated fixtures for UI development only."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--rate",
        type=float,
        default=8.0,
        help="Flows replayed per second (default: 8).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help=(
            "ctu13 only: skip this many rows before replaying. The file is "
            "in capture order and early scenario-11 rows are almost all "
            "background traffic. Try 130000 for scenario 13."
        ),
    )

    args = parser.parse_args()

    _source_mode = args.source

    print("=" * 70)
    print("UniNDR REPLAY SERVER - no model is loaded, nothing is detected")
    print("=" * 70)
    print(f"Source      : {args.source}")

    if args.source == "ctu13":
        print(f"              {RESULTS_CSV}")
        print("              Real XGBoost / autoencoder scores, replayed")
        print("              through the gate in backend/inference.py.")
        print("              This file carries no packet or byte counts, so")
        print("              throughput and packets/sec will read zero.")

        density = ctu13_density(args.start)

        print(f"Start row   : {args.start}")
        print(
            f"Detections  : {density * 100:.2f}% of the next 20,000 rows "
            f"({density * args.rate * 60:.1f}/min at this rate)"
        )

        if density < 0.005:
            print(
                "              This stretch is nearly all background "
                "traffic.\n"
                "              Use --start to reach a busier part of the "
                "capture,\n"
                "              or --source synthetic for UI work."
            )

    else:
        print("              GENERATED FIXTURES - not real traffic and not")
        print("              real model output. UI development only.")

    print(f"Gate        : XGB >= {XGB_THRESHOLD}, "
          f"AE >= {AE_THRESHOLD} (with XGB >= 0.05)")
    print(f"Rate        : {args.rate} flows/sec")
    print(f"Listening   : http://{args.host}:{args.port}")
    print("=" * 70)

    stop = threading.Event()

    worker = threading.Thread(
        target=feed,
        args=(args.source, args.rate, stop, args.start),
        daemon=True,
    )
    worker.start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
