"""
CTU-13 HYBRID IDS - THROUGHPUT / PERFORMANCE TEST

Measures the real inference pipeline so we can state and demonstrate the
traffic rate the IDS was tested against (a SIH requirement).

It reuses the EXISTING aggregation code from scripts/live_capture.py
(get_flow_key, update_flow, extract_features, build_payload, build_state and
the module-level `flows` dict). It does NOT modify live_capture.py or the
backend, and every prediction still comes from the trained models via the
existing /predict endpoint.

Modes
-----
    live   (default)  Sniff real packets on the Npcap interface, aggregate
                      them into the existing unidirectional flows, and POST
                      each expired flow to /predict - timing everything.

    api               No capture. Synthesise flows from a template and POST
                      them at a target rate across worker threads. Measures
                      pure prediction throughput / latency and works even if
                      Npcap capture is unavailable.

Optionally --generate drives scripts/test_traffic.py in a background thread so
the whole benchmark is a single command.

Metrics reported
----------------
    Test duration, packets captured/sec, flows/sec, predictions/sec,
    approximate Mbps, average and P95 API latency, successful predictions,
    API errors.

Examples
--------
    # Measure live capture while you generate traffic in another terminal:
    python scripts/throughput_test.py --mode live --duration 30

    # One-command: generate a UDP flood and measure the pipeline:
    python scripts/throughput_test.py --mode live --duration 15 --generate udp

    # Pure API throughput (no capture):
    python scripts/throughput_test.py --mode api --duration 20 --rate 500 --workers 8
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

# --- import the existing live-capture pipeline without modifying it ---
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import live_capture as lc   # noqa: E402


DEFAULT_API = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/"


# =========================================================
# SHARED COUNTERS
# =========================================================

class Stats:

    def __init__(self):
        self.lock = threading.Lock()
        self.packets = 0
        self.bytes = 0
        self.flows = 0
        self.predictions = 0
        self.errors = 0
        self.latencies = []      # seconds

    def add_packet(self, size):
        with self.lock:
            self.packets += 1
            self.bytes += size

    def add_prediction(self, latency, ok):
        with self.lock:
            self.flows += 1
            if ok:
                self.predictions += 1
                self.latencies.append(latency)
            else:
                self.errors += 1


# =========================================================
# API POST (quiet - does NOT reuse send_prediction, which prints)
# =========================================================

def post_flow(api_url, payload, stats):

    t0 = time.perf_counter()
    ok = False
    try:
        resp = requests.post(api_url, json=payload, timeout=15)
        ok = (resp.status_code == 200)
    except Exception:
        ok = False
    latency = time.perf_counter() - t0
    stats.add_prediction(latency, ok)


# =========================================================
# LIVE MODE
# =========================================================

def run_live(args, stats):

    stop = threading.Event()

    # Fresh flow table (reuse live_capture's own structures/logic).
    lc.flows.clear()

    def on_packet(packet):
        # Mirror live_capture.packet_callback's filter.
        from scapy.all import IP, TCP, UDP
        if IP not in packet:
            return
        if not (TCP in packet or UDP in packet):
            return
        stats.add_packet(len(packet))
        lc.update_flow(packet)

    def sniffer():
        from scapy.all import sniff
        while not stop.is_set():
            sniff(
                iface=args.iface,
                prn=on_packet,
                store=False,
                timeout=1,
            )

    sniff_thread = threading.Thread(target=sniffer, daemon=True)
    sniff_thread.start()

    print(f"[INFO] Capturing on {args.iface} for {args.duration:.0f}s "
          f"(flow window {args.window}s)...")

    start = time.perf_counter()

    # Expire flows older than the window and POST them, just like
    # live_capture.process_flows() but with timing and no per-flow printing.
    while time.perf_counter() - start < args.duration:

        now = time.time()
        expired = []

        for key, flow in list(lc.flows.items()):
            if now - flow["start_time"] >= args.window:
                features = lc.extract_features(flow)
                _, payload = lc.build_payload(flow, features)
                post_flow(args.api, payload, stats)
                expired.append(key)

        for key in expired:
            lc.flows.pop(key, None)

        time.sleep(0.2)

    stop.set()

    # Flush every remaining flow so nothing is silently dropped.
    for key, flow in list(lc.flows.items()):
        features = lc.extract_features(flow)
        _, payload = lc.build_payload(flow, features)
        post_flow(args.api, payload, stats)
    lc.flows.clear()

    sniff_thread.join(timeout=3.0)


# =========================================================
# API MODE
# =========================================================

TEMPLATE_FLOW = {
    "Dur": 5.0, "Sport": 49760, "Dport": 443, "sTos": 0, "dTos": 0,
    "TotPkts": 10, "TotBytes": 1500, "SrcBytes": 1500,
    "PacketsPerSecond": 2.0, "BytesPerSecond": 300.0, "AvgPacketSize": 150.0,
    "DstBytes": 0, "SrcByteRatio": 1.0, "DstByteRatio": 0.0,
    "SourceFlowCount30s": 1, "UniqueDstIPs30s": 1, "UniqueDstPorts30s": 1,
    "UniqueSrcPorts30s": 1, "SourceTotalBytes30s": 1500,
    "SourceTotalPackets30s": 10, "DestinationFlowCount30s": 1,
    "UniqueSrcIPs30s": 1, "DestinationTotalBytes30s": 0,
    "DestinationRepeatCount": 1, "InterArrivalTime": 0.5,
    "PairInterArrivalTime": 0.5, "FlowsPerSecond30s": 0.2,
    "PacketsPerSecond30s": 2.0, "BytesPerSecond30s": 300.0,
    "SourceOutboundRatio": 1.0,
    "Proto": "tcp", "Dir": "->", "State": "S_",
}


def run_api(args, stats):

    stop = threading.Event()
    rate = args.rate or 500.0
    interval = 1.0 / rate

    def worker():
        while not stop.is_set():
            payload = dict(TEMPLATE_FLOW)
            # Vary the source port so flows look distinct.
            payload["Sport"] = 1024 + (stats.flows % 60000)
            post_flow(args.api, payload, stats)
            time.sleep(interval * args.workers)

    print(f"[INFO] API mode: target {rate:.0f} req/s across "
          f"{args.workers} workers for {args.duration:.0f}s...")

    threads = [
        threading.Thread(target=worker, daemon=True)
        for _ in range(args.workers)
    ]
    for t in threads:
        t.start()

    time.sleep(args.duration)
    stop.set()
    for t in threads:
        t.join(timeout=3.0)


# =========================================================
# OPTIONAL TRAFFIC GENERATOR
# =========================================================

def start_generator(kind, duration):
    """Run scripts/test_traffic.py in the background; return the Popen."""

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "test_traffic.py"),
        "--type", kind,
        "--duration", str(int(duration)),
        "--rate", "500",
    ]
    print(f"[INFO] Launching generator: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# =========================================================
# REPORT
# =========================================================

def percentile(sorted_values, pct):
    if not sorted_values:
        return None
    k = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[k]


def print_report(args, stats, elapsed, generated):

    pps = stats.packets / elapsed if elapsed > 0 else 0.0
    fps = stats.flows / elapsed if elapsed > 0 else 0.0
    pred_ps = stats.predictions / elapsed if elapsed > 0 else 0.0
    mbps = (stats.bytes * 8) / elapsed / 1e6 if elapsed > 0 else 0.0

    lat = sorted(stats.latencies)
    avg_ms = (sum(lat) / len(lat) * 1000) if lat else 0.0
    p95 = percentile(lat, 95)
    p95_ms = (p95 * 1000) if (p95 is not None and len(lat) >= 20) else None

    print()
    print("=" * 50)
    print("IDS THROUGHPUT TEST")
    print("=" * 50)
    print(f"Mode                : {args.mode}")
    print(f"Duration            : {elapsed:.1f} sec")
    if generated is not None:
        print(f"Packets generated   : {generated}")
    if args.mode == "live":
        print(f"Packets captured    : {stats.packets}")
        print(f"Packets/sec         : {pps:.1f}")
        print(f"Throughput          : {mbps:.2f} Mbps (captured)")
    print(f"Flows observed      : {stats.flows}")
    print(f"Flows/sec           : {fps:.1f}")
    print(f"Predictions         : {stats.predictions}")
    print(f"Predictions/sec     : {pred_ps:.1f}")
    print(f"Avg API latency     : {avg_ms:.1f} ms")
    if p95_ms is not None:
        print(f"P95 API latency     : {p95_ms:.1f} ms")
    else:
        print(f"P95 API latency     : n/a (need >=20 samples, got {len(lat)})")
    print(f"Successful          : {stats.predictions}")
    print(f"API errors          : {stats.errors}")
    print("=" * 50)
    print("[NOTE] Latency is dominated by the per-flow autoencoder forward "
          "pass. If the backend was started with AE_DEBUG=1 (default), its "
          "per-flow debug printing lowers throughput; start it with "
          "AE_DEBUG=0 for peak numbers.")


# =========================================================
# MAIN
# =========================================================

def check_backend(api_url):
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def main():

    parser = argparse.ArgumentParser(
        description="Throughput / latency benchmark for the CTU-13 IDS pipeline.",
    )
    parser.add_argument("--mode", choices=["live", "api"], default="live")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--iface", default=lc.INTERFACE,
                        help="Npcap interface for live mode.")
    parser.add_argument("--window", type=float, default=lc.WINDOW_SECONDS,
                        help="Flow expiry window (live mode).")
    parser.add_argument("--rate", type=float, default=None,
                        help="Target req/s (api mode).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Worker threads (api mode).")
    parser.add_argument("--generate",
                        choices=["none", "benign-udp", "udp", "syn", "burst"],
                        default="none",
                        help="Also run test_traffic.py in the background.")

    args = parser.parse_args()

    print("=" * 50)
    print("IDS THROUGHPUT TEST - starting")
    print("=" * 50)

    if not check_backend(args.api):
        print("[ERROR] Backend not reachable. Start it first:")
        print("        python backend/main.py")
        sys.exit(1)
    print("[OK] Backend reachable.")

    gen_proc = None
    if args.generate != "none":
        gen_proc = start_generator(args.generate, args.duration)

    stats = Stats()
    start = time.perf_counter()

    try:
        if args.mode == "live":
            run_live(args, stats)
        else:
            run_api(args, stats)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        if gen_proc is not None:
            try:
                gen_proc.terminate()
            except Exception:
                pass

    elapsed = time.perf_counter() - start
    generated = None  # we do not scrape the generator's own count

    print_report(args, stats, elapsed, generated)


if __name__ == "__main__":
    main()
