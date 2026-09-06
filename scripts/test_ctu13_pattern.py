"""
CTU-13 PATTERN VALIDATION

  *** LOCAL LAB TEST ONLY ***

Purpose: VALIDATION, not forcing a detection.

This exercises the EXISTING trained XGBoost + Autoencoder hybrid against live
traffic whose flow-level shape resembles the CTU-13 BOTNET class the models
were trained on, and reports whatever the models actually decide.

It generates traffic only. It never talks to /predict and never asserts a
verdict - the prediction comes from scripts/live_capture.py -> /predict ->
the trained models, exactly as in normal operation.

What the CTU-13 BOTNET class looks like
---------------------------------------
Derived empirically from the shipped artifacts by joining
models/autoencoder_predictions.csv (real CTU-13 test-set metadata) with
models/autoencoder_xgb_hybrid_predictions.csv (per-row XGBoost probability),
547,490 rows. There is exactly one malicious class: BOTNET. Among the flows
XGBoost detects at threshold 0.20:

    Dport : 25 (SMTP spam), 443, 6667 (IRC C2), 65500, 80
    Proto : tcp, icmp
    State : S_, S_RA (unanswered / refused SYNs), FSPA_FSPA

i.e. a spam bot / IRC C2 beacon - SHORT, SPARSE connections with SMALL
packets, not a flood. That is the shape reproduced here.

Honest expectation
------------------
The dominant XGBoost feature is Sport (33.9% of total gain) and it behaves as
a hard gate: source ports below ~7000 score high, while the ephemeral range
Windows assigns to every socket (49152-65535) scores ~0.0003. CTU-13 bots used
low/fixed source ports; an ordinary OS socket cannot. This script deliberately
does NOT bind a low source port to game that, so the expected verdict is
BENIGN. Run --explain for the supporting numbers.

PCAP replay is not offered: no capture files exist in this repository
(data/scenario*/ holds only README stubs) and replay would require raw packet
injection plus Administrator.

Usage
-----
    python scripts/test_ctu13_pattern.py
    python scripts/test_ctu13_pattern.py --pattern spam-bot --duration 30
    python scripts/test_ctu13_pattern.py --pattern irc-c2 --proto tcp
    python scripts/test_ctu13_pattern.py --explain
"""

import argparse
import socket
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the safety guards and pacing from the stage-2 generator.
from test_traffic import (          # noqa: E402
    MAX_DURATION,
    MAX_SIZE,
    clamp,
    print_banner,
    validate_target,
)


# =========================================================
# CTU-13 BOTNET PATTERNS
#
# Destination ports are taken from the flows XGBoost actually detects
# in the shipped CTU-13 test-set predictions (see module docstring).
# =========================================================

PATTERNS = {
    "spam-bot": {
        "label": "SPAM-BOT (SMTP beacon, CTU-13 BOTNET profile)",
        "port": 25,
        "packets_per_beacon": 3,
        "interval": 5.0,
        "size": 32,
    },
    "dns-beacon": {
        "label": "DNS-BEACON (CTU-13 BOTNET profile)",
        "port": 53,
        "packets_per_beacon": 5,
        "interval": 5.0,
        "size": 32,
    },
    "irc-c2": {
        "label": "IRC-C2 (port 6667 beacon, CTU-13 BOTNET profile)",
        "port": 6667,
        "packets_per_beacon": 4,
        "interval": 5.0,
        "size": 48,
    },
}

DEFAULT_HOST = "127.0.0.1"


# =========================================================
# BEACON GENERATOR
#
# Each beacon opens a NEW socket, sends a few small packets and closes -
# the way a spam bot opens a fresh SMTP session per message. The source
# port is whatever the OS assigns; we deliberately do not bind a low port.
# =========================================================

def run_beacons(host, port, proto, packets_per_beacon, interval, size, duration):

    payload = (b"CTU13-PATTERN" + b"\x00" * max(0, size - 13))[:size]

    packets = 0
    beacons = 0
    wire_bytes = 0

    # IP(20) + UDP(8) or IP(20) + TCP(20), approximate on-wire overhead.
    header_est = 28 if proto == "udp" else 40

    start = time.perf_counter()
    next_beacon = start

    while time.perf_counter() - start < duration:

        now = time.perf_counter()
        if now < next_beacon:
            time.sleep(min(0.05, next_beacon - now))
            continue

        sent_this_beacon = 0

        if proto == "udp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                for _ in range(packets_per_beacon):
                    if time.perf_counter() - start >= duration:
                        break
                    sock.sendto(payload, (host, port))
                    sent_this_beacon += 1
                    time.sleep(interval / max(packets_per_beacon, 1) * 0.5)
            finally:
                sock.close()

        else:
            # TCP: a connect attempt to a port with no listener puts a real
            # SYN on the wire (and gets an RST back). Non-blocking so we do
            # not stall on the handshake.
            for _ in range(packets_per_beacon):
                if time.perf_counter() - start >= duration:
                    break
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setblocking(False)
                try:
                    s.connect_ex((host, port))
                    sent_this_beacon += 1
                finally:
                    s.close()
                time.sleep(interval / max(packets_per_beacon, 1) * 0.5)

        if sent_this_beacon:
            beacons += 1
            packets += sent_this_beacon
            wire_bytes += sent_this_beacon * (len(payload) + header_est)

        next_beacon += interval

    elapsed = time.perf_counter() - start
    return packets, beacons, wire_bytes, elapsed


# =========================================================
# OFFLINE DIAGNOSTIC (--explain)
#
# Read-only analysis of the EXISTING XGBoost model. Generates no traffic,
# modifies nothing, and asserts no verdict.
# =========================================================

def explain():

    import pandas as pd
    import xgboost as xgb

    model_path = SCRIPT_DIR.parent / "models" / "ctu13_multiscenario_xgboost.json"

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    booster = model.get_booster()
    features = list(booster.feature_names)

    print()
    print("=" * 66)
    print("WHY LIVE OS-SOCKET TRAFFIC SCORES BENIGN (model analysis)")
    print("=" * 66)
    print(f"Model: {model_path.name}  (read-only, unmodified)")

    scores = booster.get_score(importance_type="gain")
    total = sum(scores.values()) or 1.0

    print()
    print("XGBoost feature importance (gain, top 8):")
    for name, value in sorted(scores.items(), key=lambda x: -x[1])[:8]:
        print(f"  {name:24s} {100 * value / total:5.1f}%")

    # Reference vector that the model scores as malicious; sweep only Sport.
    ref = dict(
        Dur=9.121, Sport=339, Dport=65500, sTos=192, dTos=0, TotPkts=53,
        TotBytes=988, SrcBytes=494, PacketsPerSecond=53 / 9.121,
        BytesPerSecond=988 / 9.121, AvgPacketSize=19, DstBytes=494,
        SrcByteRatio=0.5, DstByteRatio=0.5, SourceFlowCount30s=265,
        UniqueDstIPs30s=17, UniqueDstPorts30s=100, UniqueSrcPorts30s=100,
        SourceTotalBytes30s=988 * 265, SourceTotalPackets30s=53 * 265,
        DestinationFlowCount30s=1, UniqueSrcIPs30s=1,
        DestinationTotalBytes30s=494, DestinationRepeatCount=1,
        InterArrivalTime=9.121 / 53, PairInterArrivalTime=9.121 / 53,
        FlowsPerSecond30s=265 / 30, PacketsPerSecond30s=53 / 9.121,
        BytesPerSecond30s=988 / 9.121, SourceOutboundRatio=0.5,
    )

    ports = [80, 443, 1500, 3000, 5000, 8000, 20000, 40000, 49152, 52000, 60000]
    rows = [{**ref, "Sport": p} for p in ports]
    probs = model.predict_proba(pd.DataFrame(rows)[features])[:, 1]

    print()
    print("Sport sensitivity (every other feature held fixed):")
    print(f"  {'Sport':>7}   {'XGB score':>10}   verdict @ 0.20")
    for port, prob in zip(ports, probs):
        verdict = "MALICIOUS" if prob >= 0.20 else "benign"
        note = "  <- range Windows assigns to sockets" if port >= 49152 else ""
        print(f"  {port:>7}   {prob:>10.6f}   {verdict}{note}")

    print()
    print("Sport is the single largest contributor to the model's decision.")
    print("CTU-13 botnets used low / fixed source ports. An ordinary OS socket")
    print("is always given an ephemeral port (49152-65535 on Windows), which")
    print("sits in the model's benign region. This generator does not bind a")
    print("low source port to work around that, so BENIGN is the expected and")
    print("honest result for socket-generated live traffic.")
    print("=" * 66)


# =========================================================
# REPORT
# =========================================================

def print_report(label, host, port, proto, packets, beacons, wire_bytes, elapsed):

    pps = packets / elapsed if elapsed > 0 else 0.0
    mbps = (wire_bytes * 8) / elapsed / 1e6 if elapsed > 0 else 0.0

    print()
    print("=" * 50)
    print("CTU-13 PATTERN VALIDATION")
    print("=" * 50)
    print(f"Pattern     : {label}")
    print(f"Destination : {host}:{port}/{proto}")
    print(f"Duration    : {elapsed:.2f} sec")
    print(f"Packets     : {packets}")
    print(f"Flows       : {beacons}")
    print(f"Packets/sec : {pps:.1f}")
    print(f"Approx Mbps : {mbps:.4f}")
    print("=" * 50)
    print()
    print("NOTE: PCAP replay is not available (no capture files in this repo;")
    print("      replay needs raw injection + Administrator).")
    print("NOTE: This script only generates traffic. The verdict comes from")
    print("      live_capture.py -> /predict -> the trained models. Expect")
    print("      BENIGN: the OS-assigned source port lies outside the range")
    print("      the model associates with CTU-13 botnets (--explain).")


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="CTU-13 pattern validation traffic generator (LOCAL LAB ONLY).",
    )
    parser.add_argument("--pattern", choices=list(PATTERNS.keys()),
                        default="spam-bot")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Destination (loopback/private only).")
    parser.add_argument("--port", type=int, default=None,
                        help="Override the pattern's destination port.")
    parser.add_argument("--proto", choices=["udp", "tcp"], default="udp")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--size", type=int, default=None,
                        help="Payload bytes (default: pattern-specific).")
    parser.add_argument("--packets-per-beacon", type=int, default=None)
    parser.add_argument("--interval", type=float, default=None,
                        help="Seconds between beacons (default 5.0, matching "
                             "live_capture.py's flow window).")
    parser.add_argument("--explain", action="store_true",
                        help="Offline model analysis; generates no traffic.")

    args = parser.parse_args()

    if args.explain:
        explain()
        return

    spec = PATTERNS[args.pattern]

    host = validate_target(args.host)
    port = args.port if args.port is not None else spec["port"]
    size = args.size if args.size is not None else spec["size"]
    ppb = (args.packets_per_beacon if args.packets_per_beacon is not None
           else spec["packets_per_beacon"])
    interval = args.interval if args.interval is not None else spec["interval"]

    if not (0 < port < 65536):
        raise SystemExit(f"[REFUSED] Invalid port {port}.")

    duration = clamp("--duration", args.duration, MAX_DURATION)
    size = clamp("--size", size, MAX_SIZE)

    print_banner()
    print(f"[INFO] Pattern={spec['label']}")
    print(f"[INFO] dest={host}:{port}/{args.proto}  "
          f"packets/beacon={ppb}  interval={interval}s  size={size}B  "
          f"duration={duration}s")
    print("[INFO] Source port: OS-assigned (not bound low - see --explain).")
    print("[INFO] Generating...")

    try:
        packets, beacons, wire_bytes, elapsed = run_beacons(
            host, port, args.proto, ppb, interval, size, duration
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        return

    print_report(spec["label"], host, port, args.proto,
                 packets, beacons, wire_bytes, elapsed)


if __name__ == "__main__":
    main()
