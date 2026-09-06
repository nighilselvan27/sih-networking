"""
CTU-13 HYBRID IDS - CONTROLLED TRAFFIC GENERATOR

  *** LOCAL LAB TEST ONLY ***

Generates controlled traffic toward a LOOPBACK / PRIVATE destination so the
live IDS pipeline can be exercised in real time. It never targets a public
address and never does anything destructive - it only sends packets to a
local test port so that scripts/live_capture.py can observe them.

This does NOT talk to the model or the API. It only puts packets on the wire.
The IDS verdict always comes from the existing trained models via /predict.

Traffic types
-------------
    benign-udp   Steady, moderate-rate UDP to a local port.       (default)
    benign-tcp   A real TCP connection to a local echo listener.
    udp          High-rate UDP flood to a local port.
    syn          Rapid TCP SYNs (connect attempts) to a closed local port.
    burst        High-rate but benign multi-socket UDP burst.

Examples
--------
    python scripts/test_traffic.py
    python scripts/test_traffic.py --type udp --duration 10 --rate 500 --port 9999
    python scripts/test_traffic.py --type syn --duration 5 --rate 800
    python scripts/test_traffic.py --type benign-tcp --count 50 --size 512

Run with no arguments it reproduces the original behaviour:
benign UDP, 5000 packets, 127.0.0.1:9999.
"""

import argparse
import ipaddress
import socket
import sys
import threading
import time


# =========================================================
# SAFETY LIMITS
# =========================================================

MAX_RATE = 20000        # packets per second
MAX_DURATION = 300      # seconds
MAX_COUNT = 2_000_000   # packets
MAX_SIZE = 65507        # max UDP payload for IPv4

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9999


# =========================================================
# TARGET VALIDATION
#
# Only loopback or RFC1918 / private addresses are allowed.
# Public, multicast, broadcast and reserved targets are refused.
# =========================================================

def validate_target(host):

    try:
        ip = ipaddress.ip_address(host)

    except ValueError:
        # Allow hostnames only if they resolve to a local/private address.
        try:
            resolved = socket.gethostbyname(host)
            ip = ipaddress.ip_address(resolved)
            print(f"[INFO] Resolved {host} -> {resolved}")

        except Exception as exc:
            raise SystemExit(
                f"[REFUSED] Could not resolve host '{host}': {exc}"
            )

    if ip.is_loopback or ip.is_private:
        return str(ip)

    raise SystemExit(
        f"[REFUSED] Destination {ip} is not loopback or private.\n"
        f"          This is a LOCAL LAB TEST tool and will not target "
        f"public / routable addresses."
    )


def clamp(name, value, maximum):

    if value is None:
        return None

    if value <= 0:
        raise SystemExit(f"[REFUSED] {name} must be positive.")

    if value > maximum:
        print(
            f"[WARN] {name}={value} exceeds the safety cap "
            f"{maximum}; clamping to {maximum}."
        )
        return maximum

    return value


# =========================================================
# RATE-PACED SENDER
#
# A deadline accumulator keeps the long-run rate accurate instead of
# drifting the way sleep(1/rate) would.
# =========================================================

def paced_loop(rate, count, duration, send_one):
    """
    Call send_one() at approximately `rate` packets/sec until either
    `count` packets are sent or `duration` seconds elapse (whichever
    comes first). Returns (packets_sent, elapsed_seconds).

    rate=None means "as fast as possible".
    """

    sent = 0
    start = time.perf_counter()

    interval = (1.0 / rate) if rate else 0.0
    next_send = start

    while True:

        now = time.perf_counter()
        elapsed = now - start

        if count is not None and sent >= count:
            break

        if duration is not None and elapsed >= duration:
            break

        send_one()
        sent += 1

        if interval:
            next_send += interval
            sleep_for = next_send - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

    elapsed = time.perf_counter() - start
    return sent, elapsed


# =========================================================
# TRAFFIC TYPES
# =========================================================

def run_benign_udp(host, port, count, rate, size, duration):

    payload = b"CTU13-BENIGN-UDP" + b"\x00" * max(0, size - 16)
    payload = payload[:size]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_one():
        sock.sendto(payload, (host, port))

    try:
        sent, elapsed = paced_loop(rate, count, duration, send_one)
    finally:
        sock.close()

    return sent, elapsed, len(payload), 28  # IP(20)+UDP(8)


def run_udp_flood(host, port, count, rate, size, duration):

    payload = b"CTU13-UDP-FLOOD" + b"\x00" * max(0, size - 15)
    payload = payload[:size]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_one():
        sock.sendto(payload, (host, port))

    try:
        sent, elapsed = paced_loop(rate, count, duration, send_one)
    finally:
        sock.close()

    return sent, elapsed, len(payload), 28


def run_burst(host, port, count, rate, size, duration):
    """
    High-rate but benign: spread UDP across a few sockets and ports so the
    IDS sees several concurrent high-rate flows. Still ordinary UDP data.
    """

    payload = b"CTU13-BURST" + b"\x00" * max(0, size - 11)
    payload = payload[:size]

    n_sockets = 4
    socks = [
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for _ in range(n_sockets)
    ]
    ports = [port + i for i in range(n_sockets)]

    state = {"i": 0}

    def send_one():
        i = state["i"] % n_sockets
        socks[i].sendto(payload, (host, ports[i]))
        state["i"] += 1

    try:
        sent, elapsed = paced_loop(rate, count, duration, send_one)
    finally:
        for s in socks:
            s.close()

    return sent, elapsed, len(payload), 28


def run_syn(host, port, count, rate, size, duration):
    """
    Controlled TCP SYN test.

    A non-blocking connect() to a port with no listener makes the OS emit a
    real SYN packet. Each attempt uses a fresh ephemeral source port, so the
    IDS observes many tiny short-lived unidirectional flows - the flow-level
    signature of a SYN scan/flood.

    No source-IP spoofing (that would require raw sockets / admin); every SYN
    genuinely originates from this host.
    """

    def send_one():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setblocking(False)
        try:
            s.connect_ex((host, port))   # fires the SYN, returns immediately
        finally:
            # Close right away; we only wanted the SYN on the wire.
            s.close()

    sent, elapsed = paced_loop(rate, count, duration, send_one)

    # SYN packets carry no payload; header estimate IP(20)+TCP(20).
    return sent, elapsed, 0, 40


def run_benign_tcp(host, port, count, rate, size, duration):
    """
    A genuine, benign TCP conversation against a local echo listener started
    by this script. Sends `count` messages of `size` bytes and reads the
    echoes back, then closes cleanly.
    """

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(1)

    stop = threading.Event()

    def echo_server():
        listener.settimeout(1.0)
        try:
            conn, _ = listener.accept()
        except Exception:
            return
        conn.settimeout(1.0)
        with conn:
            while not stop.is_set():
                try:
                    data = conn.recv(65536)
                except socket.timeout:
                    continue
                except Exception:
                    break
                if not data:
                    break
                try:
                    conn.sendall(data)
                except Exception:
                    break

    server_thread = threading.Thread(target=echo_server, daemon=True)
    server_thread.start()

    time.sleep(0.2)  # let the listener come up

    payload = b"CTU13-BENIGN-TCP" + b"\x00" * max(0, size - 16)
    payload = payload[:size]

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(3.0)
    client.connect((host, port))

    def send_one():
        client.sendall(payload)
        try:
            client.recv(len(payload))
        except socket.timeout:
            pass

    try:
        sent, elapsed = paced_loop(rate, count, duration, send_one)
    finally:
        try:
            client.close()
        finally:
            stop.set()
            listener.close()
            server_thread.join(timeout=2.0)

    return sent, elapsed, len(payload), 40


# =========================================================
# DISPATCH
# =========================================================

RUNNERS = {
    "benign-udp": ("BENIGN UDP", run_benign_udp),
    "benign-tcp": ("BENIGN TCP", run_benign_tcp),
    "udp":        ("UDP FLOOD", run_udp_flood),
    "syn":        ("TCP SYN", run_syn),
    "burst":      ("HIGH-RATE BENIGN BURST", run_burst),
}


def print_banner():
    print("=" * 50)
    print("  LOCAL LAB TEST ONLY - loopback / private targets only")
    print("=" * 50)


def print_report(label, host, port, sent, elapsed, payload_size, header_est):

    pps = sent / elapsed if elapsed > 0 else 0.0
    total_bytes = sent * (payload_size + header_est)
    mbps = (total_bytes * 8) / elapsed / 1e6 if elapsed > 0 else 0.0

    print()
    print("=" * 50)
    print("TRAFFIC TEST")
    print("=" * 50)
    print(f"Type        : {label}")
    print(f"Destination : {host}:{port}")
    print(f"Packets     : {sent}")
    print(f"Duration    : {elapsed:.2f} sec")
    print(f"Packets/sec : {pps:.1f}")
    print(f"Throughput  : {mbps:.2f} Mbps (approx, incl. ~{header_est}B header/pkt)")
    print("=" * 50)


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Controlled LOCAL-ONLY traffic generator for the CTU-13 IDS.",
    )

    parser.add_argument(
        "--type",
        choices=list(RUNNERS.keys()),
        default="benign-udp",
        help="Traffic type (default: benign-udp).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Destination (loopback/private only).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--count", type=int, default=None,
                        help="Number of packets to send.")
    parser.add_argument("--rate", type=float, default=None,
                        help="Target packets/sec (default: as fast as possible).")
    parser.add_argument("--size", type=int, default=64,
                        help="Payload size in bytes (default: 64).")
    parser.add_argument("--duration", type=float, default=None,
                        help="Max seconds to run.")

    args = parser.parse_args()

    # Preserve the original no-argument behaviour:
    # benign UDP, 5000 packets, 127.0.0.1:9999.
    if len(sys.argv) == 1:
        args.count = 5000
        args.size = 15

    host = validate_target(args.host)

    if not (0 < args.port < 65536):
        raise SystemExit(f"[REFUSED] Invalid port {args.port}.")

    rate = clamp("--rate", args.rate, MAX_RATE)
    duration = clamp("--duration", args.duration, MAX_DURATION)
    count = clamp("--count", args.count, MAX_COUNT)
    size = clamp("--size", args.size, MAX_SIZE)

    if count is None and duration is None:
        # Bounded default so the tool always terminates.
        count = 5000

    label, runner = RUNNERS[args.type]

    print_banner()
    print(f"[INFO] Type={label}  dest={host}:{args.port}  "
          f"count={count}  rate={rate}  size={size}  duration={duration}")
    print("[INFO] Starting...")

    try:
        sent, elapsed, payload_size, header_est = runner(
            host, args.port, count, rate, size, duration
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        return

    print_report(label, host, args.port, sent, elapsed, payload_size, header_est)


if __name__ == "__main__":
    main()
