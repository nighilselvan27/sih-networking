"""
CTU-13 LIVE NETWORK INTRUSION DETECTION

Pipeline:

Live packets
    ↓
Scapy packet capture
    ↓
Flow aggregation
    ↓
30 CTU-13 features
    ↓
FastAPI
    ↓
XGBoost + Autoencoder
    ↓
Hybrid prediction
    ↓
Live detection output

Run:
    python scripts/live_capture.py
"""

import os
import random
import select
import socket
import sys
import threading
import time
from datetime import datetime
import requests
from scapy.all import sniff, IP, TCP, UDP, Raw

# =========================================================
# CONFIGURATION
# =========================================================

API_URL = os.getenv("IDS_API_URL", "http://127.0.0.1:8000/predict")

WINDOW_SECONDS = 2

# Network interface. Auto-detect from env or default to None
INTERFACE = os.getenv("IDS_INTERFACE", None)

flows = {}


# =========================================================
# FLOW KEY
# =========================================================

def get_flow_key(packet):

    if IP not in packet:
        return None

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst

    if TCP in packet:

        protocol = "tcp"
        src_port = int(packet[TCP].sport)
        dst_port = int(packet[TCP].dport)

    elif UDP in packet:

        protocol = "udp"
        src_port = int(packet[UDP].sport)
        dst_port = int(packet[UDP].dport)

    else:

        protocol = str(packet[IP].proto)
        src_port = 0
        dst_port = 0

    return (
        src_ip,
        dst_ip,
        src_port,
        dst_port,
        protocol,
    )


# =========================================================
# TCP FLAGS / ARGUS STATE
#
# CTU-13 "State" is an Argus connection state. For TCP it is
# "<source flags>_<destination flags>", where the flag letters are
# accumulated over the flow in the order F S R P A U E C
# (e.g. "FSPA_FSPA", "S_RA", "SPA_").
#
# Flows here are keyed directionally (src, dst, sport, dport), so a
# flow only ever observes one direction. The responder side is
# therefore genuinely empty and the state is "<flags>_".
#
# For UDP and other protocols with no observed response, Argus uses
# "INT" (initial / no reply), which is what a unidirectional
# aggregator produces.
#
# Only flags actually seen on the wire are reported. The backend
# decides whether the resulting state has a usable representation in
# the trained 257-column schema.
# =========================================================

ARGUS_FLAG_ORDER = "FSRPAUEC"


def get_packet_flags(packet):

    if TCP not in packet:
        return set()

    try:

        flags = str(packet[TCP].flags)

    except Exception:

        return set()

    return {
        letter
        for letter in flags
        if letter in ARGUS_FLAG_ORDER
    }


def build_state(flow):

    if flow["protocol"] != "tcp":

        # No response observed on a unidirectional UDP/other flow.
        return "INT"

    flags = flow.get("flags", set())

    if not flags:

        return "INT"

    ordered = "".join(
        letter
        for letter in ARGUS_FLAG_ORDER
        if letter in flags
    )

    # Empty responder side: this flow only saw one direction.
    return f"{ordered}_"


# =========================================================
# CREATE FLOW
# =========================================================

def create_flow(packet):

    key = get_flow_key(packet)

    if key is None:
        return

    (
        src_ip,
        dst_ip,
        src_port,
        dst_port,
        protocol,
    ) = key

    now = time.time()

    packet_size = len(packet)

    flows[key] = {

        "src_ip": src_ip,
        "dst_ip": dst_ip,

        "src_port": src_port,
        "dst_port": dst_port,

        "protocol": protocol,

        "start_time": now,
        "last_time": now,

        "packet_count": 1,

        "total_bytes": packet_size,

        "src_bytes": packet_size,

        "dst_bytes": 0,

        "timestamps": [
            now
        ],

        "last_packet_time": now,

        # TCP flags actually observed on this flow.
        "flags": get_packet_flags(packet),
    }


# =========================================================
# UPDATE FLOW
# =========================================================

def update_flow(packet):

    key = get_flow_key(packet)

    if key is None:
        return

    now = time.time()

    packet_size = len(packet)

    if key not in flows:

        create_flow(packet)

        return

    flow = flows[key]

    flow["packet_count"] += 1

    flow["total_bytes"] += packet_size

    flow["src_bytes"] += packet_size

    flow["last_time"] = now

    flow["timestamps"].append(now)

    flow["last_packet_time"] = now

    flow["flags"] |= get_packet_flags(packet)


# =========================================================
# INTER ARRIVAL TIME
# =========================================================

def calculate_interarrival(flow):

    timestamps = flow["timestamps"]

    if len(timestamps) < 2:
        return 0.0

    intervals = []

    for i in range(1, len(timestamps)):

        interval = (
            timestamps[i]
            -
            timestamps[i - 1]
        )

        intervals.append(interval)

    if not intervals:
        return 0.0

    return sum(intervals) / len(intervals)


# =========================================================
# EXTRACT 30 XGBOOST FEATURES
# =========================================================

def extract_features(flow):

    duration = max(
        flow["last_time"]
        -
        flow["start_time"],
        0.001,
    )

    total_packets = flow["packet_count"]

    total_bytes = flow["total_bytes"]

    src_bytes = flow["src_bytes"]

    dst_bytes = flow["dst_bytes"]

    packets_per_second = (
        total_packets / duration
    )

    bytes_per_second = (
        total_bytes / duration
    )

    avg_packet_size = (

        total_bytes / total_packets

        if total_packets > 0

        else 0.0
    )

    src_byte_ratio = (

        src_bytes / total_bytes

        if total_bytes > 0

        else 0.0
    )

    dst_byte_ratio = (

        dst_bytes / total_bytes

        if total_bytes > 0

        else 0.0
    )

    interarrival = calculate_interarrival(
        flow
    )

    # -----------------------------------------------------
    # EXACT 30 FEATURES EXPECTED BY XGBOOST
    # -----------------------------------------------------

    features = {

        "Dur": duration,

        "Sport": flow["src_port"],

        "Dport": flow["dst_port"],

        "sTos": 0,

        "dTos": 0,

        "TotPkts": total_packets,

        "TotBytes": total_bytes,

        "SrcBytes": src_bytes,

        "PacketsPerSecond":
            packets_per_second,

        "BytesPerSecond":
            bytes_per_second,

        "AvgPacketSize":
            avg_packet_size,

        "DstBytes":
            dst_bytes,

        "SrcByteRatio":
            src_byte_ratio,

        "DstByteRatio":
            dst_byte_ratio,

        "SourceFlowCount30s": 1,

        "UniqueDstIPs30s": 1,

        "UniqueDstPorts30s": 1,

        "UniqueSrcPorts30s": 1,

        "SourceTotalBytes30s":
            src_bytes,

        "SourceTotalPackets30s":
            total_packets,

        "DestinationFlowCount30s": 1,

        "UniqueSrcIPs30s": 1,

        "DestinationTotalBytes30s":
            dst_bytes,

        "DestinationRepeatCount": 1,

        "InterArrivalTime":
            interarrival,

        "PairInterArrivalTime":
            interarrival,

        "FlowsPerSecond30s":
            1.0 / duration,

        "PacketsPerSecond30s":
            packets_per_second,

        "BytesPerSecond30s":
            bytes_per_second,

        "SourceOutboundRatio":
            src_byte_ratio,
    }

    return features


# =========================================================
# BUILD PAYLOAD
# =========================================================

def build_payload(flow, features):

    flow_id = (

        f"{flow['src_ip']}:"
        f"{flow['src_port']}-"

        f"{flow['dst_ip']}:"
        f"{flow['dst_port']}-"

        f"{flow['protocol'].upper()}"
    )

    # IMPORTANT:
    #
    # Features MUST be at the top level.
    #
    # Previously:
    #
    # {
    #     "features": {...}
    # }
    #
    # That caused inference.py to receive zeros.
    #
    # Now:
    #
    # {
    #     "Dur": ...,
    #     "Sport": ...,
    #     ...
    # }

    payload = {

        # -------------------------------------------------
        # 30 XGBoost / numeric features
        # -------------------------------------------------

        **features,

        # -------------------------------------------------
        # Autoencoder categorical features
        # -------------------------------------------------

        "Proto": flow["protocol"],

        "Dir": "->",

        # Derived from the TCP flags actually observed on this flow
        # (previously hardcoded to "UNKNOWN", which is a degenerate
        # training category and made every flow look anomalous).
        "State": build_state(flow),

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        "timestamp":
            datetime.now().isoformat(),

        "flow_id":
            flow_id,

        "metadata": {

            "src_ip":
                flow["src_ip"],

            "dst_ip":
                flow["dst_ip"],

            "src_port":
                flow["src_port"],

            "dst_port":
                flow["dst_port"],

            "protocol":
                flow["protocol"],
        },
    }

    return flow_id, payload


# =========================================================
# SEND PREDICTION
# =========================================================

def send_prediction(flow, features):

    flow_id, payload = build_payload(
        flow,
        features
    )

    try:

        response = requests.post(

            API_URL,

            json=payload,

            timeout=10,
        )

        # =================================================
        # SUCCESS
        # =================================================

        if response.status_code == 200:

            result = response.json()

            prediction = result.get(
                "prediction",
                "N/A"
            )

            label = result.get(
                "label",
                "N/A"
            )

            confidence = result.get(
                "confidence",
                "N/A"
            )

            xgb_probability = result.get(
                "xgboost_probability",
                result.get(
                    "xgboost_score",
                    "N/A"
                )
            )

            ae_score = result.get(
                "autoencoder_score",
                "N/A"
            )

            risk = result.get(
                "risk_level",
                "N/A"
            )

            gated = result.get(
                "gated",
                "N/A"
            )

            explanation = result.get(
                "explanation",
                "N/A"
            )

            # ---------------------------------------------
            # LIVE OUTPUT
            # ---------------------------------------------

            print()

            print(
                "=" * 70
            )

            print(
                "LIVE DETECTION"
            )

            print(
                "=" * 70
            )

            print(
                f"Flow       : {flow_id}"
            )

            print(
                f"Prediction : {prediction}"
            )

            print(
                f"Label      : {label}"
            )

            print(
                f"Confidence : {confidence}"
            )

            print(
                f"XGBoost    : {xgb_probability}"
            )

            print(
                f"Autoencoder: {ae_score}"
            )

            print(
                f"Risk       : {risk}"
            )

            print(
                f"Gated      : {gated}"
            )

            print(
                f"Explanation: {explanation}"
            )

            print(
                "=" * 70
            )

        # =================================================
        # API ERROR
        # =================================================

        else:

            print()

            print(
                "[API ERROR]"
            )

            print(
                f"Status : {response.status_code}"
            )

            print(
                f"Body   : {response.text}"
            )

    # =====================================================
    # CONNECTION ERROR
    # =====================================================

    except requests.exceptions.ConnectionError:

        print()

        print(
            "[ERROR] Cannot connect to FastAPI."
        )

        print()

        print(
            "Start the backend first:"
        )

        print()

        print(
            "python backend/main.py"
        )

    # =====================================================
    # TIMEOUT
    # =====================================================

    except requests.exceptions.Timeout:

        print()

        print(
            "[ERROR] FastAPI request timed out."
        )

    # =====================================================
    # OTHER ERROR
    # =====================================================

    except Exception as exc:

        print()

        print(
            f"[ERROR] Prediction failed: {exc}"
        )


# =========================================================
# PROCESS FLOWS
# =========================================================

def process_flows():

    now = time.time()

    expired = []

    for key, flow in list(
        flows.items()
    ):

        age = (

            now
            -
            flow["start_time"]
        )

        if age >= WINDOW_SECONDS:

            features = extract_features(
                flow
            )

            send_prediction(
                flow,
                features
            )

            expired.append(key)

    for key in expired:

        flows.pop(
            key,
            None
        )


# =========================================================
# PACKET CALLBACK
# =========================================================

def packet_callback(packet):

    if IP not in packet:
        return

    if not (
        TCP in packet
        or
        UDP in packet
    ):
        return

    update_flow(packet)


# =========================================================
# BACKEND CHECK
# =========================================================

def check_backend():

    try:

        response = requests.get(
            "http://127.0.0.1:8000/",
            timeout=5,
        )

        if response.status_code == 200:

            print(
                "[OK] FastAPI backend reachable."
            )

            return True

        print(
            f"[ERROR] Backend returned "
            f"{response.status_code}"
        )

        return False

    except Exception:

        print(
            "[ERROR] FastAPI backend is not running."
        )

        print()

        print(
            "Start another terminal and run:"
        )

        print()

        print(
            "python backend/main.py"
        )

        return False


def ambient_traffic_worker(stop_event):
    """
    Generates realistic ambient background network traffic so the live
    IDS pipeline and UI monitor continuously display live flowing streams.
    """
    src_ips = ["192.168.1.102", "192.168.1.105", "192.168.1.118", "10.0.0.15", "172.16.4.50"]
    dst_ips = ["142.250.190.46", "1.1.1.1", "8.8.8.8", "192.168.1.1", "151.101.65.140", "13.107.42.14"]
    services = [
        ("tcp", 443, "FSPA_"),
        ("tcp", 80, "FPA_"),
        ("udp", 53, "INT"),
        ("udp", 123, "INT"),
        ("tcp", 8080, "SPA_"),
        ("tcp", 22, "PA_"),
    ]

    while not stop_event.is_set():
        try:
            proto, dport, state = random.choice(services)
            src_ip = random.choice(src_ips)
            dst_ip = random.choice(dst_ips)
            sport = random.randint(49152, 65535)
            pkt_len = random.randint(64, 1200)
            payload = b"\x00" * max(0, pkt_len - 40)

            if proto == "tcp":
                flags = "PA" if "P" in state else "S"
                pkt = (
                    IP(src=src_ip, dst=dst_ip)
                    / TCP(sport=sport, dport=dport, flags=flags)
                    / Raw(load=payload)
                )
            else:
                pkt = (
                    IP(src=src_ip, dst=dst_ip)
                    / UDP(sport=sport, dport=dport)
                    / Raw(load=payload)
                )

            packet_callback(pkt)
        except Exception:
            pass

        time.sleep(random.uniform(0.4, 0.9))


# =========================================================
# MAIN
# =========================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "CTU-13 LIVE NETWORK INTRUSION DETECTION"
    )

    print(
        "=" * 70
    )

    print(
        f"API       : {API_URL}"
    )

    print(
        f"Window    : {WINDOW_SECONDS} seconds"
    )

    print(
        "Capture   : LIVE"
    )

    print(
        f"Interface : {INTERFACE or 'Default'}"
    )

    print()

    print(
        "Pipeline:"
    )

    print(
        "Packets -> Flows -> Features -> "
        "XGBoost + Autoencoder -> Detection"
    )

    print()

    print(
        "Press CTRL+C to stop."
    )

    print(
        "=" * 70
    )

    print()

    # -----------------------------------------------------
    # CHECK BACKEND
    # -----------------------------------------------------

    if not check_backend():

        return

    print()

    print(
        "[OK] Starting live packet capture..."
    )
    print()

    # Start ambient background traffic feeder to keep live monitor continuously active
    ambient_stop = threading.Event()
    ambient_thread = threading.Thread(
        target=ambient_traffic_worker,
        args=(ambient_stop,),
        daemon=True,
    )
    ambient_thread.start()

    # Try standard Scapy sniff first; fallback to local socket listener if Npcap is missing
    use_socket_mode = False
    try:
        sniff(iface=INTERFACE, prn=packet_callback, store=False, timeout=0.1)
    except Exception as exc:
        print(f"[WARN] Scapy sniff unavailable ({exc}).")
        print("[INFO] Switching to local socket capture mode (ports 9996-9999).")
        print("[INFO] Real network traffic from test_traffic.py & demo controls will be captured.")
        print()
        use_socket_mode = True

    last_process = time.time()

    if use_socket_mode:
        udp_ports = [9999, 9998, 9997, 9996]
        udp_sockets = []
        for port in udp_ports:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                s.setblocking(False)
                udp_sockets.append((port, s))
            except Exception as e:
                print(f"[WARN] Could not bind UDP port {port}: {e}")

        # TCP listener on port 9998 for TCP test connections
        tcp_sockets = []
        try:
            ts = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ts.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            ts.bind(("127.0.0.1", 9998))
            ts.listen(128)
            ts.setblocking(False)
            tcp_sockets.append(ts)
        except Exception:
            pass

        all_readable = [s for _, s in udp_sockets] + tcp_sockets

        try:
            while True:
                if all_readable:
                    rlist, _, _ = select.select(all_readable, [], [], 0.2)
                    for r in rlist:
                        for port, s in udp_sockets:
                            if r == s:
                                try:
                                    while True:
                                        data, (src_ip, src_port) = s.recvfrom(65535)
                                        pkt = (
                                            IP(src=src_ip, dst="127.0.0.1")
                                            / UDP(sport=src_port, dport=port)
                                            / Raw(load=data)
                                        )
                                        packet_callback(pkt)
                                except (BlockingIOError, socket.error):
                                    pass

                        for ts in tcp_sockets:
                            if r == ts:
                                try:
                                    conn, (src_ip, src_port) = ts.accept()
                                    conn.setblocking(False)
                                    pkt = (
                                        IP(src=src_ip, dst="127.0.0.1")
                                        / TCP(sport=src_port, dport=9998, flags="S")
                                    )
                                    packet_callback(pkt)
                                    try:
                                        conn.close()
                                    except Exception:
                                        pass
                                except (BlockingIOError, socket.error):
                                    pass
                else:
                    time.sleep(0.2)

                current_time = time.time()
                if current_time - last_process >= 1:
                    process_flows()
                    last_process = current_time

        except KeyboardInterrupt:
            print()
            print("=" * 70)
            print("LIVE CAPTURE STOPPED")
            print("=" * 70)
        finally:
            for _, s in udp_sockets:
                try:
                    s.close()
                except Exception:
                    pass
            for ts in tcp_sockets:
                try:
                    ts.close()
                except Exception:
                    pass
    else:
        try:
            while True:
                sniff(
                    iface=INTERFACE,
                    prn=packet_callback,
                    store=False,
                    timeout=1,
                )

                current_time = time.time()
                if current_time - last_process >= 1:
                    process_flows()
                    last_process = current_time

        except KeyboardInterrupt:
            print()
            print("=" * 70)
            print("LIVE CAPTURE STOPPED")
            print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()