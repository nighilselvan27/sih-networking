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

import time
import requests

from datetime import datetime
from scapy.all import sniff, IP, TCP, UDP


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000/predict"

WINDOW_SECONDS = 5

# Network interface.
#
# Leave as None first.
# If Scapy cannot capture traffic, we can specify
# the Npcap interface later.
INTERFACE = r"\Device\NPF_Loopback"

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

    last_process = time.time()

    try:

        while True:

            sniff(

                iface=INTERFACE,

                prn=packet_callback,

                store=False,

                timeout=1,
            )

            current_time = time.time()

            if (
                current_time
                -
                last_process
                >= 1
            ):

                process_flows()

                last_process = (
                    current_time
                )

    except KeyboardInterrupt:

        print()

        print(
            "=" * 70
        )

        print(
            "LIVE CAPTURE STOPPED"
        )

        print(
            "=" * 70
        )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()