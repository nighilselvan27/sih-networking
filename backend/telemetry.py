"""
Read-side telemetry for the CTU-13 Hybrid IDS.

The detection pipeline is unchanged by this module. `record()` is called
AFTER `inference.predict()` has already produced a verdict, and it only
stores what the request and the verdict already contain. Nothing here
computes, alters or second-guesses a prediction.

Why this exists: the API is stateless. `scripts/live_capture.py` POSTs each
expired flow window to /predict, prints the result and discards it. An
operator console has nothing to read. This module keeps a bounded in-memory
history of what the API has already scored, so the UI can show it.

Everything is in-memory and volatile. Restarting the API clears it. There is
no database and no file written by this module.

Stdlib only.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, Dict, Iterable, List, Optional, Tuple
import queue
import threading
import time


# ============================================================
# CONFIGURATION
# ============================================================

# Number of scored flows retained. At the observed live rate (one record
# per flow per 5-second capture window) this is a deep history; under the
# throughput harness it is a few minutes. Oldest records are evicted.
MAX_RECORDS = 5000

# One second of resolution, one hour of depth.
MAX_BUCKETS = 3600

# A subscriber that cannot keep up loses events rather than stalling the
# capture path. Dropping frames in the UI is acceptable; blocking the
# prediction request is not.
SUBSCRIBER_QUEUE_SIZE = 256

# Rolling window used by /api/stats.
STATS_WINDOW_SECONDS = 60

# (window_seconds, bucket_seconds) per chart range. Each yields 60 points.
RANGE_SPEC: Dict[str, Tuple[int, int]] = {
    "1m": (60, 1),
    "5m": (300, 5),
    "15m": (900, 15),
    "1h": (3600, 60),
}

DEFAULT_RANGE = "5m"


# ============================================================
# STATE
# ============================================================

_lock = threading.Lock()

_records: deque = deque(maxlen=MAX_RECORDS)

# {epoch_second: {flows, packets, bytes, threats, xgb_alerts, ae_alerts}}
_buckets: "deque[Tuple[int, Dict[str, float]]]" = deque(maxlen=MAX_BUCKETS)

_subscribers: set = set()
_subscribers_lock = threading.Lock()

_seq = 0

# Totals since process start. These are not windowed and never evicted.
_totals = {
    "flows": 0,
    "threats": 0,
    "packets": 0.0,
    "bytes": 0.0,
    "xgb_alerts": 0,
    "ae_alerts": 0,
    "errors": 0,
}

_started_at = time.time()
_last_record_at: Optional[float] = None


# ============================================================
# COERCION HELPERS
# ============================================================

def _as_float(value: Any, default: float = 0.0) -> float:

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    # NaN / inf would poison every downstream sum and serialise as invalid
    # JSON. Treat them as missing.
    if result != result or result in (float("inf"), float("-inf")):
        return default

    return result


def _as_int(value: Any, default: int = 0) -> int:

    return int(_as_float(value, default))


def _as_str(value: Any, default: str = "") -> str:

    if value is None:
        return default

    text = str(value).strip()

    return text if text else default


def _parse_flow_id(flow_id: str) -> Dict[str, Any]:
    """
    Split the identifier live_capture.py builds:

        "<src_ip>:<sport>-<dst_ip>:<dport>-<PROTO>"

    Used only as a last resort, when neither `metadata` nor SrcAddr/DstAddr
    were supplied. Returns empty values rather than raising.
    """

    empty = {
        "src_ip": "",
        "dst_ip": "",
        "src_port": 0,
        "dst_port": 0,
        "protocol": "",
    }

    try:
        rest, protocol = flow_id.rsplit("-", 1)
        left, right = rest.split("-", 1)

        src_ip, src_port = left.rsplit(":", 1)
        dst_ip, dst_port = right.rsplit(":", 1)

        return {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": _as_int(src_port),
            "dst_port": _as_int(dst_port),
            "protocol": protocol.lower(),
        }

    except (ValueError, AttributeError):
        return empty


# ============================================================
# RECORD ASSEMBLY
# ============================================================

def _build_record(
    request_data: Dict[str, Any],
    result: Dict[str, Any],
    seq: int,
    received_at: float,
) -> Dict[str, Any]:
    """
    Assemble one stored record from the request that was scored and the
    verdict that was returned. Every value is copied; none is derived.
    """

    source = request_data if isinstance(request_data, dict) else {}
    verdict = result if isinstance(result, dict) else {}

    metadata = source.get("metadata") or {}

    if not isinstance(metadata, dict):
        metadata = {}

    flow_id = _as_str(verdict.get("flow_id") or source.get("flow_id"))

    # Identity, in the same precedence order inference.predict() uses when
    # it rebuilds flow_id (inference.py:1021-1049).
    src_ip = _as_str(metadata.get("src_ip") or source.get("SrcAddr"))
    dst_ip = _as_str(metadata.get("dst_ip") or source.get("DstAddr"))
    src_port = _as_int(metadata.get("src_port", source.get("Sport")))
    dst_port = _as_int(metadata.get("dst_port", source.get("Dport")))
    protocol = _as_str(
        metadata.get("protocol") or source.get("Proto")
    ).lower()

    if not (src_ip and dst_ip) and flow_id:

        parsed = _parse_flow_id(flow_id)

        src_ip = src_ip or parsed["src_ip"]
        dst_ip = dst_ip or parsed["dst_ip"]
        src_port = src_port or parsed["src_port"]
        dst_port = dst_port or parsed["dst_port"]
        protocol = protocol or parsed["protocol"]

    prediction = _as_int(verdict.get("prediction"))

    return {
        "seq": seq,
        "received_at": received_at,

        # ----------------------------------------------------
        # Identity (from the request)
        # ----------------------------------------------------
        "flow_id": flow_id,
        "src_ip": src_ip or "unknown",
        "dst_ip": dst_ip or "unknown",
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": protocol or "unknown",
        "direction": _as_str(source.get("Dir"), "->"),
        "state": _as_str(source.get("State")),

        # ----------------------------------------------------
        # Volume and timing (from the request)
        # ----------------------------------------------------
        "packets": _as_float(source.get("TotPkts")),
        "bytes": _as_float(source.get("TotBytes")),
        "src_bytes": _as_float(source.get("SrcBytes")),
        "dst_bytes": _as_float(source.get("DstBytes")),
        "duration": _as_float(source.get("Dur")),
        "packets_per_second": _as_float(source.get("PacketsPerSecond")),
        "bytes_per_second": _as_float(source.get("BytesPerSecond")),
        "avg_packet_size": _as_float(source.get("AvgPacketSize")),

        # ----------------------------------------------------
        # Verdict (from inference.predict(), verbatim)
        # ----------------------------------------------------
        "timestamp": _as_str(verdict.get("timestamp")),
        "prediction": prediction,
        "label": _as_str(verdict.get("label")),
        "threat_class": _as_str(verdict.get("threat_class")),
        "risk": _as_str(verdict.get("risk") or verdict.get("risk_level")),
        "confidence": _as_float(verdict.get("confidence")),
        "xgboost_score": _as_float(
            verdict.get("xgboost_score", verdict.get("xgboost_probability"))
        ),
        "autoencoder_score": _as_float(verdict.get("autoencoder_score")),
        "autoencoder_normalized": _as_float(
            verdict.get("autoencoder_normalized")
        ),
        "hybrid_score": _as_float(verdict.get("hybrid_score")),
        "gated": bool(verdict.get("gated")),
        "xgboost_malicious": bool(
            verdict.get("xgboost_malicious", verdict.get("xgboost_anomaly"))
        ),
        "autoencoder_anomalous": bool(
            verdict.get(
                "autoencoder_anomalous", verdict.get("autoencoder_anomaly")
            )
        ),
        "explanation": _as_str(verdict.get("explanation")),
        "supporting_features": verdict.get("supporting_features") or {},
        "evidence": verdict.get("evidence") or {},
        "details": verdict.get("details") or {},

        # ----------------------------------------------------
        # Operator state (this module only)
        # ----------------------------------------------------
        "acknowledged": False,
    }


# ============================================================
# BUCKETS
# ============================================================

def _empty_bucket() -> Dict[str, float]:

    return {
        "flows": 0.0,
        "packets": 0.0,
        "bytes": 0.0,
        "threats": 0.0,
        "xgb_alerts": 0.0,
        "ae_alerts": 0.0,
    }


def _accumulate_bucket(record: Dict[str, Any], received_at: float) -> None:
    """Caller must hold _lock."""

    second = int(received_at)

    if _buckets and _buckets[-1][0] == second:
        bucket = _buckets[-1][1]

    else:
        bucket = _empty_bucket()
        _buckets.append((second, bucket))

    bucket["flows"] += 1.0
    bucket["packets"] += record["packets"]
    bucket["bytes"] += record["bytes"]

    if record["prediction"] == 1:
        bucket["threats"] += 1.0

    if record["xgboost_malicious"]:
        bucket["xgb_alerts"] += 1.0

    if record["autoencoder_anomalous"]:
        bucket["ae_alerts"] += 1.0


# ============================================================
# PUBLIC: WRITE PATH
# ============================================================

def record(request_data: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Store one already-scored flow.

    Called from the /predict route AFTER inference.predict() returns. This
    function must never raise into the request path: a telemetry failure
    must not turn a successful prediction into a 500.
    """

    global _seq, _last_record_at

    try:
        received_at = time.time()

        with _lock:
            _seq += 1
            entry = _build_record(request_data, result, _seq, received_at)

            _records.append(entry)
            _accumulate_bucket(entry, received_at)

            _totals["flows"] += 1
            _totals["packets"] += entry["packets"]
            _totals["bytes"] += entry["bytes"]

            if entry["prediction"] == 1:
                _totals["threats"] += 1

            if entry["xgboost_malicious"]:
                _totals["xgb_alerts"] += 1

            if entry["autoencoder_anomalous"]:
                _totals["ae_alerts"] += 1

            _last_record_at = received_at

        _publish(entry)

    except Exception as exc:  # pragma: no cover - defensive
        print(f"Telemetry record error (prediction unaffected): {exc}")


def record_error(message: str) -> None:
    """Count a prediction that raised, so the UI can show a real error rate."""

    with _lock:
        _totals["errors"] += 1


# ============================================================
# PUBLIC: SUBSCRIBERS (SSE)
# ============================================================

def subscribe() -> "queue.Queue":

    channel: "queue.Queue" = queue.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)

    with _subscribers_lock:
        _subscribers.add(channel)

    return channel


def unsubscribe(channel: "queue.Queue") -> None:

    with _subscribers_lock:
        _subscribers.discard(channel)


def subscriber_count() -> int:

    with _subscribers_lock:
        return len(_subscribers)


def _publish(entry: Dict[str, Any]) -> None:

    with _subscribers_lock:
        channels = list(_subscribers)

    for channel in channels:

        try:
            channel.put_nowait(entry)

        except queue.Full:
            # Slow consumer. Drop this event rather than block capture.
            pass


# ============================================================
# PUBLIC: READ PATH
# ============================================================

def _snapshot() -> List[Dict[str, Any]]:
    """Newest-first copy of the buffer. Caller gets a stable list."""

    with _lock:
        return list(reversed(_records))


def _matches(
    entry: Dict[str, Any],
    protocol: Optional[str],
    verdict: Optional[str],
    risk: Optional[str],
    search: Optional[str],
) -> bool:

    if protocol and entry["protocol"] != protocol.lower():
        return False

    if verdict:
        wanted = verdict.upper()

        if wanted == "MALICIOUS" and entry["prediction"] != 1:
            return False

        if wanted == "BENIGN" and entry["prediction"] != 0:
            return False

    if risk and entry["risk"].upper() != risk.upper():
        return False

    if search:
        needle = search.lower()

        haystack = (
            f"{entry['flow_id']} {entry['src_ip']} {entry['dst_ip']} "
            f"{entry['src_port']} {entry['dst_port']} {entry['protocol']} "
            f"{entry['state']} {entry['risk']} {entry['label']}"
        ).lower()

        if needle not in haystack:
            return False

    return True


def list_flows(
    limit: int = 100,
    before_seq: Optional[int] = None,
    protocol: Optional[str] = None,
    verdict: Optional[str] = None,
    risk: Optional[str] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:

    limit = max(1, min(int(limit), 500))

    matched: List[Dict[str, Any]] = []

    for entry in _snapshot():

        if before_seq is not None and entry["seq"] >= before_seq:
            continue

        if not _matches(entry, protocol, verdict, risk, search):
            continue

        matched.append(entry)

        if len(matched) > limit:
            break

    has_more = len(matched) > limit
    page = matched[:limit]

    return {
        "flows": page,
        "has_more": has_more,
        "next_before_seq": page[-1]["seq"] if page and has_more else None,
        "buffer_size": len(_records),
        "buffer_capacity": MAX_RECORDS,
    }


def get_flow(seq: int) -> Optional[Dict[str, Any]]:

    with _lock:
        for entry in reversed(_records):

            if entry["seq"] == seq:
                return dict(entry)

    return None


def list_alerts(
    limit: int = 100,
    risk: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Alerts are exactly the flows the pipeline scored as malicious
    (prediction == 1). No separate alerting rule exists, and none is
    invented here.
    """

    limit = max(1, min(int(limit), 500))

    matched: List[Dict[str, Any]] = []
    counts = Counter()

    for entry in _snapshot():

        if entry["prediction"] != 1:
            continue

        counts[entry["risk"].upper() or "UNKNOWN"] += 1

        if entry["acknowledged"]:
            counts["ACKNOWLEDGED"] += 1

        if risk and entry["risk"].upper() != risk.upper():
            continue

        if acknowledged is not None and entry["acknowledged"] != acknowledged:
            continue

        if search and not _matches(entry, None, None, None, search):
            continue

        if len(matched) < limit:
            matched.append(entry)

    return {
        "alerts": matched,
        "counts": {
            "critical": counts.get("CRITICAL", 0),
            "high": counts.get("HIGH", 0),
            "medium": counts.get("MEDIUM", 0),
            "acknowledged": counts.get("ACKNOWLEDGED", 0),
            "total": sum(
                counts.get(key, 0)
                for key in ("CRITICAL", "HIGH", "MEDIUM")
            ),
        },
    }


def acknowledge(seq: int) -> Optional[Dict[str, Any]]:
    """
    Mark one alert acknowledged. In-memory only — this is cleared when the
    API restarts, and the UI says so.
    """

    with _lock:
        for entry in reversed(_records):

            if entry["seq"] == seq:
                entry["acknowledged"] = True
                return dict(entry)

    return None


def stats(window_seconds: int = STATS_WINDOW_SECONDS) -> Dict[str, Any]:

    now = time.time()
    cutoff = now - window_seconds

    with _lock:
        window = [
            entry for entry in _records if entry["received_at"] >= cutoff
        ]
        buffer_size = len(_records)
        totals = dict(_totals)
        last_record_at = _last_record_at
        started_at = _started_at

    flows = len(window)
    threats = sum(1 for entry in window if entry["prediction"] == 1)
    packets = sum(entry["packets"] for entry in window)
    total_bytes = sum(entry["bytes"] for entry in window)

    unique_sources = len({entry["src_ip"] for entry in window})
    unique_destinations = len({entry["dst_ip"] for entry in window})

    if last_record_at is None:
        seconds_since_last_flow = None
        capture_state = "idle"

    else:
        seconds_since_last_flow = now - last_record_at
        capture_state = (
            "receiving" if seconds_since_last_flow <= 15 else "idle"
        )

    return {
        "window_seconds": window_seconds,

        # Rates are computed over the window from values the capture layer
        # submitted. They describe traffic observed at the API, which is
        # what this process can actually measure.
        "flows": flows,
        "flows_per_second": flows / window_seconds,
        "packets": packets,
        "packets_per_second": packets / window_seconds,
        "bytes": total_bytes,
        "bytes_per_second": total_bytes / window_seconds,
        "bits_per_second": (total_bytes * 8) / window_seconds,

        "threats": threats,
        "threat_share": (threats / flows) if flows else 0.0,

        "unique_sources": unique_sources,
        "unique_destinations": unique_destinations,

        "capture_state": capture_state,
        "seconds_since_last_flow": seconds_since_last_flow,

        "buffer_size": buffer_size,
        "buffer_capacity": MAX_RECORDS,
        "subscribers": subscriber_count(),

        "totals": totals,
        "uptime_seconds": now - started_at,
    }


def timeseries(range_key: str = DEFAULT_RANGE) -> Dict[str, Any]:

    if range_key not in RANGE_SPEC:
        range_key = DEFAULT_RANGE

    window_seconds, bucket_seconds = RANGE_SPEC[range_key]

    now = int(time.time())
    start = now - window_seconds + 1

    point_count = window_seconds // bucket_seconds

    with _lock:
        relevant = [
            (second, dict(values))
            for second, values in _buckets
            if second >= start
        ]

    # Pre-allocate every bucket so the chart shows real zeros for quiet
    # periods instead of a gap the eye reads as missing data.
    points = [
        {
            "ts": start + (index * bucket_seconds),
            "flows": 0.0,
            "packets": 0.0,
            "bytes": 0.0,
            "threats": 0.0,
            "xgb_alerts": 0.0,
            "ae_alerts": 0.0,
        }
        for index in range(point_count)
    ]

    for second, values in relevant:

        index = (second - start) // bucket_seconds

        if 0 <= index < point_count:

            for key in (
                "flows",
                "packets",
                "bytes",
                "threats",
                "xgb_alerts",
                "ae_alerts",
            ):
                points[index][key] += values[key]

    # Convert counts to per-second rates so ranges stay comparable.
    for point in points:

        point["flows_per_second"] = point["flows"] / bucket_seconds
        point["packets_per_second"] = point["packets"] / bucket_seconds
        point["bits_per_second"] = (point["bytes"] * 8) / bucket_seconds

    return {
        "range": range_key,
        "window_seconds": window_seconds,
        "bucket_seconds": bucket_seconds,
        "points": points,
    }


def distribution(limit: int = 10) -> Dict[str, Any]:
    """
    Protocol mix and top talkers, computed over the retained buffer.
    """

    protocols = Counter()
    sources = Counter()
    destinations = Counter()
    risks = Counter()

    source_bytes = Counter()
    destination_bytes = Counter()

    with _lock:
        entries = list(_records)

    for entry in entries:

        protocols[entry["protocol"]] += 1
        sources[entry["src_ip"]] += 1
        destinations[entry["dst_ip"]] += 1
        risks[entry["risk"].upper() or "UNKNOWN"] += 1

        source_bytes[entry["src_ip"]] += entry["bytes"]
        destination_bytes[entry["dst_ip"]] += entry["bytes"]

    def _talkers(counter: Counter, byte_counter: Counter) -> List[Dict]:

        return [
            {
                "address": address,
                "flows": count,
                "bytes": byte_counter.get(address, 0.0),
            }
            for address, count in counter.most_common(limit)
        ]

    return {
        "sample_size": len(entries),
        "protocols": [
            {"protocol": name, "flows": count}
            for name, count in protocols.most_common()
        ],
        "risks": [
            {"risk": name, "flows": count}
            for name, count in risks.most_common()
        ],
        "top_sources": _talkers(sources, source_bytes),
        "top_destinations": _talkers(destinations, destination_bytes),
    }


def reset() -> None:
    """Clear the buffer. Used by the Settings page and by tests."""

    global _seq, _last_record_at

    with _lock:
        _records.clear()
        _buckets.clear()
        _seq = 0
        _last_record_at = None

        for key in _totals:
            _totals[key] = 0 if isinstance(_totals[key], int) else 0.0
