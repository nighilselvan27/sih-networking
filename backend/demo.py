"""
Demonstration controls for the local attack-testing harness.

This module can start `scripts/test_traffic.py` — the controlled, local-only
traffic generator the README documents — so the console can drive a live
demonstration without a second terminal.

Safety model, in order:

1.  Disabled by default. Requires the environment variable
    IDS_DEMO_CONTROLS=1 to be set on the API process.
2.  Loopback callers only. A request from any other address is refused.
3.  Fixed presets. The client sends a preset NAME. Host, port, rate, size
    and duration are hard-coded here and never read from the request, so
    there is no path from HTTP input to a command argument.
4.  No shell. subprocess is given an argv list with shell=False.
5.  One run at a time, with a wall-clock kill.

`scripts/test_traffic.py` independently refuses any destination that is not
loopback or private, and caps rate, count, size and duration. This module
does not weaken those checks and does not bypass the script.

Stdlib only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import subprocess
import sys
import threading
import time


BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
TRAFFIC_SCRIPT = SCRIPTS_DIR / "test_traffic.py"

ENV_FLAG = "IDS_DEMO_CONTROLS"

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

# Hard ceiling regardless of preset. test_traffic.py caps --duration at
# 300s; this process additionally kills anything still alive after this.
MAX_RUN_SECONDS = 45


# ============================================================
# PRESETS
#
# Each entry is a fixed argument list, mirroring the invocations in
# README.md. Nothing here is parameterised by the caller.
# ============================================================

PRESETS: Dict[str, Dict[str, Any]] = {
    "benign-udp": {
        "label": "Benign UDP",
        "description": "Steady low-rate UDP to a local port.",
        "expectation": "Normal traffic. Expected to score benign.",
        "args": [
            "--type", "benign-udp",
            "--duration", "10",
            "--rate", "200",
            "--port", "9999",
        ],
    },
    "udp-flood": {
        "label": "UDP flood",
        "description": "High-rate UDP toward a single local port.",
        "expectation": (
            "High packet and byte rate. The verdict comes from the model."
        ),
        "args": [
            "--type", "udp",
            "--duration", "10",
            "--rate", "3000",
            "--port", "9999",
        ],
    },
    "tcp-syn": {
        "label": "TCP SYN",
        "description": (
            "Real SYNs toward a closed local port. No source-IP spoofing."
        ),
        "expectation": (
            "Half-open connection attempts. The verdict comes from the model."
        ),
        "args": [
            "--type", "syn",
            "--duration", "10",
            "--rate", "1500",
            "--port", "9998",
        ],
    },
    "burst": {
        "label": "Multi-socket burst",
        "description": "High-rate benign traffic across four local ports.",
        "expectation": "Elevated rate across several flows.",
        "args": [
            "--type", "burst",
            "--duration", "10",
            "--rate", "2000",
        ],
    },
    "benign-tcp": {
        "label": "Benign TCP",
        "description": (
            "Complete TCP conversation against a local echo listener."
        ),
        "expectation": "Normal traffic. Expected to score benign.",
        "args": [
            "--type", "benign-tcp",
            "--count", "200",
            "--size", "512",
        ],
    },
}


# ============================================================
# STATE
# ============================================================

_lock = threading.Lock()

_process: Optional[subprocess.Popen] = None
_current: Optional[Dict[str, Any]] = None
_last: Optional[Dict[str, Any]] = None


class DemoError(Exception):
    """Raised for any refusal. The route maps this to a 4xx."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# ============================================================
# GUARDS
# ============================================================

def is_enabled() -> bool:

    return os.getenv(ENV_FLAG, "").strip() == "1"


def is_loopback(client_host: Optional[str]) -> bool:

    return (client_host or "") in LOOPBACK_HOSTS


def _require_permission(client_host: Optional[str]) -> None:

    if not is_enabled():
        raise DemoError(
            f"Demo controls are disabled. Set {ENV_FLAG}=1 on the API "
            f"process to enable them.",
            status_code=403,
        )

    if not is_loopback(client_host):
        raise DemoError(
            "Demo controls accept loopback requests only.",
            status_code=403,
        )

    if not TRAFFIC_SCRIPT.exists():
        raise DemoError(
            f"Traffic generator not found: {TRAFFIC_SCRIPT}",
            status_code=500,
        )


# ============================================================
# LIFECYCLE
# ============================================================

def _poll_locked() -> None:
    """Reap a finished process. Caller must hold _lock."""

    global _process, _current, _last

    if _process is None or _current is None:
        return

    code = _process.poll()
    elapsed = time.time() - _current["started_at"]

    if code is None and elapsed > MAX_RUN_SECONDS:

        try:
            _process.terminate()

        except Exception:
            pass

        code = -1
        _current["outcome"] = "timeout"

    if code is None:
        return

    _current["finished_at"] = time.time()
    _current["exit_code"] = code
    _current.setdefault(
        "outcome", "completed" if code == 0 else "failed"
    )

    _last = _current
    _current = None
    _process = None


def _public(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:

    if entry is None:
        return None

    return {
        "preset": entry["preset"],
        "label": entry["label"],
        "command": entry["command"],
        "started_at": entry["started_at"],
        "finished_at": entry.get("finished_at"),
        "exit_code": entry.get("exit_code"),
        "outcome": entry.get("outcome", "running"),
    }


def status(client_host: Optional[str] = None) -> Dict[str, Any]:
    """
    Always answers, whether or not demo controls are enabled — the UI uses
    this to decide whether to render the panel at all.
    """

    with _lock:
        _poll_locked()

        running = _public(_current)
        last = _public(_last)

    return {
        "enabled": is_enabled(),
        "loopback": is_loopback(client_host),
        "script_available": TRAFFIC_SCRIPT.exists(),
        "env_flag": ENV_FLAG,
        "max_run_seconds": MAX_RUN_SECONDS,
        "running": running,
        "last_run": last,
        "presets": [
            {
                "name": name,
                "label": preset["label"],
                "description": preset["description"],
                "expectation": preset["expectation"],
                "command": _command_text(name),
            }
            for name, preset in PRESETS.items()
        ],
    }


def _command_text(name: str) -> str:
    """The equivalent terminal command, shown in the UI for transparency."""

    args = " ".join(PRESETS[name]["args"])

    return f"python scripts/test_traffic.py {args}"


def run(preset: str, client_host: Optional[str] = None) -> Dict[str, Any]:

    global _process, _current

    _require_permission(client_host)

    if preset not in PRESETS:
        raise DemoError(
            f"Unknown preset '{preset}'. "
            f"Allowed: {', '.join(sorted(PRESETS))}.",
            status_code=400,
        )

    with _lock:
        _poll_locked()

        if _process is not None:
            raise DemoError(
                "A demo run is already in progress.",
                status_code=409,
            )

        # argv list, shell=False. The only variable part is the preset key,
        # which has already been checked against PRESETS.
        argv: List[str] = [
            sys.executable,
            str(TRAFFIC_SCRIPT),
            *PRESETS[preset]["args"],
        ]

        try:
            _process = subprocess.Popen(
                argv,
                shell=False,
                cwd=str(BASE_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        except Exception as exc:
            _process = None

            raise DemoError(
                f"Could not start the traffic generator: {exc}",
                status_code=500,
            )

        _current = {
            "preset": preset,
            "label": PRESETS[preset]["label"],
            "command": _command_text(preset),
            "started_at": time.time(),
        }

        started = _public(_current)

    return {"started": True, "run": started}


def stop(client_host: Optional[str] = None) -> Dict[str, Any]:

    global _process, _current, _last

    _require_permission(client_host)

    with _lock:
        _poll_locked()

        if _process is None:
            return {"stopped": False, "reason": "No demo run in progress."}

        try:
            _process.terminate()

            # terminate() is asynchronous; without this wait the poll below
            # would still see the process alive and fail to reap it.
            try:
                _process.wait(timeout=3)

            except subprocess.TimeoutExpired:
                _process.kill()
                _process.wait(timeout=3)

        except Exception as exc:
            raise DemoError(
                f"Could not stop the traffic generator: {exc}",
                status_code=500,
            )

        if _current is not None:
            _current["outcome"] = "stopped"

        _poll_locked()

        return {"stopped": True, "last_run": _public(_last)}
