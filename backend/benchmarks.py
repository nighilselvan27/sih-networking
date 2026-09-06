"""
Read-only access to the benchmark artifacts already in the repository.

Every number this module returns is parsed from a file on disk at request
time. Nothing is computed, averaged, rounded for presentation, or filled in
when a file is missing — an absent artifact is reported as absent so the UI
can say so rather than show a plausible-looking substitute.

Only the small summary artifacts are read. The per-row prediction dumps in
models/ (35-92 MB each) are deliberately never opened.

Stdlib only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import json


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


# ============================================================
# PARSING
# ============================================================

def _coerce(value: str) -> Any:
    """
    CSV cells are strings. Turn numeric cells into numbers so the frontend
    can sort and format them, and turn genuinely empty cells into null
    (synthetic_benchmark_per_attack.csv has an empty ROC_AUC column).
    """

    text = (value or "").strip()

    if text == "":
        return None

    try:
        number = float(text)

    except ValueError:
        return text

    if number != number or number in (float("inf"), float("-inf")):
        return None

    if number.is_integer() and "." not in text and "e" not in text.lower():
        return int(number)

    return number


def read_csv(path: Path) -> Dict[str, Any]:

    if not path.exists():
        return {
            "file": path.name,
            "path": str(path),
            "available": False,
            "columns": [],
            "rows": [],
        }

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:

            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])

            rows = [
                {key: _coerce(row.get(key, "")) for key in columns}
                for row in reader
            ]

        return {
            "file": path.name,
            "path": str(path),
            "available": True,
            "columns": columns,
            "rows": rows,
        }

    except Exception as exc:
        return {
            "file": path.name,
            "path": str(path),
            "available": False,
            "error": str(exc),
            "columns": [],
            "rows": [],
        }


def read_json(path: Path) -> Dict[str, Any]:

    if not path.exists():
        return {
            "file": path.name,
            "path": str(path),
            "available": False,
            "data": None,
        }

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        return {
            "file": path.name,
            "path": str(path),
            "available": True,
            "data": data,
        }

    except Exception as exc:
        return {
            "file": path.name,
            "path": str(path),
            "available": False,
            "error": str(exc),
            "data": None,
        }


def _downsample(rows: List[Dict[str, Any]], target: int) -> List[Dict]:
    """
    Evenly thin a long series for charting, always keeping the first and
    last row. Used only for the 91-row threshold sweep and the 44-epoch
    training history, both of which are already small; this is a guard for
    the case where those files grow.
    """

    if len(rows) <= target:
        return rows

    step = len(rows) / target

    indices = sorted({int(index * step) for index in range(target)})
    indices = [index for index in indices if index < len(rows)]

    if indices and indices[-1] != len(rows) - 1:
        indices.append(len(rows) - 1)

    return [rows[index] for index in indices]


# ============================================================
# PUBLIC
# ============================================================

def collect() -> Dict[str, Any]:
    """
    Assemble every benchmark section the Benchmarks page renders.

    Each section carries the filename it came from, so the UI can attribute
    every table and chart to a real artifact.
    """

    threshold_sweep = read_csv(MODELS_DIR / "xgboost_threshold_results.csv")

    if threshold_sweep["available"]:
        threshold_sweep["rows"] = _downsample(threshold_sweep["rows"], 120)

    training_history = read_csv(
        MODELS_DIR / "autoencoder_training_history.csv"
    )

    if training_history["available"]:
        training_history["rows"] = _downsample(
            training_history["rows"], 120
        )

    sections = {
        # Single-scenario comparison of four candidate classifiers.
        "model_comparison": read_csv(
            MODELS_DIR / "model_comparison.csv"
        ),

        # Cross-scenario: trained on 1-10, tested on 11-13.
        "multiscenario": read_csv(
            MODELS_DIR / "multiscenario_benchmark.csv"
        ),
        "multiscenario_per_scenario": read_csv(
            MODELS_DIR / "multiscenario_benchmark_per_scenario.csv"
        ),

        # The deployed detector: XGBoost, Autoencoder and the gated hybrid,
        # with confusion matrices.
        "hybrid": read_csv(
            MODELS_DIR / "autoencoder_xgboost_hybrid_benchmark.csv"
        ),
        "hybrid_per_scenario": read_csv(
            MODELS_DIR / "autoencoder_xgboost_hybrid_per_scenario.csv"
        ),

        # Autoencoder in isolation, including reconstruction-error stats.
        "autoencoder": read_csv(
            MODELS_DIR / "autoencoder_benchmark.csv"
        ),
        "autoencoder_per_scenario": read_csv(
            MODELS_DIR / "autoencoder_benchmark_per_scenario.csv"
        ),
        "autoencoder_training_history": training_history,

        # Operating-point selection.
        "threshold_sweep": threshold_sweep,
        "threshold_recommendation": read_json(
            MODELS_DIR / "threshold_recommendation.json"
        ),
        "best_threshold": read_json(
            MODELS_DIR / "xgboost_best_threshold.json"
        ),

        # Detection rate per synthetic attack class.
        "synthetic_per_attack": read_csv(
            MODELS_DIR / "synthetic_benchmark_per_attack.csv"
        ),

        # Deployed configuration as recorded at training time.
        "hybrid_config": read_json(
            MODELS_DIR / "autoencoder_xgboost_hybrid_config.json"
        ),

        # Streaming replay of the held-out scenarios.
        "streaming_summary": read_json(
            OUTPUTS_DIR / "streaming_summary.json"
        ),
    }

    missing = sorted(
        section["file"]
        for section in sections.values()
        if not section.get("available")
    )

    return {
        "sections": sections,
        "missing": missing,
        "models_dir": str(MODELS_DIR),
        "outputs_dir": str(OUTPUTS_DIR),
    }


def feature_list() -> Dict[str, Any]:
    """
    The 30 model input features, read from the artifact the training run
    wrote rather than restated here.
    """

    path = MODELS_DIR / "ctu13_multiscenario_features.txt"

    if not path.exists():
        return {"file": path.name, "available": False, "features": []}

    try:
        features = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        return {
            "file": path.name,
            "available": True,
            "features": features,
        }

    except Exception as exc:
        return {
            "file": path.name,
            "available": False,
            "error": str(exc),
            "features": [],
        }


def autoencoder_schema() -> Dict[str, Any]:
    """
    Autoencoder input schema: 30 numeric features plus the one-hot blocks
    that bring the input to 257 dimensions.
    """

    parsed = read_json(MODELS_DIR / "autoencoder_features.json")

    if not parsed["available"]:
        return {
            "file": parsed["file"],
            "available": False,
            "numeric_features": [],
            "categorical_features": [],
            "input_dimension": None,
            "categorical_breakdown": [],
        }

    data = parsed["data"] or {}

    numeric = data.get("numeric_features") or []
    categorical = data.get("categorical_features") or []
    encoded = data.get("encoded_feature_names") or []

    # Count how many encoded columns each categorical feature contributes.
    breakdown = []

    for feature in categorical:
        prefix = f"{feature}_"

        breakdown.append(
            {
                "feature": feature,
                "columns": sum(
                    1 for name in encoded if str(name).startswith(prefix)
                ),
            }
        )

    return {
        "file": parsed["file"],
        "available": True,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "input_dimension": data.get("input_dimension"),
        "encoded_feature_count": len(encoded),
        "categorical_breakdown": breakdown,
    }
