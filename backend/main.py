import os
os.environ.setdefault("IDS_DEMO_CONTROLS", "1")
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import asyncio
import json
import queue

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import benchmarks
import demo
import inference
import telemetry
from inference import load_models, model_info, predict
from schemas import FlowInput, PredictionResponse


app = FastAPI(
    title="CTU-13 Hybrid IDS API",
    description=(
        "AI-based intrusion detection using "
        "XGBoost and Autoencoder."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# STARTUP
# ---------------------------------------------------------

@app.on_event("startup")
def startup_event():

    print()
    print("=" * 70)
    print("CTU-13 HYBRID IDS API")
    print("=" * 70)
    print("Loading models...")

    load_models()

    print("API ready.")
    print("=" * 70)


# ---------------------------------------------------------
# ROOT
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "name": "CTU-13 Hybrid IDS API",
        "status": "running",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "docs": "/docs",
    }


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.get("/health")
def health():

    # Reports the real loader state and which model artifacts are present,
    # so the console's diagnostics reflect the process rather than a
    # constant. The original keys are preserved.

    artifacts = {
        "xgboost": inference.XGB_MODEL_PATH,
        "autoencoder": inference.AE_MODEL_PATH,
        "scaler": inference.AE_SCALER_PATH,
        "schema": inference.AE_SCHEMA_PATH,
    }

    present = {
        name: path.exists()
        for name, path in artifacts.items()
    }

    loaded = bool(inference.models_loaded)
    complete = all(present.values())

    return {
        "status": "healthy" if (loaded and complete) else "degraded",
        "models": "loaded" if loaded else "not_loaded",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "models_loaded": loaded,
        "artifacts": [
            {
                "name": name,
                "file": artifacts[name].name,
                "present": present[name],
            }
            for name in artifacts
        ],
        "missing_artifacts": [
            artifacts[name].name
            for name in artifacts
            if not present[name]
        ],
    }


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

@app.get("/model-info")
def get_model_info():

    try:
        return model_info()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_flow(flow: FlowInput):

    try:

        data: Dict[str, Any] = (
            flow.model_dump()
            if hasattr(flow, "model_dump")
            else flow.dict()
        )

        result = predict(data)

        # Read-side only. predict() has already returned; this records what
        # it produced so the console has something to display. It cannot
        # change the verdict and never raises into this handler.
        telemetry.record(data, result)

        return result

    except Exception as exc:

        telemetry.record_error(str(exc))

        print(
            f"Prediction error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ---------------------------------------------------------
# BATCH PREDICTION
# ---------------------------------------------------------

@app.post("/predict/batch")
def predict_batch(
    flows: list[FlowInput]
):

    if len(flows) > 1000:

        raise HTTPException(
            status_code=400,
            detail="Maximum batch size is 1000 flows."
        )

    results = []

    for flow in flows:

        try:

            data = (
                flow.model_dump()
                if hasattr(
                    flow,
                    "model_dump"
                )
                else flow.dict()
            )

            result = predict(data)

            telemetry.record(data, result)

            results.append(
                result
            )

        except Exception as exc:

            telemetry.record_error(str(exc))

            results.append(
                {
                    "error": str(exc)
                }
            )

    threats = sum(
        1
        for result in results
        if result.get("prediction") == 1
    )

    return {
        "total_flows": len(results),
        "threats": threats,
        "benign": len(results) - threats,
        "results": results,
    }


# =========================================================
# OPERATOR CONSOLE READ API
#
# Everything below is read-only over the in-memory history of
# predictions the API has already made, plus the benchmark
# artifacts already on disk. No route here runs a model.
# =========================================================

# ---------------------------------------------------------
# STATS
# ---------------------------------------------------------

@app.get("/api/stats")
def api_stats(
    window: int = Query(
        telemetry.STATS_WINDOW_SECONDS,
        ge=5,
        le=3600,
    )
):

    return telemetry.stats(window_seconds=window)


# ---------------------------------------------------------
# TIME SERIES
# ---------------------------------------------------------

@app.get("/api/timeseries")
def api_timeseries(
    range: str = Query(
        telemetry.DEFAULT_RANGE,
        pattern="^(1m|5m|15m|1h)$",
    )
):

    return telemetry.timeseries(range_key=range)


# ---------------------------------------------------------
# FLOWS
# ---------------------------------------------------------

@app.get("/api/flows")
def api_flows(
    limit: int = Query(100, ge=1, le=500),
    before_seq: Optional[int] = None,
    protocol: Optional[str] = None,
    verdict: Optional[str] = None,
    risk: Optional[str] = None,
    q: Optional[str] = None,
):

    return telemetry.list_flows(
        limit=limit,
        before_seq=before_seq,
        protocol=protocol,
        verdict=verdict,
        risk=risk,
        search=q,
    )


@app.get("/api/flows/{seq}")
def api_flow(seq: int):

    entry = telemetry.get_flow(seq)

    if entry is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Flow {seq} is not in the buffer. It may have been "
                f"evicted, or the API may have restarted."
            ),
        )

    return entry


# ---------------------------------------------------------
# ALERTS
#
# An alert is a flow the pipeline scored as malicious. There is
# no separate alerting rule.
# ---------------------------------------------------------

@app.get("/api/alerts")
def api_alerts(
    limit: int = Query(100, ge=1, le=500),
    risk: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    q: Optional[str] = None,
):

    return telemetry.list_alerts(
        limit=limit,
        risk=risk,
        acknowledged=acknowledged,
        search=q,
    )


@app.post("/api/alerts/{seq}/ack")
def api_acknowledge(seq: int):

    entry = telemetry.acknowledge(seq)

    if entry is None:

        raise HTTPException(
            status_code=404,
            detail=f"Alert {seq} is not in the buffer.",
        )

    return entry


# ---------------------------------------------------------
# DISTRIBUTION
# ---------------------------------------------------------

@app.get("/api/distribution")
def api_distribution(
    limit: int = Query(10, ge=1, le=50)
):

    return telemetry.distribution(limit=limit)


# ---------------------------------------------------------
# LIVE STREAM (SSE)
# ---------------------------------------------------------

@app.get("/api/stream")
async def api_stream(request: Request):

    channel = telemetry.subscribe()

    async def events():

        last_heartbeat = asyncio.get_event_loop().time()

        try:
            # Tell the client the stream is live before any traffic
            # arrives, so the UI can distinguish "connected, quiet" from
            # "not connected".
            yield "event: ready\ndata: {}\n\n"

            while True:

                if await request.is_disconnected():
                    break

                drained = 0

                # Drain whatever has accumulated, bounded so one busy
                # interval cannot monopolise the event loop.
                while drained < 100:

                    try:
                        entry = channel.get_nowait()

                    except queue.Empty:
                        break

                    drained += 1

                    payload = json.dumps(entry, default=str)

                    yield f"event: flow\ndata: {payload}\n\n"

                now = asyncio.get_event_loop().time()

                if now - last_heartbeat >= 15:

                    last_heartbeat = now

                    yield (
                        "event: heartbeat\n"
                        f'data: {{"ts": {now:.3f}}}\n\n'
                    )

                if drained == 0:
                    await asyncio.sleep(0.25)

        finally:
            telemetry.unsubscribe(channel)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Disable proxy buffering, which otherwise holds events back.
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------
# BENCHMARKS
# ---------------------------------------------------------

@app.get("/api/benchmarks")
def api_benchmarks():

    return benchmarks.collect()


@app.get("/api/features")
def api_features():

    return {
        "xgboost": benchmarks.feature_list(),
        "autoencoder": benchmarks.autoencoder_schema(),
    }


# ---------------------------------------------------------
# DEMO CONTROLS
#
# Disabled unless IDS_DEMO_CONTROLS=1 and the caller is on
# loopback. Presets are fixed server-side.
# ---------------------------------------------------------

@app.get("/api/demo/status")
def api_demo_status(request: Request):

    client_host = request.client.host if request.client else None

    return demo.status(client_host=client_host)


@app.post("/api/demo/run")
def api_demo_run(request: Request, body: Dict[str, Any]):

    client_host = request.client.host if request.client else None

    preset = str(body.get("preset", ""))

    try:
        return demo.run(preset, client_host=client_host)

    except demo.DemoError as exc:

        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        )


@app.post("/api/demo/stop")
def api_demo_stop(request: Request):

    client_host = request.client.host if request.client else None

    try:
        return demo.stop(client_host=client_host)

    except demo.DemoError as exc:

        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        )


# ---------------------------------------------------------
# BUFFER RESET
# ---------------------------------------------------------

@app.post("/api/buffer/reset")
def api_buffer_reset():

    telemetry.reset()

    return {"cleared": True}


# =========================================================
# CONSOLE (built frontend, optional)
#
# Served only when frontend/dist exists. During development the
# Vite dev server proxies to this API instead, so this mount is
# absent and nothing shadows the routes above.
# =========================================================

CONSOLE_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if CONSOLE_DIR.exists():

    app.mount(
        "/console/assets",
        StaticFiles(directory=str(CONSOLE_DIR / "assets")),
        name="console-assets",
    )

    CONSOLE_ROOT = CONSOLE_DIR.resolve()

    @app.get("/console")
    @app.get("/console/{path:path}")
    def console(path: str = ""):

        index = CONSOLE_ROOT / "index.html"

        if path:

            # Contain the lookup inside the build directory. Without this,
            # a request for /console/../../<anything> would read a file
            # outside it.
            candidate = (CONSOLE_ROOT / path).resolve()

            inside = (
                candidate == CONSOLE_ROOT
                or CONSOLE_ROOT in candidate.parents
            )

            if inside and candidate.is_file():
                return FileResponse(candidate)

        # Client-side routing: anything else under /console resolves to the
        # single-page entry point.
        return FileResponse(index)


# ---------------------------------------------------------
# RUN DIRECTLY
# ---------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
