from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

    return {
        "status": "healthy",
        "models": "loaded",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
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

        return result

    except Exception as exc:

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

            results.append(
                predict(data)
            )

        except Exception as exc:

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