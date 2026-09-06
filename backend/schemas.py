from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class FlowInput(BaseModel):
    Dur: float = 0.0
    Sport: float = 0.0
    Dport: float = 0.0
    sTos: float = 0.0
    dTos: float = 0.0
    TotPkts: float = 0.0
    TotBytes: float = 0.0
    SrcBytes: float = 0.0
    PacketsPerSecond: float = 0.0
    BytesPerSecond: float = 0.0
    AvgPacketSize: float = 0.0
    DstBytes: float = 0.0
    SrcByteRatio: float = 0.0
    DstByteRatio: float = 0.0
    SourceFlowCount30s: float = 0.0
    UniqueDstIPs30s: float = 0.0
    UniqueDstPorts30s: float = 0.0
    UniqueSrcPorts30s: float = 0.0
    SourceTotalBytes30s: float = 0.0
    SourceTotalPackets30s: float = 0.0
    DestinationFlowCount30s: float = 0.0
    UniqueSrcIPs30s: float = 0.0
    DestinationTotalBytes30s: float = 0.0
    DestinationRepeatCount: float = 0.0
    InterArrivalTime: float = 0.0
    PairInterArrivalTime: float = 0.0
    FlowsPerSecond30s: float = 0.0
    PacketsPerSecond30s: float = 0.0
    BytesPerSecond30s: float = 0.0
    SourceOutboundRatio: float = 0.0

    Proto: str = "tcp"
    Dir: str = "->"
    # NOTE: "CON" is not one of the 257 encoded autoencoder columns
    # (the trained schema has no State_CON). "INT" is a real Argus
    # state meaning "no response observed", and is well represented
    # in training (frequency 0.158).
    State: str = "INT"

    SrcAddr: Optional[str] = None
    DstAddr: Optional[str] = None

    class Config:
        extra = "allow"


class PredictionResponse(BaseModel):
    # extra="allow" keeps the richer keys that inference.predict()
    # returns (xgboost_score, hybrid_score, risk, gated, evidence, ...)
    # instead of letting response_model filter them out.
    model_config = ConfigDict(extra="allow")

    # --- standardized alert schema ---
    # timestamp is capture-side when live_capture.py supplies it,
    # otherwise a server-side time recorded at prediction.
    timestamp: str
    # same identifier format live_capture.py builds:
    # "<src_ip>:<sport>-<dst_ip>:<dport>-<PROTO>"
    flow_id: str
    # CTU-13 ThreatClass vocabulary: BENIGN | BOTNET
    threat_class: str
    # this flow's actual values for the model's highest-gain features
    supporting_features: Dict[str, Any] = Field(default_factory=dict)

    prediction: int
    label: str
    confidence: float
    xgboost_probability: float
    autoencoder_score: float
    autoencoder_anomaly: bool
    xgboost_anomaly: bool
    gated: bool
    risk_level: str
    explanation: str
    details: Dict[str, Any] = Field(default_factory=dict)