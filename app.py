"""
app.py  FastAPI Backend
--------------------------
Credit Card Fraud Detection REST API.

Endpoints:
  GET  /          → health check
  POST /predict   → single transaction prediction
  POST /predict-batch → multiple transactions

Usage:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── App Setup ────
app = FastAPI(
    title="💳 Credit Fraud Detection API",
    description="XGBoost credit card transactions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Load ───────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "model/best_model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    print(f"✅ Model loaded successfully: {MODEL_PATH}")
except FileNotFoundError as e:
    model = None
    print(f"⚠️  {e}")

# ─── Feature Names ────────────────────────────────────────────────────────────
# creditcard.csv
FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class Transaction(BaseModel):
    """Details of each transaction."""
    Time: float = Field(..., example=406.0, description="Seconds since first transaction")
    V1: float = Field(..., example=-1.359807); V2: float = Field(..., example=-0.072782)
    V3: float = Field(..., example=2.536347);  V4: float = Field(..., example=1.378155)
    V5: float = Field(..., example=-0.338321); V6: float = Field(..., example=0.462388)
    V7: float = Field(..., example=0.239599);  V8: float = Field(..., example=0.098698)
    V9: float = Field(..., example=0.363787);  V10: float = Field(..., example=0.090794)
    V11: float = Field(..., example=-0.551600); V12: float = Field(..., example=-0.617801)
    V13: float = Field(..., example=-0.991390); V14: float = Field(..., example=-0.311169)
    V15: float = Field(..., example=1.468177);  V16: float = Field(..., example=-0.470401)
    V17: float = Field(..., example=0.207971);  V18: float = Field(..., example=0.025791)
    V19: float = Field(..., example=0.403993);  V20: float = Field(..., example=0.251412)
    V21: float = Field(..., example=-0.018307); V22: float = Field(..., example=0.277838)
    V23: float = Field(..., example=-0.110474); V24: float = Field(..., example=0.066928)
    V25: float = Field(..., example=0.128539);  V26: float = Field(..., example=-0.189115)
    V27: float = Field(..., example=0.133558);  V28: float = Field(..., example=-0.021053)
    Amount: float = Field(..., example=149.62, description="Transaction amount in USD")

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 406.0, "Amount": 149.62,
                **{f"V{i}": round(float(np.random.randn()), 6) for i in range(1, 29)}
            }
        }


class PredictionResponse(BaseModel):
    prediction: int           # 0 = Legit, 1 = Fraud
    label: str                # "Legit" or "Fraud"
    fraud_probability: float  # 0.0 – 1.0
    confidence: str           # "Low / Medium / High"


class BatchRequest(BaseModel):
    transactions: List[Transaction]


class BatchResponse(BaseModel):
    results: List[PredictionResponse]
    total: int
    fraud_count: int
    legit_count: int

# ─── Helper ───────────────────────────────────────────────────────────────────

def _confidence(prob: float) -> str:
    if prob < 0.3 or prob > 0.7:
        return "High"
    if prob < 0.4 or prob > 0.6:
        return "Medium"
    return "Low"


def _predict_df(df: pd.DataFrame) -> List[PredictionResponse]:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available now. Talk to Admin."
        )
    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= 0.5).astype(int)

    responses = []
    for pred, prob in zip(preds, probs):
        responses.append(PredictionResponse(
            prediction=int(pred),
            label="Fraud" if pred == 1 else "Legit",
            fraud_probability=round(float(prob), 6),
            confidence=_confidence(float(prob)),
        ))
    return responses

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.model_dump()])[FEATURE_COLUMNS]
    return _predict_df(data)[0]


@app.post("/predict-batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(batch: BatchRequest):
    if not batch.transactions:
        raise HTTPException(status_code=400, detail="No Transaction.")

    rows = [t.model_dump() for t in batch.transactions]
    df = pd.DataFrame(rows)[FEATURE_COLUMNS]
    results = _predict_df(df)

    fraud_count = sum(1 for r in results if r.prediction == 1)
    return BatchResponse(
        results=results,
        total=len(results),
        fraud_count=fraud_count,
        legit_count=len(results) - fraud_count,
    )
