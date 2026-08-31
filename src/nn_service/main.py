"""
Standalone FT-Transformer scoring service -- deliberately its own process, never
xgboost. Two separate copies of the OpenMP runtime (one bundled with the PyTorch wheel,
one pulled in via scikit-learn/XGBoost's Homebrew-linked OpenMP) segfaulted this exact
project's development machine once already when torch and xgboost shared a process (see
notebooks/04b_neural_network_benchmark.ipynb's crash writeup and the README's Neural
network benchmark section). The main scoring API (src/api/) calls this service over HTTP
for `nn_score` rather than importing torch directly, so that conflict can't occur here by
construction, the same reasoning that keeps the offline benchmark notebooks split apart.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest
from pydantic import BaseModel

from .model import FTTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("nn-service")

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

NN_SCORE_LATENCY = Histogram(
    "nn_score_latency_seconds",
    "Latency of /score_nn requests (model forward pass only)",
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1),
)

state: dict = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading FT-Transformer artifacts from %s", MODEL_DIR)
    with open(os.path.join(MODEL_DIR, "ft_transformer_config.json")) as f:
        config = json.load(f)
    with open(os.path.join(MODEL_DIR, "ft_transformer_scaler.json")) as f:
        scaler = json.load(f)
    with open(os.path.join(MODEL_DIR, "ft_transformer_categories.json")) as f:
        cats = json.load(f)

    model = FTTransformer(
        n_numeric=config["n_numeric"], n_categories=config["n_categories"],
        d_token=config["d_token"], n_blocks=config["n_blocks"],
        n_heads=config["n_heads"], dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "ft_transformer.pt"), map_location="cpu"
    ))
    model.eval()

    state["model"] = model
    state["features"] = scaler["features"]
    state["mean"] = np.array(scaler["mean"], dtype=np.float32)
    state["scale"] = np.array(scaler["scale"], dtype=np.float32)
    state["cat_to_idx"] = cats["categories"]
    state["unk_idx"] = cats["unk_idx"]
    logger.info(
        "FT-Transformer loaded: %d numeric features, %d categories (+unknown)",
        config["n_numeric"], len(state["cat_to_idx"]),
    )

    yield
    state.clear()


app = FastAPI(title="FT-Transformer NN Scoring Service", version="1.0", lifespan=lifespan)


class FeaturesIn(BaseModel):
    features: dict[str, float]
    category: str


class ScoreOut(BaseModel):
    nn_score: float


@app.get("/health")
def health():
    if state.get("model") is None:
        raise HTTPException(503, "not ready")
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/score_nn", response_model=ScoreOut)
def score_nn(payload: FeaturesIn):
    model = state.get("model")
    if model is None:
        raise HTTPException(503, "not ready")

    start = time.perf_counter()
    features = state["features"]
    try:
        raw = np.array([payload.features[f] for f in features], dtype=np.float32)
    except KeyError as e:
        raise HTTPException(422, f"missing feature: {e}")

    x_num = (raw - state["mean"]) / state["scale"]
    x_num_t = torch.from_numpy(x_num).unsqueeze(0)

    cat_idx = state["cat_to_idx"].get(payload.category, state["unk_idx"])
    x_cat_t = torch.tensor([cat_idx], dtype=torch.long)

    with torch.no_grad():
        logit = model(x_num_t, x_cat_t)
        proba = torch.sigmoid(logit).item()

    NN_SCORE_LATENCY.observe(time.perf_counter() - start)
    return ScoreOut(nn_score=proba)
