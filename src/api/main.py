import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .feature_store import FeatureStore
from .metrics import PREDICTIONS_TOTAL, SCORE_DISTRIBUTION, SCORE_LATENCY
from .model import FraudModel
from .schemas import HealthOut, ScoreOut, TransactionIn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fraud-api")

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
DATA_DIR = os.environ.get("DATA_DIR", "data")
ALPHA = 20.0

# The deployed model (notebook 04f) is a stack of this XGBoost model plus an FT-Transformer
# score (`nn_score`). The transformer lives in its own process (src/nn_service/) rather
# than being imported here directly -- PyTorch and XGBoost's native code segfault when
# both run in the same process (see notebooks/04b's crash writeup and the README's neural
# network benchmark section), so this API stays xgboost-only and calls out over HTTP.
NN_SERVICE_URL = os.environ.get("NN_SERVICE_URL", "http://localhost:8100")

state: dict = {"model": None, "store": None}


def _bootstrap_rate_dicts(raw_txns: pd.DataFrame, global_rate: float, alpha: float = ALPHA):
    def rates(key: str) -> dict:
        g = raw_txns.groupby(key)["is_fraud"].agg(["sum", "count"])
        smoothed = (g["sum"] + alpha * global_rate) / (g["count"] + alpha)
        return smoothed.to_dict()

    return rates("merchant_id"), rates("category")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from %s", MODEL_DIR)
    model = FraudModel(MODEL_DIR)
    state["model"] = model

    # train+val only -- NOT data/raw/transactions.parquet. The raw file is the full,
    # unsplit dataset (train+val+test together); using it here would compute merchant/
    # category risk rates using the test period's own fraud labels, which are exactly the
    # transactions this service is about to score in the streaming demo. That's the same
    # class of leakage notebook 05's threshold calibration had (and was fixed for) -- the
    # model was trained on strictly causal, pre-test encodings, so serving must match that,
    # using only the historical data actually available before this service started.
    logger.info("Loading train+val history (merchant/category rates + per-card behavior)")
    history_frames = {}
    for name in ["train", "val"]:
        path = os.path.join(DATA_DIR, "processed", f"{name}.parquet")
        if os.path.exists(path):
            history_frames[name] = pd.read_parquet(
                path, columns=["user_id", "merchant_id", "category", "timestamp", "amount", "is_fraud"]
            )
    history = pd.concat(history_frames.values(), ignore_index=True).sort_values("timestamp")

    # The smoothing prior matches notebook 03's GLOBAL_RATE exactly: train's own fraud rate,
    # not train+val combined -- the encoding formula's prior must match what the model was
    # actually trained against.
    train_global_rate = float(history_frames["train"]["is_fraud"].mean()) if "train" in history_frames \
        else float(history["is_fraud"].mean())

    logger.info("Bootstrapping merchant/category risk encodings from train+val")
    merchant_rates, category_rates = _bootstrap_rate_dicts(history, train_global_rate)
    store = FeatureStore(merchant_rates, category_rates, train_global_rate)
    state["store"] = store

    logger.info("Bootstrapping per-card behavioral history from train+val")
    for row in history.itertuples(index=False):
        store.bootstrap_user(row.user_id, row.timestamp, row.amount)
        store.bootstrap_merchant(row.merchant_id, row.timestamp)
    logger.info("Bootstrapped %d cards from %d historical rows", store.n_users, len(history))

    nn_client = httpx.Client(base_url=NN_SERVICE_URL, timeout=2.0)
    logger.info("Waiting for the nn-service to be ready at %s ...", NN_SERVICE_URL)
    for _ in range(60):
        try:
            if nn_client.get("/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(2)
    else:
        raise RuntimeError(f"nn-service at {NN_SERVICE_URL} never became ready")
    state["nn_client"] = nn_client

    yield
    nn_client.close()
    state.clear()


app = FastAPI(title="Fraud Detection Scoring API", version="1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthOut)
def health():
    model, store = state.get("model"), state.get("store")
    if model is None or store is None:
        raise HTTPException(503, "not ready")
    return HealthOut(status="ok", model_version=model.model_version, users_tracked=store.n_users)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/score", response_model=ScoreOut)
def score(txn: TransactionIn):
    model, store, nn_client = state.get("model"), state.get("store"), state.get("nn_client")
    if model is None or store is None or nn_client is None:
        raise HTTPException(503, "not ready")

    start = time.perf_counter()
    features = store.compute_and_update(txn.model_dump())

    # The deployed model is a stack (notebook 04f): base features + the FT-Transformer's
    # own prediction as an extra feature. Scored out-of-process -- see NN_SERVICE_URL above.
    try:
        nn_resp = nn_client.post(
            "/score_nn", json={"features": features, "category": txn.category}
        )
        nn_resp.raise_for_status()
        features = {**features, "nn_score": nn_resp.json()["nn_score"]}
    except httpx.HTTPError as e:
        raise HTTPException(502, f"nn-service call failed: {e}")

    proba, decision, threshold, reasons = model.score(features, category=txn.category)
    latency_ms = (time.perf_counter() - start) * 1000

    SCORE_LATENCY.observe(latency_ms / 1000)
    SCORE_DISTRIBUTION.observe(proba)
    PREDICTIONS_TOTAL.labels(decision=decision).inc()

    return ScoreOut(
        transaction_id=txn.transaction_id,
        score=proba,
        decision=decision,
        threshold=threshold,
        latency_ms=latency_ms,
        top_reasons=reasons,
    )
