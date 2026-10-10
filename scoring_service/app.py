import os
import sys
import time
import json
import pickle
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis.asyncio as aioredis
from kafka import KafkaProducer
import numpy as np
import torch

# Ensure workspace root and ml are in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml"))

from ml.router import RouterConfig, LoadState, route
from ml.models import TabularDeepModel

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "artifacts")
BOOTSTRAP_SERVERS = "localhost:9092"
DECISIONS_TOPIC = "txn.decisions"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

app = FastAPI(title="Adaptive Dual-Path Payment Fraud Scoring Service", version="1.0.0")

class TransactionRequest(BaseModel):
    txnId: str
    tsEvent: int
    payerId: str
    payeeId: str
    deviceId: Optional[str] = "DEV-DEFAULT"
    amount: float
    currency: Optional[str] = "INR"
    channel: Optional[str] = "UPI"
    merchantCategory: Optional[str] = "RETAIL"
    ipGeo: Optional[str] = "IN-KA"
    mode: Optional[str] = None  # Per-request mode override ('always_deep', 'always_fast', 'adaptive')

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

# Prometheus metric collectors
SCORING_LATENCY = Histogram(
    "fraud_scoring_latency_seconds",
    "End-to-end scoring latency in seconds",
    ["route", "action"],
    buckets=[0.001, 0.002, 0.005, 0.010, 0.020, 0.035, 0.050, 0.100]
)
ROUTE_COUNTER = Counter("fraud_scoring_routes_total", "Count of transactions by route", ["route"])
FALLBACK_COUNTER = Counter("fraud_scoring_fallbacks_total", "Count of deadline timeouts falling back to fast model")
INFLIGHT_GAUGE = Gauge("fraud_scoring_inflight_requests", "Current active in-flight deep requests")
TPS_GAUGE = Gauge("fraud_scoring_current_tps", "Current EWMA TPS estimate")

# Global service state
state = {
    "prep": None,
    "fast_model": None,
    "calibrators": None,
    "deep_model": None,
    "redis": None,
    "kafka_producer": None,
    "load_state": LoadState(),
    "router_config": RouterConfig(
        lo=0.015,
        hi=0.80,
        high_value=20000.0,
        tps_soft=1000.0,
        tps_hard=2500.0,
        inflight_max=64,
        deadline_ms=18.0
    ),
    "mode": os.environ.get("SCORING_MODE", "adaptive"),  # 'adaptive', 'always_deep', 'always_fast'
    "metrics": {
        "total_requests": 0,
        "fast_routes": 0,
        "deep_routes": 0,
        "fallbacks": 0,
        "audit_routes": 0,
        "total_latency_ms": 0.0
    }
}

@app.on_event("startup")
async def startup_event():
    print("Initializing Scoring Service models, Redis, and Kafka...")
    # Load ML Artifacts
    with open(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"), "rb") as f:
        state["prep"] = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "fast_model.pkl"), "rb") as f:
        state["fast_model"] = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "calibrators.pkl"), "rb") as f:
        state["calibrators"] = pickle.load(f)

    # Load Deep PyTorch Model
    prep = state["prep"]
    cat_dims = [len(cat) + 1 for cat in prep["encoder"].categories_]
    deep = TabularDeepModel(cat_dims=cat_dims, num_dim=len(prep["num_cols"]))
    deep.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "deep_model.pth"), weights_only=True))
    deep.eval()
    state["deep_model"] = deep

    # Connect to Async Redis
    state["redis"] = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    # Connect Kafka Producer for txn.decisions
    try:
        state["kafka_producer"] = KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
            bootstrap_timeout_ms=1000
        )
    except Exception as e:
        print(f"Notice: Kafka broker not reached ({e}). Kafka telemetry disabled.")
        state["kafka_producer"] = None

    # Model Warmup to eliminate first-request cold-start latency spike
    state["num_mean"] = prep["scaler"].mean_.astype(np.float32)
    state["num_scale"] = prep["scaler"].scale_.astype(np.float32)
    state["cat_maps"] = [{cat: idx for idx, cat in enumerate(cats)} for cats in prep["encoder"].categories_]

    import concurrent.futures
    state["executor"] = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    warm_num = np.zeros((1, len(prep["num_cols"])), dtype=np.float32)
    warm_cat = np.zeros((1, len(prep["cat_cols"])), dtype=np.int64)
    _ = state["fast_model"].predict_proba(np.hstack([warm_num, warm_cat]))
    with torch.no_grad():
        _ = state["deep_model"](torch.tensor(warm_cat), torch.tensor(warm_num))
    print("Scoring Service successfully initialized with models warmed up.")

@app.on_event("shutdown")
async def shutdown_event():
    if state["redis"]:
        await state["redis"].aclose()
    if state["kafka_producer"]:
        state["kafka_producer"].close()
    if state.get("executor"):
        state["executor"].shutdown(wait=False)

def _sync_deep_eval(X_cat, X_num) -> float:
    with torch.no_grad():
        logits = state["deep_model"](torch.tensor(X_cat), torch.tensor(X_num))
        raw = torch.sigmoid(logits).item()
        cal = float(state["calibrators"]["deep"].predict([raw])[0])
        return cal

async def async_deep_inference(X_cat, X_num, simulate_delay_ms: float = 0.0) -> float:
    """Dispatches to deep model in threadpool without blocking the asyncio loop."""
    if simulate_delay_ms > 0:
        await asyncio.sleep(simulate_delay_ms / 1000.0)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(state["executor"], _sync_deep_eval, X_cat, X_num)

@app.post("/score")
async def score_endpoint(txn: TransactionRequest, test_delay_ms: float = 0.0):
    t_start = time.perf_counter()
    state["load_state"].tick()
    state["metrics"]["total_requests"] += 1

    r = state["redis"]
    payer_key = f"payer:{txn.payerId}"

    # 1. Fetch Online Velocity Features from Redis (< 0.2 ms)
    try:
        online_feats = await r.hmget(payer_key, [
            "cnt1m", "cnt10m", "cnt1h", "amtSum24h", "secsSinceLast", "distinctPayees1h"
        ])
    except Exception:
        online_feats = [None] * 6

    cnt1m = float(online_feats[0]) if online_feats[0] is not None else 0.0
    cnt10m = float(online_feats[1]) if online_feats[1] is not None else 0.0
    cnt1h = float(online_feats[2]) if online_feats[2] is not None else 0.0
    amtSum24h = float(online_feats[3]) if online_feats[3] is not None else 0.0
    secsSinceLast = float(online_feats[4]) if online_feats[4] is not None else -1.0
    distinctPayees1h = float(online_feats[5]) if online_feats[5] is not None else 0.0

    # 2. Fast direct vector standardization (< 0.005 ms)
    raw_num = np.array([txn.amount, cnt1m, cnt10m, cnt1h, amtSum24h, secsSinceLast, distinctPayees1h], dtype=np.float32)
    norm_num = (raw_num - state["num_mean"]) / state["num_scale"]
    X_num = norm_num.reshape(1, -1)

    cm = state["cat_maps"]
    c0 = cm[0].get(txn.channel, 0)
    c1 = cm[1].get(txn.merchantCategory, 0)
    c2 = cm[2].get(txn.ipGeo, 0)
    X_cat = np.array([[c0, c1, c2]], dtype=np.int64)
    X_all = np.hstack([X_num, X_cat])

    # 3. Fast Model In-Process (< 2 ms)
    fast_raw = state["fast_model"].predict_proba(X_all)[0, 1]
    fast_score = float(state["calibrators"]["fast"].predict([fast_raw])[0])

    mode = txn.mode or state.get("mode", "adaptive")
    if mode == "always_deep":
        decision_route = "deep"
    elif mode == "always_fast":
        decision_route = "fast"
    else:
        # Adaptive dual-path routing
        decision_route = route(fast_score, txn.amount, state["load_state"], state["router_config"])

    final_score = fast_score
    selected_route = "fast"

    if decision_route in ("deep", "deep_audit"):
        state["load_state"].inflight += 1
        INFLIGHT_GAUGE.set(state["load_state"].inflight)
        try:
            # Enforce hard deadline based on remaining total latency budget
            elapsed_before_deep = (time.perf_counter() - t_start) * 1000.0
            timeout_sec = max(0.005, (state["router_config"].deadline_ms - elapsed_before_deep) / 1000.0)
            deep_score = await asyncio.wait_for(
                async_deep_inference(X_cat, X_num, simulate_delay_ms=test_delay_ms),
                timeout=timeout_sec
            )
            final_score = deep_score
            selected_route = decision_route
            state["metrics"]["deep_routes"] += 1
            ROUTE_COUNTER.labels(route=decision_route).inc()
        except asyncio.TimeoutError:
            # Deadline expired! Return fast-path score immediately
            final_score = fast_score
            selected_route = "fast_fallback"
            state["metrics"]["fallbacks"] += 1
            ROUTE_COUNTER.labels(route="fast_fallback").inc()
            FALLBACK_COUNTER.inc()
        finally:
            state["load_state"].inflight -= 1
            INFLIGHT_GAUGE.set(state["load_state"].inflight)
    else:
        state["metrics"]["fast_routes"] += 1
        ROUTE_COUNTER.labels(route="fast").inc()

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    state["metrics"]["total_latency_ms"] += elapsed_ms
    TPS_GAUGE.set(state["load_state"].tps)

    # 5. Determine Business Action
    if final_score < 0.15:
        action = "APPROVE"
    elif final_score < 0.70:
        action = "REVIEW"
    else:
        action = "DECLINE"

    SCORING_LATENCY.labels(route=selected_route, action=action).observe(elapsed_ms / 1000.0)

    # 6. Publish decision to Kafka txn.decisions
    decision_payload = {
        "txnId": txn.txnId,
        "payerId": txn.payerId,
        "score": round(final_score, 4),
        "action": action,
        "route": selected_route,
        "latencyMs": round(elapsed_ms, 2),
        "loadFactor": round(state["load_state"].tps / state["router_config"].tps_hard, 3),
        "timestamp": int(time.time() * 1000)
    }

    try:
        state["kafka_producer"].send(DECISIONS_TOPIC, key=txn.payerId, value=decision_payload)
    except Exception as e:
        pass

    return decision_payload

@app.get("/metrics")
async def metrics_endpoint():
    total = max(1, state["metrics"]["total_requests"])
    avg_latency = state["metrics"]["total_latency_ms"] / total
    deep_share = (state["metrics"]["deep_routes"] / total) * 100.0
    return {
        "mode": state.get("mode", "adaptive"),
        "total_requests": state["metrics"]["total_requests"],
        "fast_routes": state["metrics"]["fast_routes"],
        "deep_routes": state["metrics"]["deep_routes"],
        "deep_share_pct": round(deep_share, 2),
        "fallbacks": state["metrics"]["fallbacks"],
        "avg_latency_ms": round(avg_latency, 2),
        "current_tps": round(state["load_state"].tps, 1),
        "current_inflight": state["load_state"].inflight
    }

@app.post("/set_mode")
async def set_mode_endpoint(mode: str):
    if mode not in ("adaptive", "always_deep", "always_fast"):
        raise HTTPException(status_code=400, detail="Invalid mode")
    state["mode"] = mode
    # Reset metrics for comparative runs
    state["metrics"] = {
        "total_requests": 0,
        "fast_routes": 0,
        "deep_routes": 0,
        "fallbacks": 0,
        "audit_routes": 0,
        "total_latency_ms": 0.0
    }
    return {"status": "ok", "mode": mode}

@app.post("/reset_metrics")
async def reset_metrics_endpoint():
    state["metrics"] = {
        "total_requests": 0,
        "fast_routes": 0,
        "deep_routes": 0,
        "fallbacks": 0,
        "audit_routes": 0,
        "total_latency_ms": 0.0
    }
    return {"status": "ok"}

# Instrument with Prometheus (exposes /prometheus endpoint for Grafana/Prometheus scraper)
Instrumentator().instrument(app).expose(app, endpoint="/prometheus")

@app.get("/health")
async def health_endpoint():
    return {"status": "ok", "service": "scoring-service"}
