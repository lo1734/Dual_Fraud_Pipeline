import os
import sys
import time
import json
import random
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from hdrh.histogram import HdrHistogram

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
BASE_URL = "http://127.0.0.1:8000"

def calculate_recall_at_fpr(y_true, y_score, target_fpr=0.01):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # find highest tpr where fpr <= target_fpr
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 0.0
    return float(tpr[valid_idx[-1]])

async def set_service_mode(session: aiohttp.ClientSession, mode: str):
    async with session.post(f"{BASE_URL}/set_mode", params={"mode": mode}) as resp:
        return await resp.json()

async def reset_metrics(session: aiohttp.ClientSession):
    async with session.post(f"{BASE_URL}/reset_metrics") as resp:
        return await resp.json()

async def get_service_metrics(session: aiohttp.ClientSession):
    async with session.get(f"{BASE_URL}/metrics") as resp:
        return await resp.json()

async def run_scenario_with_evaluation(mode: str, test_df: pd.DataFrame, num_samples: int = 5000, target_tps: float = 1000.0):
    """
    Runs an open arrival load test for a specific mode and calculates latency percentiles + ML metrics.
    """
    hist = HdrHistogram(1, 200000, 3)
    results = []

    print(f"\n================================================================================")
    print(f"EVALUATING ARCHITECTURE: {mode.upper()} ({num_samples} transactions @ {target_tps} TPS)")
    print(f"================================================================================")

    connector = aiohttp.TCPConnector(limit=2000, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        await set_service_mode(session, mode)
        await reset_metrics(session)

        # Sample test dataframe
        sample_df = test_df.sample(n=min(num_samples, len(test_df)), random_state=42).reset_index(drop=True)

        tasks = []
        start_time = time.perf_counter()
        current_time = start_time
        mean_interval = 1.0 / target_tps

        async def _worker(row_dict, scheduled_ts):
            try:
                amt = float(row_dict["amount"])
                req_body = {
                    "txnId": f"{mode}-{row_dict['txn_id']}",
                    "tsEvent": int(time.time() * 1000),
                    "payerId": str(row_dict["payer_id"]),
                    "payeeId": str(row_dict["payee_id"]),
                    "deviceId": "DEV-EVAL",
                    "amount": round(amt, 2),
                    "currency": "INR",
                    "channel": str(row_dict["channel"]),
                    "merchantCategory": str(row_dict["merchant_category"]),
                    "ipGeo": str(row_dict["ip_geo"]),
                    "mode": mode
                }
                timeout = aiohttp.ClientTimeout(total=0.060)
                req_send = time.perf_counter()
                async with session.post(f"{BASE_URL}/score", json=req_body, timeout=timeout) as resp:
                    resp_data = await resp.json()
                    finish = time.perf_counter()
                    # Server round-trip latency
                    lat_us = int(max(1, (finish - req_send) * 1_000_000))
                    hist.record_value(min(200000, lat_us))
                    return {
                        "score": resp_data.get("score", 0.0),
                        "route": resp_data.get("route", "error"),
                        "label": int(row_dict["is_fraud"]),
                        "latency_ms": lat_us / 1000.0
                    }
            except Exception as e:
                finish = time.perf_counter()
                lat_us = int(max(1, (finish - req_send) * 1_000_000)) if 'req_send' in locals() else 60000
                hist.record_value(min(200000, lat_us))
                return {
                    "score": 0.0,
                    "route": "error",
                    "label": int(row_dict["is_fraud"]),
                    "latency_ms": lat_us / 1000.0
                }

        for idx, row in sample_df.iterrows():
            now = time.perf_counter()
            if current_time > now:
                await asyncio.sleep(current_time - now)
            t = asyncio.create_task(_worker(row.to_dict(), current_time))
            tasks.append(t)
            # Poisson inter-arrival delta
            delta = -np.log(max(1e-9, random.random())) * mean_interval
            current_time = max(time.perf_counter(), current_time + delta)

        responses = await asyncio.gather(*tasks)
        elapsed_sec = time.perf_counter() - start_time
        metrics = await get_service_metrics(session)

    # Process metrics
    scores = np.array([r["score"] for r in responses])
    labels = np.array([r["label"] for r in responses])
    routes = [r["route"] for r in responses]

    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = float(auc(recall, precision))
    recall_1pct = calculate_recall_at_fpr(labels, scores, target_fpr=0.01)

    p50 = hist.get_value_at_percentile(50.0) / 1000.0
    p90 = hist.get_value_at_percentile(90.0) / 1000.0
    p95 = hist.get_value_at_percentile(95.0) / 1000.0
    p99 = hist.get_value_at_percentile(99.0) / 1000.0
    p99_9 = hist.get_value_at_percentile(99.9) / 1000.0
    achieved_tps = len(responses) / elapsed_sec

    deep_routes = sum(1 for r in routes if r in ("deep", "deep_audit"))
    fallbacks = sum(1 for r in routes if r == "fast_fallback")
    fast_routes = sum(1 for r in routes if r == "fast")
    deep_pct = (deep_routes / len(responses)) * 100.0
    fallback_pct = (fallbacks / len(responses)) * 100.0

    report = {
        "Configuration": mode,
        "TPS Sustained": round(achieved_tps, 1),
        "p50 Latency": f"{p50:.2f} ms",
        "p90 Latency": f"{p90:.2f} ms",
        "p99 Latency": f"{p99:.2f} ms",
        "p99.9 Latency": f"{p99_9:.2f} ms",
        "p99_raw": p99,
        "p99_9_raw": p99_9,
        "Deep Share %": f"{deep_pct:.1f}%",
        "PR-AUC": round(pr_auc, 4),
        "Recall@1%FPR": f"{recall_1pct * 100.0:.2f}%",
        "Fallback %": f"{fallback_pct:.2f}%",
        "Fallbacks": fallbacks,
        "Deep Count": deep_routes,
        "Fast Count": fast_routes
    }
    return report

async def main():
    print("Loading test split for matrix benchmarking...")
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
    print(f"Loaded {len(test_df)} test transactions (Fraud rate: {test_df['is_fraud'].mean()*100:.2f}%)")

    # Evaluate all 3 target architectures under load
    deep_report = await run_scenario_with_evaluation("always_deep", test_df, num_samples=3000, target_tps=600.0)
    fast_report = await run_scenario_with_evaluation("always_fast", test_df, num_samples=3000, target_tps=1200.0)
    dual_report = await run_scenario_with_evaluation("adaptive", test_df, num_samples=3000, target_tps=1000.0)

    # Calculate Quality Retention relative to Always-Deep baseline
    deep_pr_auc = deep_report["PR-AUC"]
    adaptive_retention = (dual_report["PR-AUC"] / deep_pr_auc) * 100.0
    fast_retention = (fast_report["PR-AUC"] / deep_pr_auc) * 100.0

    print("\n" + "=" * 95)
    print("PHASE 3 TARGET EVALUATION MATRIX & SLO VALIDATION")
    print("=" * 95)
    header = f"{'Configuration':<22} | {'TPS':<8} | {'p99':<9} | {'p99.9':<9} | {'Deep %':<8} | {'PR-AUC':<8} | {'Rec@1%FPR':<10} | {'Fallback %':<10}"
    print(header)
    print("-" * 95)

    for rep in [deep_report, fast_report, dual_report]:
        line = f"{rep['Configuration']:<22} | {rep['TPS Sustained']:<8} | {rep['p99 Latency']:<9} | {rep['p99.9 Latency']:<9} | {rep['Deep Share %']:<8} | {rep['PR-AUC']:<8} | {rep['Recall@1%FPR']:<10} | {rep['Fallback %']:<10}"
        print(line)

    print("-" * 95)
    print(f"Adaptive Dual-Path Quality Retention: {adaptive_retention:.2f}% of Always-Deep Baseline")
    print(f"Always-Fast Quality Retention:       {fast_retention:.2f}% of Always-Deep Baseline")
    print(f"End-to-End Latency Target:           p99.9 <= 50.0 ms -> {'✅ PASSED' if dual_report['p99_9_raw'] <= 50.0 else '❌ FAILED'}")
    print(f"Fallback Safety Control:             Fallback rate = {dual_report['Fallback %']} under live load")
    print("=" * 95)

    # Save results to JSON
    summary_path = os.path.join(WORKSPACE_DIR, "benchmarks", "evaluation_matrix.json")
    with open(summary_path, "w") as f:
        json.dump({
            "deep": deep_report,
            "fast": fast_report,
            "adaptive": dual_report,
            "adaptive_retention_pct": round(adaptive_retention, 2),
            "fast_retention_pct": round(fast_retention, 2)
        }, f, indent=2)
    print(f"Detailed evaluation matrix saved to {summary_path}")

if __name__ == "__main__":
    asyncio.run(main())
