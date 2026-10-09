import os
import sys

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import redis
from feature_pusher import get_feast_store

NUM_SAMPLES = 5000
SAMPLE_PAYERS = ["P64", "P26", "P63", "P41", "P12", "P44", "P90", "P10", "P30", "P31"]

def benchmark_direct_redis():
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    latencies = []
    
    # Warmup
    for _ in range(100):
        r.hmget("payer:P64", "cnt1m", "cnt10m", "cnt1h", "amtSum24h", "secsSinceLast", "distinctPayees1h")

    for i in range(NUM_SAMPLES):
        payer = SAMPLE_PAYERS[i % len(SAMPLE_PAYERS)]
        t0 = time.perf_counter()
        res = r.hmget(f"payer:{payer}", "cnt1m", "cnt10m", "cnt1h", "amtSum24h", "secsSinceLast", "distinctPayees1h")
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # in ms

    return np.array(latencies)

def benchmark_direct_redis_hgetall():
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    latencies = []
    
    # Warmup
    for _ in range(100):
        r.hgetall("payer:P64")

    for i in range(NUM_SAMPLES):
        payer = SAMPLE_PAYERS[i % len(SAMPLE_PAYERS)]
        t0 = time.perf_counter()
        res = r.hgetall(f"payer:{payer}")
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return np.array(latencies)

def benchmark_feast_python_sdk():
    store = get_feast_store()
    features = [
        "velocity:cnt_1m",
        "velocity:cnt_10m",
        "velocity:cnt_1h",
        "velocity:amt_sum_24h",
        "velocity:secs_since_last",
        "velocity:distinct_payees_1h"
    ]
    latencies = []

    # Warmup
    for _ in range(50):
        store.get_online_features(features=features, entity_rows=[{"payer_id": "P64"}])

    for i in range(1000): # 1000 samples for Feast SDK due to Python overhead
        payer = SAMPLE_PAYERS[i % len(SAMPLE_PAYERS)]
        t0 = time.perf_counter()
        res = store.get_online_features(features=features, entity_rows=[{"payer_id": payer}])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return np.array(latencies)

def print_metrics(name, latencies):
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    p99_9 = np.percentile(latencies, 99.9)
    print(f"| {name:<30} | {p50:6.3f} ms | {p90:6.3f} ms | {p95:6.3f} ms | {p99:6.3f} ms | {p99_9:6.3f} ms |")

def main():
    print("=" * 80)
    print("ONLINE FEATURE RETRIEVAL BENCHMARK (Phase 1.5 Decision Gate)")
    print("SLO Requirement: p99.9 < 5.0 ms")
    print("=" * 80)

    print("\nRunning Direct Redis HMGET Benchmark (5,000 requests)...")
    lat_hmget = benchmark_direct_redis()

    print("Running Direct Redis HGETALL Benchmark (5,000 requests)...")
    lat_hgetall = benchmark_direct_redis_hgetall()

    print("Running Feast Python SDK Benchmark (1,000 requests)...")
    lat_feast = benchmark_feast_python_sdk()

    print("\n" + "=" * 80)
    print(f"| {'Retrieval Strategy':<30} | {'p50':<9} | {'p90':<9} | {'p95':<9} | {'p99':<9} | {'p99.9':<9} |")
    print("|" + "-" * 32 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|")
    print_metrics("Direct Redis HMGET (Hot Path)", lat_hmget)
    print_metrics("Direct Redis HGETALL", lat_hgetall)
    print_metrics("Feast Python SDK (Online Store)", lat_feast)
    print("=" * 80)

    print("\nDecision Gate Conclusion:")
    if np.percentile(lat_hmget, 99.9) < 5.0:
        print("✅ Direct Redis HMGET satisfies p99.9 < 5.0 ms SLO.")
    else:
        print("❌ Direct Redis HMGET did not meet p99.9 < 5.0 ms.")

    if np.percentile(lat_feast, 99.9) < 5.0:
        print("✅ Feast Python SDK satisfies p99.9 < 5.0 ms SLO.")
    else:
        print("⚠️ Feast Python SDK has higher latency due to Python protobuf/deserialization overhead. Direct Redis hash reads recommended for hot path.")

if __name__ == "__main__":
    main()
