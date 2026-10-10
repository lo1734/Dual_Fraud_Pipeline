import os
import sys
import time
import json
import numpy as np
import pandas as pd
import requests

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from ml.feedback.delayed_label_joiner import DelayedLabelJoiner
from ml.monitoring.drift_detector import FeatureDriftDetector, compute_model_score_divergence

BASE_URL = "http://127.0.0.1:8000"
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")

def test_1_delayed_label_feedback_pipeline():
    """
    Test 1: Verifies delayed-label joiner matching transactions from txn.labels
    with historical decisions from txn.decisions, persisting outcomes and computing
    realized false discovery and missed fraud rates.
    """
    print("\n--- Test 1: Delayed-Label Feedback & Temporal Join Engine ---")
    test_db = os.path.join(DATA_DIR, "test_feedback.db")
    if os.path.exists(test_db):
        os.remove(test_db)

    joiner = DelayedLabelJoiner(db_path=test_db)

    # 1. Simulate 5 historical scoring decisions
    decisions = [
        {"txnId": "TXN-FEEDBACK-01", "payerId": "P10", "score": 0.02, "action": "APPROVE", "route": "fast", "latencyMs": 1.5, "loadFactor": 0.1, "timestamp": 1700000000000},
        {"txnId": "TXN-FEEDBACK-02", "payerId": "P20", "score": 0.92, "action": "DECLINE", "route": "deep", "latencyMs": 12.0, "loadFactor": 0.2, "timestamp": 1700000010000},
        {"txnId": "TXN-FEEDBACK-03", "payerId": "P30", "score": 0.05, "action": "APPROVE", "route": "fast", "latencyMs": 1.2, "loadFactor": 0.1, "timestamp": 1700000020000},
        {"txnId": "TXN-FEEDBACK-04", "payerId": "P40", "score": 0.88, "action": "DECLINE", "route": "deep", "latencyMs": 15.1, "loadFactor": 0.3, "timestamp": 1700000030000},
        {"txnId": "TXN-FEEDBACK-05", "payerId": "P50", "score": 0.35, "action": "REVIEW", "route": "fast_fallback", "latencyMs": 24.0, "loadFactor": 0.8, "timestamp": 1700000040000},
    ]
    for d in decisions:
        joiner.record_decision(d)

    # 2. Simulate delayed chargeback & legitimate dispute labels arriving days later
    labels = [
        {"txnId": "TXN-FEEDBACK-01", "isFraud": 0, "chargebackCode": "NONE", "timestamp": 1701200000000},      # True Negative (Approved & Legit)
        {"txnId": "TXN-FEEDBACK-02", "isFraud": 1, "chargebackCode": "CB_FRAUD_104", "timestamp": 1701200010000}, # True Positive (Declined & Fraud)
        {"txnId": "TXN-FEEDBACK-03", "isFraud": 1, "chargebackCode": "CB_FRAUD_104", "timestamp": 1701200020000}, # False Negative (Approved & Chargebacked!) -> Missed Fraud
        {"txnId": "TXN-FEEDBACK-04", "isFraud": 0, "chargebackCode": "DISPUTE_LEGIT", "timestamp": 1701200030000},# False Positive (Declined but Legit!) -> False Discovery
        {"txnId": "TXN-FEEDBACK-05", "isFraud": 1, "chargebackCode": "CB_FRAUD_104", "timestamp": 1701200040000}, # True Positive (Review/Decline fraud)
    ]
    for lbl in labels:
        joined = joiner.record_label(lbl)
        assert joined is not None, f"Failed to join label for txn {lbl['txnId']}"

    # 3. Compute operational metrics
    metrics = joiner.compute_operational_metrics()
    print("Computed Realized Operational Metrics:", json.dumps(metrics, indent=2))

    assert metrics["total_labeled"] == 5
    assert metrics["total_fraud"] == 3
    assert metrics["missed_fraud_count"] == 1  # TXN-FEEDBACK-03
    assert metrics["false_discovery_count"] == 1  # TXN-FEEDBACK-04
    if os.path.exists(test_db):
        os.remove(test_db)
    print("✅ Delayed-label join engine accurately tracks realized fraud and false discoveries.")

def test_2_feature_and_model_drift_monitoring():
    """
    Test 2: Verifies feature distribution drift detection (PSI, Wasserstein)
    and Fast vs Deep model score divergence tracking.
    """
    print("\n--- Test 2: Feature & Model Drift Monitoring ---")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    feature_cols = ["amount", "cnt_1m", "cnt_10m", "cnt_1h", "amt_sum_24h", "secs_since_last", "distinct_payees_1h"]

    # 1. Normal holdout distribution (should show low PSI < 0.1)
    ref_df = train_df.sample(n=10000, random_state=42)
    detector = FeatureDriftDetector(ref_df, feature_cols)

    test_normal = train_df.drop(ref_df.index).sample(n=5000, random_state=99)
    normal_report = detector.evaluate_drift_report(test_normal)
    print("In-Distribution Baseline PSI Report:", normal_report["feature_psi"])
    for col, psi in normal_report["feature_psi"].items():
        assert psi < 0.10, f"Unexpected high PSI on normal split for {col}: {psi}"
    assert normal_report["action_required"] == "NONE"

    # 2. Synthetic Drift Injection: shift amount and velocity counts
    drifted_df = test_normal.copy()
    drifted_df["amount"] = drifted_df["amount"] * 8.5 + 5000.0  # Massive monetary inflation
    drifted_df["cnt_1m"] = drifted_df["cnt_1m"] + 15             # Severe velocity spike

    drift_report = detector.evaluate_drift_report(drifted_df)
    print("Drift Injected Report:", drift_report)

    assert drift_report["feature_psi"]["amount"] > 0.25, "Failed to catch severe amount drift"
    assert drift_report["action_required"] == "RETRAIN", "Drift detector should flag RETRAIN"
    print("✅ Feature drift detector successfully caught synthetic distribution shift (PSI > 0.25).")

    # 3. Model Score Divergence
    fast_scores = np.array([0.05, 0.10, 0.85, 0.90, 0.02])
    deep_scores = np.array([0.06, 0.12, 0.88, 0.89, 0.01])
    divergence = compute_model_score_divergence(fast_scores, deep_scores)
    assert not divergence["divergence_alert"]

    drifted_fast = np.array([0.80, 0.85, 0.10, 0.15, 0.90])  # Inverted predictions
    divergence_drift = compute_model_score_divergence(drifted_fast, deep_scores)
    assert divergence_drift["divergence_alert"]
    print("✅ Model score divergence successfully caught prediction mismatch.")

def test_3_shadow_mode_execution():
    """
    Test 3: Verifies Shadow Mode activation, ensuring candidate deep model
    is evaluated asynchronously with zero latency penalty on live responses.
    """
    print("\n--- Test 3: Shadow Mode Asynchronous Execution ---")
    # Activate shadow mode
    r_shadow = requests.post(f"{BASE_URL}/deploy/shadow").json()
    assert r_shadow["status"] == "ok"
    assert r_shadow["deployment"] == "shadow"

    # Send a scoring request
    test_req = {
        "txnId": f"SHADOW-TEST-{int(time.time()*1000)}",
        "tsEvent": int(time.time() * 1000),
        "payerId": "P10",
        "payeeId": "M100",
        "amount": 500.0,
        "channel": "UPI",
        "merchantCategory": "RETAIL",
        "ipGeo": "IN-KA"
    }

    t0 = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/score", json=test_req).json()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Shadow request live response: {resp} (took {elapsed_ms:.2f} ms)")
    assert resp["txnId"] == test_req["txnId"]
    assert elapsed_ms < 50.0, f"Shadow mode request took too long: {elapsed_ms} ms (SLO is 50.0 ms)"

    # Check shadow telemetry after short async execution
    time.sleep(0.1)
    status = requests.get(f"{BASE_URL}/deploy/status").json()
    print("Deployment Status after shadow request:", status)
    assert status["shadow_requests"] >= 1
    print("✅ Shadow Mode executed asynchronously without user-facing latency impact.")

def test_4_canary_routing_and_rollback():
    """
    Test 4: Verifies Canary Mode traffic allocation, promotion, and instant rollback.
    """
    print("\n--- Test 4: Canary Traffic Allocation & Rollback Controller ---")
    # 1. Activate Canary mode with 25% traffic
    r_canary = requests.post(f"{BASE_URL}/deploy/canary", params={"traffic_pct": 25.0}).json()
    assert r_canary["status"] == "ok"
    assert r_canary["traffic_pct"] == 25.0

    status = requests.get(f"{BASE_URL}/deploy/status").json()
    assert status["status"] == "canary"
    assert status["canary_traffic_pct"] == 25.0

    # 2. Verify Promote to Production
    r_promote = requests.post(f"{BASE_URL}/deploy/promote").json()
    assert r_promote["status"] == "ok"
    assert r_promote["deployment"] == "promoted_to_production"

    status = requests.get(f"{BASE_URL}/deploy/status").json()
    assert status["status"] == "production"
    assert status["canary_traffic_pct"] == 0.0

    # 3. Verify Rollback back to Baseline
    r_rollback = requests.post(f"{BASE_URL}/deploy/rollback").json()
    assert r_rollback["status"] == "ok"
    assert r_rollback["deployment"] == "rolled_back_to_baseline"

    status = requests.get(f"{BASE_URL}/deploy/status").json()
    assert status["status"] == "production"
    print("✅ Canary deployment, promotion, and instant rollback verified successfully.")

def main():
    print("=" * 80)
    print("PHASE 4 COMPREHENSIVE OPERATIONAL HARDENING VERIFICATION SUITE")
    print("=" * 80)
    test_1_delayed_label_feedback_pipeline()
    test_2_feature_and_model_drift_monitoring()
    test_3_shadow_mode_execution()
    test_4_canary_routing_and_rollback()
    print("\n" + "=" * 80)
    print("🎉 ALL PHASE 4 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
