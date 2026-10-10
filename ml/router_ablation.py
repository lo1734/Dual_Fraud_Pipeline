import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_curve

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_DIR, "ml"))
from models import TabularDeepModel

DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "ml", "artifacts")
BENCHMARKS_DIR = os.path.join(WORKSPACE_DIR, "benchmarks")

def calculate_recall_at_fpr(y_true, y_score, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid_idx = np.where(fpr <= target_fpr)[0]
    return float(tpr[valid_idx[-1]]) if len(valid_idx) > 0 else 0.0

def run_router_ablation():
    print("=" * 80)
    print("STEP 3: FORMAL ROUTER VARIANTS ABLATION SUITE (COST VS QUALITY)")
    print("=" * 80)

    # 1. Load artifacts and test set
    with open(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"), "rb") as f:
        prep = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "fast_model.pkl"), "rb") as f:
        fast_model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "calibrators.pkl"), "rb") as f:
        calibrators = pickle.load(f)

    df_test = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
    y_test = df_test["is_fraud"].values
    amounts = df_test["amount"].values

    # 2. Compute Raw Scores
    X_cat = prep["encoder"].transform(df_test[prep["cat_cols"]]).astype(np.int64)
    X_cat[X_cat < 0] = 0
    X_num = prep["scaler"].transform(df_test[prep["num_cols"]]).astype(np.float32)
    X_all = np.hstack([X_num, X_cat])

    fast_raw = fast_model.predict_proba(X_all)[:, 1]
    fast_cal = calibrators["fast"].predict(fast_raw)

    cat_dims = [len(cat) + 1 for cat in prep["encoder"].categories_]
    deep_model = TabularDeepModel(cat_dims=cat_dims, num_dim=len(prep["num_cols"]))
    deep_model.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "deep_model.pth"), weights_only=True))
    deep_model.eval()

    with torch.no_grad():
        deep_logits = deep_model(torch.tensor(X_cat), torch.tensor(X_num))
        deep_raw = torch.sigmoid(deep_logits).numpy()
        deep_cal = calibrators["deep"].predict(deep_raw)

    deep_prauc = average_precision_score(y_test, deep_cal)
    deep_rec1 = calculate_recall_at_fpr(y_test, deep_cal, target_fpr=0.01)

    fast_prauc = average_precision_score(y_test, fast_cal)
    fast_rec1 = calculate_recall_at_fpr(y_test, fast_cal, target_fpr=0.01)

    print(f"Always-Deep Baseline: PR-AUC = {deep_prauc:.4f} | Recall@1%FPR = {deep_rec1*100:.2f}% | Deep Share = 100.0%")
    print(f"Always-Fast Baseline: PR-AUC = {fast_prauc:.4f} | Recall@1%FPR = {fast_rec1*100:.2f}% | Deep Share = 0.0%\n")

    ablations = []

    # Variant 1: Confidence-Only Routing (Uncertainty threshold only, no amount, no load shedding)
    for delta in [0.10, 0.20, 0.35, 0.45]:
        mask = np.abs(fast_cal - 0.5) <= delta
        routed_scores = np.where(mask, deep_cal, fast_cal)
        share = float(np.mean(mask) * 100.0)
        prauc = average_precision_score(y_test, routed_scores)
        rec = calculate_recall_at_fpr(y_test, routed_scores, target_fpr=0.01)
        ablations.append({
            "variant": "Confidence-Only",
            "param": f"delta={delta:.2f}",
            "deep_share_pct": round(share, 1),
            "pr_auc": round(prauc, 4),
            "recall_1pct_fpr": round(rec * 100.0, 2),
            "retention_pct": round((prauc / deep_prauc) * 100.0, 2)
        })

    # Variant 2: Amount-Only Routing (Threshold on amount only, no model confidence)
    for amt_thresh in [500.0, 1000.0, 5000.0, 20000.0]:
        mask = amounts >= amt_thresh
        routed_scores = np.where(mask, deep_cal, fast_cal)
        share = float(np.mean(mask) * 100.0)
        prauc = average_precision_score(y_test, routed_scores)
        rec = calculate_recall_at_fpr(y_test, routed_scores, target_fpr=0.01)
        ablations.append({
            "variant": "Amount-Only",
            "param": f"amount>={amt_thresh:.0f}",
            "deep_share_pct": round(share, 1),
            "pr_auc": round(prauc, 4),
            "recall_1pct_fpr": round(rec * 100.0, 2),
            "retention_pct": round((prauc / deep_prauc) * 100.0, 2)
        })

    # Variant 3: Load-Only Routing (Fixed random offload, no confidence or amount prioritization)
    np.random.seed(42)
    for target_share in [20.0, 40.0, 60.0]:
        mask = np.random.rand(len(df_test)) < (target_share / 100.0)
        routed_scores = np.where(mask, deep_cal, fast_cal)
        share = float(np.mean(mask) * 100.0)
        prauc = average_precision_score(y_test, routed_scores)
        rec = calculate_recall_at_fpr(y_test, routed_scores, target_fpr=0.01)
        ablations.append({
            "variant": "Load-Only (Random)",
            "param": f"prob={target_share:.0f}%",
            "deep_share_pct": round(share, 1),
            "pr_auc": round(prauc, 4),
            "recall_1pct_fpr": round(rec * 100.0, 2),
            "retention_pct": round((prauc / deep_prauc) * 100.0, 2)
        })

    # Variant 4: Combined Adaptive Routing (Uncertainty band + high-value bypass + load shedding)
    for lo, hi in [(0.02, 0.70), (0.02, 0.60), (0.04, 0.50)]:
        center = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        mask = (np.abs(fast_cal - center) <= half) | (amounts >= 20000.0)
        routed_scores = np.where(mask, deep_cal, fast_cal)
        share = float(np.mean(mask) * 100.0)
        prauc = average_precision_score(y_test, routed_scores)
        rec = calculate_recall_at_fpr(y_test, routed_scores, target_fpr=0.01)
        ablations.append({
            "variant": "Combined Adaptive (Ours)",
            "param": f"[{lo:.2f}, {hi:.2f}] + high-val",
            "deep_share_pct": round(share, 1),
            "pr_auc": round(prauc, 4),
            "recall_1pct_fpr": round(rec * 100.0, 2),
            "retention_pct": round((prauc / deep_prauc) * 100.0, 2)
        })

    # Format Output Table
    print(f"| {'Router Variant':<25} | {'Parameter':<22} | {'Deep Share %':<12} | {'PR-AUC':<8} | {'Recall@1%FPR':<12} | {'Retention %':<12} |")
    print("|" + "-" * 27 + "|" + "-" * 24 + "|" + "-" * 14 + "|" + "-" * 10 + "|" + "-" * 14 + "|" + "-" * 14 + "|")
    for r in ablations:
        print(f"| {r['variant']:<25} | {r['param']:<22} | {r['deep_share_pct']:10.1f}% | {r['pr_auc']:8.4f} | {r['recall_1pct_fpr']:10.2f}% | {r['retention_pct']:10.2f}% |")

    # Save to JSON
    out_path = os.path.join(BENCHMARKS_DIR, "router_ablation_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "deep_baseline": {"pr_auc": deep_prauc, "recall_1pct": deep_rec1},
            "fast_baseline": {"pr_auc": fast_prauc, "recall_1pct": fast_rec1},
            "ablations": ablations
        }, f, indent=2)

    print(f"\n✅ Router ablation results successfully written to: {out_path}")
    return ablations

if __name__ == "__main__":
    run_router_ablation()
