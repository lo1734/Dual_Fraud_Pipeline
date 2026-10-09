import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

NUMERICAL_COLS = ["amount", "cnt_1m", "cnt_10m", "cnt_1h", "amt_sum_24h", "secs_since_last", "distinct_payees_1h"]
CATEGORICAL_COLS = ["channel", "merchant_category", "ip_geo"]

class TabularDeepModel(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dim=8, hidden_dim=64):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes in cat_dims
        ])
        total_in_dim = (len(cat_dims) * emb_dim) + num_dim

        self.net = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_cat, x_num):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(emb_outs + [x_num], dim=1)
        return self.net(x).squeeze(1)

def recall_at_fixed_fpr(y_true, y_scores, target_fpr=0.01):
    """Calculates Recall at fixed False Positive Rate (e.g. 1% FPR)."""
    neg_scores = y_scores[y_true == 0]
    pos_scores = y_scores[y_true == 1]
    threshold = np.percentile(neg_scores, (1.0 - target_fpr) * 100.0)
    recall = np.mean(pos_scores >= threshold)
    return recall, threshold

def train_and_evaluate():
    print("=" * 80)
    print("PHASE 2: DUAL MODEL TRAINING & PROBABILITY CALIBRATION")
    print("=" * 80)

    # 1. Load Datasets
    df_train = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    df_val = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    df_test = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

    y_train = df_train["is_fraud"].values
    y_val = df_val["is_fraud"].values
    y_test = df_test["is_fraud"].values

    # 2. Preprocessing
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    scaler = StandardScaler()

    X_train_cat = encoder.fit_transform(df_train[CATEGORICAL_COLS]).astype(np.int64)
    X_val_cat = encoder.transform(df_val[CATEGORICAL_COLS]).astype(np.int64)
    X_test_cat = encoder.transform(df_test[CATEGORICAL_COLS]).astype(np.int64)

    # Replace -1 with 0 for embedding safety
    X_train_cat[X_train_cat < 0] = 0
    X_val_cat[X_val_cat < 0] = 0
    X_test_cat[X_test_cat < 0] = 0

    X_train_num = scaler.fit_transform(df_train[NUMERICAL_COLS]).astype(np.float32)
    X_val_num = scaler.transform(df_val[NUMERICAL_COLS]).astype(np.float32)
    X_test_num = scaler.transform(df_test[NUMERICAL_COLS]).astype(np.float32)

    X_train_all = np.hstack([X_train_num, X_train_cat])
    X_val_all = np.hstack([X_val_num, X_val_cat])
    X_test_all = np.hstack([X_test_num, X_test_cat])

    # 3. Train Fast Model (LightGBM)
    print("\n[1/4] Training Fast Model (LightGBM shallow trees for sub-2ms latency)...")
    fast_model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.08,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    fast_model.fit(X_train_all, y_train)

    val_fast_raw = fast_model.predict_proba(X_val_all)[:, 1]
    test_fast_raw = fast_model.predict_proba(X_test_all)[:, 1]

    # 4. Train Deep Model (PyTorch Tabular DNN)
    print("\n[2/4] Training Deep Model (High-capacity Tabular Neural Network)...")
    cat_dims = [len(cat) + 1 for cat in encoder.categories_]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deep_model = TabularDeepModel(cat_dims=cat_dims, num_dim=len(NUMERICAL_COLS)).to(device)

    train_ds = TensorDataset(torch.tensor(X_train_cat), torch.tensor(X_train_num), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(deep_model.parameters(), lr=1e-3, weight_decay=1e-4)

    deep_model.train()
    for epoch in range(12):
        for b_cat, b_num, b_y in train_loader:
            b_cat, b_num, b_y = b_cat.to(device), b_num.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = deep_model(b_cat, b_num)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

    deep_model.eval()
    with torch.no_grad():
        val_logits = deep_model(torch.tensor(X_val_cat).to(device), torch.tensor(X_val_num).to(device))
        val_deep_raw = torch.sigmoid(val_logits).cpu().numpy()

        test_logits = deep_model(torch.tensor(X_test_cat).to(device), torch.tensor(X_test_num).to(device))
        test_deep_raw = torch.sigmoid(test_logits).cpu().numpy()

    # 5. Probability Calibration on Temporal Validation Split (Mandatory)
    print("\n[3/4] Calibrating Probabilities using Isotonic Regression on Validation Set...")
    from sklearn.isotonic import IsotonicRegression

    calibrator_fast = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator_fast.fit(val_fast_raw, y_val)
    test_fast_cal = calibrator_fast.predict(test_fast_raw)

    calibrator_deep = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator_deep.fit(val_deep_raw, y_val)
    test_deep_cal = calibrator_deep.predict(test_deep_raw)

    # 6. Comprehensive Metrics Evaluation
    print("\n[4/4] Evaluating Models on Temporal Test Set:")
    fast_prauc = average_precision_score(y_test, test_fast_cal)
    fast_rocauc = roc_auc_score(y_test, test_fast_cal)
    fast_rec1, _ = recall_at_fixed_fpr(y_test, test_fast_cal, target_fpr=0.01)
    fast_brier = brier_score_loss(y_test, test_fast_cal)

    deep_prauc = average_precision_score(y_test, test_deep_cal)
    deep_rocauc = roc_auc_score(y_test, test_deep_cal)
    deep_rec1, _ = recall_at_fixed_fpr(y_test, test_deep_cal, target_fpr=0.01)
    deep_brier = brier_score_loss(y_test, test_deep_cal)

    print("-" * 80)
    print(f"| Model | PR-AUC | ROC-AUC | Recall@1%FPR | Brier Score (Calibration) |")
    print(f"| Fast Model (LightGBM) | {fast_prauc:.4f} | {fast_rocauc:.4f} | {fast_rec1*100:5.2f}% | {fast_brier:.5f} |")
    print(f"| Deep Model (Neural Net) | {deep_prauc:.4f} | {deep_rocauc:.4f} | {deep_rec1*100:5.2f}% | {deep_brier:.5f} |")
    print("-" * 80)

    margin = ((deep_prauc - fast_prauc) / fast_prauc) * 100.0
    print(f"Deep Model Margin: +{margin:.2f}% relative PR-AUC improvement over Fast Model.")
    assert deep_prauc > fast_prauc, "Deep model must outperform fast model to justify deep path!"

    # 7. Save Artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump({"encoder": encoder, "scaler": scaler, "num_cols": NUMERICAL_COLS, "cat_cols": CATEGORICAL_COLS}, f)

    with open(os.path.join(ARTIFACTS_DIR, "fast_model.pkl"), "wb") as f:
        pickle.dump(fast_model, f)

    with open(os.path.join(ARTIFACTS_DIR, "calibrators.pkl"), "wb") as f:
        pickle.dump({"fast": calibrator_fast, "deep": calibrator_deep}, f)

    torch.save(deep_model.state_dict(), os.path.join(ARTIFACTS_DIR, "deep_model.pth"))
    print(f"\nArtifacts successfully saved to {ARTIFACTS_DIR}.")

    return {
        "fast_prauc": fast_prauc,
        "deep_prauc": deep_prauc,
        "fast_rec1": fast_rec1,
        "deep_rec1": deep_rec1
    }

if __name__ == "__main__":
    train_and_evaluate()
