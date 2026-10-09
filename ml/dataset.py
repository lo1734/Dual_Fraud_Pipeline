import os
import time
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def generate_fraud_dataset(num_transactions=50000, num_payers=1000, num_payees=300):
    """
    Generates a realistic transaction stream with:
    - Temporal ordering
    - Velocity features calculated strictly "as of previous event"
    - Realistic fraud patterns (bursts, high amounts, high payee velocity)
    """
    print(f"Generating {num_transactions} synthetic payment transactions...")
    np.random.seed(42)

    # Time spans 7 days (in milliseconds)
    base_time = int(time.time() * 1000) - (7 * 24 * 3600 * 1000)
    time_deltas = np.sort(np.random.exponential(scale=12.0 * 1000, size=num_transactions))
    timestamps = base_time + np.cumsum(time_deltas).astype(np.int64)

    payer_ids = [f"PAYER_{np.random.randint(0, num_payers)}" for _ in range(num_transactions)]
    payee_ids = [f"PAYEE_{np.random.randint(0, num_payees)}" for _ in range(num_transactions)]
    channels = np.random.choice(["UPI", "MOBILE", "WEB"], size=num_transactions, p=[0.70, 0.20, 0.10])
    categories = np.random.choice(["GROCERY", "RESTAURANT", "ELECTRONICS", "TRAVEL", "PEER_TRANSFER"], size=num_transactions)
    geos = np.random.choice(["IN-KA", "IN-MH", "IN-DL", "IN-TN", "IN-TS", "INTERNATIONAL"], size=num_transactions, p=[0.35, 0.25, 0.15, 0.12, 0.10, 0.03])
    amounts = np.round(np.random.lognormal(mean=5.5, sigma=1.2, size=num_transactions), 2)

    # Track state per payer to compute exact "as of previous event" velocity features
    payer_history = {} # payer_id -> list of (ts, amount, payee_id)
    features_list = []
    labels = []

    fraud_payers = set(f"PAYER_{i}" for i in range(50)) # 5% fraudsters

    for i in range(num_transactions):
        ts = timestamps[i]
        p = payer_ids[i]
        payee = payee_ids[i]
        amt = amounts[i]
        geo = geos[i]
        cat = categories[i]

        is_fraud = False
        if p in fraud_payers:
            # Fraudster behaviors: velocity bursts, high amounts, electronics/travel, international
            if np.random.rand() < 0.40:
                is_fraud = True
                amt = amt * np.random.uniform(3.0, 10.0)
                if np.random.rand() < 0.6:
                    cat = "ELECTRONICS"

        # Calculate features AS OF PREVIOUS EVENT (strictly before current event is added)
        history = payer_history.get(p, [])
        if not history:
            cnt_1m = 0
            cnt_10m = 0
            cnt_1h = 0
            amt_sum_24h = 0.0
            secs_since_last = -1.0
            distinct_payees_1h = 0
        else:
            prev_ts = history[-1][0]
            secs_since_last = max(0.0, (ts - prev_ts) / 1000.0)

            # Filter historical events
            one_min_ago = ts - 60_000
            ten_min_ago = ts - 600_000
            one_hour_ago = ts - 3600_000
            day_ago = ts - 86400_000

            cnt_1m = sum(1 for h in history if h[0] >= one_min_ago)
            cnt_10m = sum(1 for h in history if h[0] >= ten_min_ago)
            cnt_1h = sum(1 for h in history if h[0] >= one_hour_ago)
            amt_sum_24h = sum(h[1] for h in history if h[0] >= day_ago)
            distinct_payees_1h = len(set(h[2] for h in history if h[0] >= one_hour_ago))

        # Velocity burst fraud rule: high rapid velocity
        if cnt_10m >= 5 and amt >= 5000:
            if np.random.rand() < 0.70:
                is_fraud = True

        features_list.append({
            "txn_id": f"TXN_{i:06d}",
            "event_ts": ts,
            "payer_id": p,
            "payee_id": payee,
            "channel": channels[i],
            "merchant_category": cat,
            "ip_geo": geo,
            "amount": amt,
            "cnt_1m": cnt_1m,
            "cnt_10m": cnt_10m,
            "cnt_1h": cnt_1h,
            "amt_sum_24h": float(amt_sum_24h),
            "secs_since_last": float(secs_since_last),
            "distinct_payees_1h": distinct_payees_1h,
            "is_fraud": 1 if is_fraud else 0
        })

        # Update payer history for subsequent events
        if p not in payer_history:
            payer_history[p] = []
        payer_history[p].append((ts, amt, payee))

    df = pd.DataFrame(features_list)
    print(f"Dataset generated: {len(df)} transactions | Fraud count: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")

    # Strict Temporal Split: Train 60%, Val 20%, Test 20%
    n = len(df)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val:].copy()

    print(f"Temporal Split: Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}")

    os.makedirs(DATA_DIR, exist_ok=True)
    df_train.to_parquet(os.path.join(DATA_DIR, "train.parquet"), index=False)
    df_val.to_parquet(os.path.join(DATA_DIR, "val.parquet"), index=False)
    df_test.to_parquet(os.path.join(DATA_DIR, "test.parquet"), index=False)
    print("Saved train.parquet, val.parquet, test.parquet to data directory.")
    return df_train, df_val, df_test

if __name__ == "__main__":
    generate_fraud_dataset()
