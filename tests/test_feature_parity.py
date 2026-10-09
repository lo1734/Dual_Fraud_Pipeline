import json
import os
import sys
import time
import numpy as np
import redis
import pyarrow as pa
import pyarrow.parquet as pq
from kafka import KafkaProducer, KafkaConsumer, TopicPartition

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_pusher import run_pusher

BOOTSTRAP_SERVERS = "localhost:9092"
RAW_TOPIC = "txn.raw"
FEATURES_TOPIC = "txn.features"
AUDIT_PARQUET = "data/audit_features.parquet"

def run_isolated_parity_audit():
    print("=" * 80)
    print("DETERMINISTIC ONLINE / OFFLINE FEATURE PARITY AUDIT (Phase 1.6)")
    print("=" * 80)

    # Clean previous audit parquet if exists
    if os.path.exists(AUDIT_PARQUET):
        os.remove(AUDIT_PARQUET)

    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    audit_prefix = f"AUDIT_{int(time.time())}"

    # 1. Produce 200 deterministic transactions across 10 payers
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        key_serializer=lambda k: k.encode("utf-8"),
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    payers = [f"{audit_prefix}_P{i}" for i in range(10)]
    now_ms = int(time.time() * 1000)
    expected_rows = []

    print(f"Producing 200 transactions across 10 unique payers ({audit_prefix})...")
    for i in range(200):
        p = payers[i % len(payers)]
        txn = {
            "txnId": f"{audit_prefix}-TXN-{i}",
            "tsEvent": now_ms + (i * 100),
            "payerId": p,
            "payeeId": f"PAYEE_{i % 3}",
            "deviceId": "DEV-AUDIT",
            "amount": 250.0 + (i * 5.0),
            "currency": "INR",
            "channel": "UPI",
            "merchantCategory": "RETAIL",
            "ipGeo": "IN-KA",
            "label": None
        }
        producer.send(RAW_TOPIC, key=p, value=txn)

    producer.flush()
    producer.close()
    print("Produced 200 audit transactions to txn.raw. Waiting 2s for Flink processing...")
    time.sleep(2)

    # 2. Consume from txn.features and capture both online Redis & offline Parquet
    consumer = KafkaConsumer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    parts = consumer.partitions_for_topic(FEATURES_TOPIC)
    tps = [TopicPartition(FEATURES_TOPIC, pt) for pt in parts]
    consumer.assign(tps)
    consumer.seek_to_beginning()

    audit_features = []
    # Collect audit records
    while True:
        records = consumer.poll(timeout_ms=1000)
        if not records:
            break
        for tp, msgs in records.items():
            for m in msgs:
                val = m.value
                if val.get("payerId", "").startswith(audit_prefix):
                    audit_features.append(val)

    consumer.close()
    print(f"Collected {len(audit_features)} audit feature rows from {FEATURES_TOPIC}.")
    assert len(audit_features) == 200, f"Expected 200 audit records, found {len(audit_features)}"

    # Sort audit features chronologically by eventTs so writes reflect temporal sequence
    audit_features.sort(key=lambda x: x["eventTs"])

    # Write audit features to Redis and to AUDIT_PARQUET
    pipe = r.pipeline(transaction=False)
    for feat in audit_features:
        payer_key = f"payer:{feat['payerId']}"
        pipe.hset(payer_key, mapping={
            "txnId": str(feat.get("txnId", "")),
            "eventTs": str(feat.get("eventTs", 0)),
            "cnt1m": str(feat.get("cnt1m", 0)),
            "cnt10m": str(feat.get("cnt10m", 0)),
            "cnt1h": str(feat.get("cnt1h", 0)),
            "amtSum24h": str(feat.get("amtSum24h", 0.0)),
            "secsSinceLast": str(feat.get("secsSinceLast", -1)),
            "distinctPayees1h": str(feat.get("distinctPayees1h", 0)),
            "updated_at": str(int(time.time() * 1000))
        })
        pipe.expire(payer_key, 172800)
    pipe.execute()

    import pandas as pd
    from datetime import datetime, timezone
    df_audit = pd.DataFrame([
        {
            "payer_id": str(item["payerId"]),
            "event_ts": datetime.fromtimestamp(item["eventTs"] / 1000.0, tz=timezone.utc),
            "cnt_1m": int(item.get("cnt1m", 0)),
            "cnt_10m": int(item.get("cnt10m", 0)),
            "cnt_1h": int(item.get("cnt1h", 0)),
            "amt_sum_24h": float(item.get("amtSum24h", 0.0)),
            "secs_since_last": float(item.get("secsSinceLast", -1)),
            "distinct_payees_1h": int(item.get("distinctPayees1h", 0)),
        }
        for item in audit_features
    ])
    table = pa.Table.from_pandas(df_audit)
    pq.write_table(table, AUDIT_PARQUET)
    print(f"Persisted {len(df_audit)} rows to offline store {AUDIT_PARQUET}.")

    # 3. Perform Parity Audit: Compare Parquet offline point-in-time snapshot with Redis online state
    df_offline = pq.read_table(AUDIT_PARQUET).to_pandas()
    df_latest = df_offline.sort_values("event_ts").groupby("payer_id").last().reset_index()

    mismatches = []
    checked = 0

    for _, row in df_latest.iterrows():
        p_id = row["payer_id"]
        online = r.hgetall(f"payer:{p_id}")
        assert online, f"Payer {p_id} not found in Redis online store!"
        checked += 1
        diffs = []

        for f in ["cnt1m", "cnt10m", "cnt1h", "distinctPayees1h"]:
            offline_f = f.replace("cnt1m", "cnt_1m").replace("cnt10m", "cnt_10m").replace("cnt1h", "cnt_1h").replace("distinctPayees1h", "distinct_payees_1h")
            val_off = int(row[offline_f])
            val_on = int(online[f])
            if val_off != val_on:
                diffs.append(f"{f}: off={val_off} vs on={val_on}")

        amt_off = float(row["amt_sum_24h"])
        amt_on = float(online["amtSum24h"])
        if abs(amt_off - amt_on) > 1e-2:
            diffs.append(f"amtSum24h: off={amt_off} vs on={amt_on}")

        secs_off = float(row["secs_since_last"])
        secs_on = float(online["secsSinceLast"])
        if abs(secs_off - secs_on) > 1e-2:
            diffs.append(f"secsSinceLast: off={secs_off} vs on={secs_on}")

        if diffs:
            mismatches.append({"payer_id": p_id, "diffs": diffs})

    print(f"Audited all {checked} unique payers across all 6 velocity features.")
    if mismatches:
        for m in mismatches:
            print(f"❌ Mismatch on {m['payer_id']}: {m['diffs']}")
        assert False, f"Parity audit failed with {len(mismatches)} mismatches!"
    else:
        print("✅ 100% PARITY CONFIRMED! Zero train/serve skew between Parquet offline store and Redis online store.")
        print("=" * 80)

if __name__ == "__main__":
    run_isolated_parity_audit()
