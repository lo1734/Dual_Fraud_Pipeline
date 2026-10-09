import json
import os
import sys
import time
import pandas as pd
import redis
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone
from kafka import KafkaConsumer, TopicPartition
from feast import FeatureStore
from feast.data_source import PushMode

BOOTSTRAP_SERVERS = "localhost:9092"
FEATURES_TOPIC = "txn.features"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
FEAST_REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feast")
PARQUET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "features.parquet")

def get_feast_store():
    return FeatureStore(repo_path=FEAST_REPO_PATH)

def run_pusher(max_records=None, batch_size=100, poll_timeout_ms=1000):
    print("Starting Feature Pusher service...")
    store = get_feast_store()
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    consumer = KafkaConsumer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    parts = consumer.partitions_for_topic(FEATURES_TOPIC)
    if not parts:
        print("No partitions found for topic!")
        return 0
    tps = [TopicPartition(FEATURES_TOPIC, p) for p in parts]
    consumer.assign(tps)
    consumer.seek_to_beginning()

    total_records = 0
    batches_processed = 0

    try:
        buffer = []
        while True:
            records = consumer.poll(timeout_ms=poll_timeout_ms)
            found = False
            for tp, messages in records.items():
                for msg in messages:
                    buffer.append(msg.value)
                    found = True

            while len(buffer) >= batch_size or (buffer and not found):
                batch_to_process = buffer[:batch_size] if len(buffer) >= batch_size else buffer[:]
                buffer = buffer[len(batch_to_process):]

                # 1. Direct Redis Hash Pipeline (Ultra-low latency hot path for scoring service)
                pipe = r.pipeline(transaction=False)
                for item in batch_to_process:
                    payer_key = f"payer:{item['payerId']}"
                    pipe.hset(payer_key, mapping={
                        "txnId": str(item.get("txnId", "")),
                        "eventTs": str(item.get("eventTs", 0)),
                        "cnt1m": str(item.get("cnt1m", 0)),
                        "cnt10m": str(item.get("cnt10m", 0)),
                        "cnt1h": str(item.get("cnt1h", 0)),
                        "amtSum24h": str(item.get("amtSum24h", 0.0)),
                        "secsSinceLast": str(item.get("secsSinceLast", -1)),
                        "distinctPayees1h": str(item.get("distinctPayees1h", 0)),
                        "updated_at": str(int(time.time() * 1000))
                    })
                    pipe.expire(payer_key, 172800) # 48h TTL
                pipe.execute()

                # 2. Push to Feast Online & Offline Store
                df_batch = pd.DataFrame([
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
                    for item in batch_to_process
                ])
                store.push("velocity_push", df_batch, to=PushMode.ONLINE_AND_OFFLINE)

                # 3. Direct Parquet append (Offline store persistence)
                table = pa.Table.from_pandas(df_batch)
                if os.path.exists(PARQUET_FILE):
                    try:
                        existing_table = pq.read_table(PARQUET_FILE)
                        combined = pa.concat_tables([existing_table, table])
                        pq.write_table(combined, PARQUET_FILE)
                    except Exception:
                        pq.write_table(table, PARQUET_FILE)
                else:
                    pq.write_table(table, PARQUET_FILE)

                total_records += len(batch_to_process)
                batches_processed += 1
                print(f"Pushed batch {batches_processed}: {len(batch_to_process)} records (Total: {total_records})")

                if max_records and total_records >= max_records:
                    break

            if max_records and total_records >= max_records:
                break
            if not found and not buffer:
                break

    finally:
        consumer.close()
        print(f"Feature Pusher finished. Total processed: {total_records} records.")
        return total_records

if __name__ == "__main__":
    run_pusher()
