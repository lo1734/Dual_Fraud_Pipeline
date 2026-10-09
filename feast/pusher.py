import json
import os
import time
import pandas as pd
import redis
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone
from kafka import KafkaConsumer
from feast import FeatureStore
from feast.data_source import PushMode

BOOTSTRAP_SERVERS = "localhost:9092"
FEATURES_TOPIC = "txn.features"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
PARQUET_FILE = "data/features.parquet"

def get_feast_store():
    # Load feast feature store from feast directory
    feast_dir = os.path.dirname(os.path.abspath(__file__))
    return FeatureStore(repo_path=feast_dir)

def run_pusher(max_batches=None, batch_size=100, poll_timeout_ms=50):
    print("Starting Feature Pusher service...")
    store = get_feast_store()
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    consumer = KafkaConsumer(
        FEATURES_TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="feature-pusher-group",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=5000
    )

    batches_processed = 0
    total_records = 0

    try:
        buffer = []
        last_flush = time.time()

        while True:
            records = consumer.poll(timeout_ms=poll_timeout_ms)
            for tp, messages in records.items():
                for msg in messages:
                    buffer.append(msg.value)

            # Flush condition: buffer limit or time window
            now = time.time()
            if buffer and (len(buffer) >= batch_size or (now - last_flush) >= 0.05):
                batch_to_process = buffer[:batch_size]
                buffer = buffer[batch_size:]
                last_flush = now

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

                if max_batches and batches_processed >= max_batches:
                    break

            if not records and not buffer:
                # If consumer timed out and nothing in buffer
                if max_batches and batches_processed >= max_batches:
                    break
                # In standalone mode or when poll finishes
                if max_batches is not None and max_batches > 0:
                    break

    finally:
        consumer.close()
        print(f"Feature Pusher finished. Total processed: {total_records} records.")

if __name__ == "__main__":
    run_pusher()
