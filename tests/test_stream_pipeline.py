import json
import time
from kafka import KafkaProducer, KafkaConsumer

BOOTSTRAP_SERVERS = "localhost:9092"
RAW_TOPIC = "txn.raw"
FEATURES_TOPIC = "txn.features"

def run_test():
    print("--- Step 1.4: End-to-End Streaming Verification ---")
    
    # 1. Setup consumer on txn.features
    consumer = KafkaConsumer(
        FEATURES_TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id=f"test-verifier-{int(time.time() * 1000)}",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=10000
    )
    # Warm up consumer connection
    consumer.poll(timeout_ms=1000)

    # 2. Setup producer for txn.raw
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        key_serializer=lambda k: k.encode("utf-8"),
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    test_payer = f"TEST_PAYER_{int(time.time())}"
    now_ms = int(time.time() * 1000)
    txns = [
        {
            "txnId": f"TEST-TXN-{i}",
            "tsEvent": now_ms + (i * 1000),
            "payerId": test_payer,
            "payeeId": f"PAYEE_{i % 2}",
            "deviceId": "DEVICE_TEST",
            "amount": 100.0 * (i + 1),
            "currency": "INR",
            "channel": "UPI",
            "merchantCategory": "RETAIL",
            "ipGeo": "IN-KA",
            "label": None
        }
        for i in range(5)
    ]

    print(f"Publishing {len(txns)} transactions for payer {test_payer} to {RAW_TOPIC}...")
    for txn in txns:
        producer.send(RAW_TOPIC, key=txn["payerId"], value=txn)
    producer.flush()
    print("Published successfully. Listening for output on txn.features...")

    received_features = []
    start_wait = time.time()
    while time.time() - start_wait < 15:
        records = consumer.poll(timeout_ms=1000)
        for tp, messages in records.items():
            for msg in messages:
                feat = msg.value
                if feat.get("payerId") == test_payer:
                    received_features.append(feat)
                    print(f"Received FeatureRow: txnId={feat.get('txnId')} | payer={feat.get('payerId')} | cnt1m={feat.get('cnt1m')} | amt24h={feat.get('amtSum24h')} | secsSinceLast={feat.get('secsSinceLast')}")
        if len(received_features) >= len(txns):
            break

    consumer.close()
    producer.close()

    assert len(received_features) > 0, "No features received on txn.features!"
    print(f"SUCCESS: Verified {len(received_features)} feature rows emitted through Flink to {FEATURES_TOPIC}!")

if __name__ == "__main__":
    run_test()
