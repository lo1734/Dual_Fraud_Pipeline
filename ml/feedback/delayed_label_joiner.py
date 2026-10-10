import os
import sys
import time
import json
import sqlite3
import pandas as pd
from typing import Dict, List, Optional
from kafka import KafkaConsumer, KafkaProducer

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
FEEDBACK_PARQUET = os.path.join(DATA_DIR, "retraining_feedback.parquet")
DB_PATH = os.path.join(DATA_DIR, "feedback_state.db")

class DelayedLabelJoiner:
    """
    Ingests delayed chargebacks/labels from txn.labels and joins them temporally
    with historical scoring decisions from txn.decisions to measure realized performance
    and produce ground-truth datasets for continuous model retraining.
    """
    def __init__(self, db_path: str = DB_PATH, bootstrap_servers: str = "localhost:9092"):
        self.db_path = db_path
        self.bootstrap_servers = bootstrap_servers
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    txn_id TEXT PRIMARY KEY,
                    payer_id TEXT,
                    score REAL,
                    action TEXT,
                    route TEXT,
                    latency_ms REAL,
                    load_factor REAL,
                    decision_ts INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS labeled_outcomes (
                    txn_id TEXT PRIMARY KEY,
                    payer_id TEXT,
                    score REAL,
                    action TEXT,
                    route TEXT,
                    latency_ms REAL,
                    decision_ts INTEGER,
                    label_ts INTEGER,
                    is_fraud INTEGER,
                    chargeback_code TEXT,
                    joined_ts INTEGER
                )
            """)
            conn.commit()

    def record_decision(self, decision: dict):
        """Records a real-time scoring decision into the joiner state buffer."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO decisions 
                (txn_id, payer_id, score, action, route, latency_ms, load_factor, decision_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision["txnId"],
                decision["payerId"],
                float(decision["score"]),
                decision["action"],
                decision["route"],
                float(decision.get("latencyMs", 0.0)),
                float(decision.get("loadFactor", 0.0)),
                int(decision["timestamp"])
            ))
            conn.commit()

    def record_label(self, label_payload: dict) -> Optional[dict]:
        """
        Receives a delayed fraud label (e.g., chargeback after 14-60 days)
        and joins with the prior decision.
        """
        txn_id = label_payload["txnId"]
        is_fraud = int(label_payload["isFraud"])
        chargeback_code = label_payload.get("chargebackCode", "NONE")
        label_ts = int(label_payload.get("timestamp", time.time() * 1000))
        joined_ts = int(time.time() * 1000)

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT txn_id, payer_id, score, action, route, latency_ms, decision_ts
                FROM decisions WHERE txn_id = ?
            """, (txn_id,))
            row = cur.fetchone()

            if not row:
                return None  # Decision not found in buffer

            joined_record = {
                "txn_id": row[0],
                "payer_id": row[1],
                "score": row[2],
                "action": row[3],
                "route": row[4],
                "latency_ms": row[5],
                "decision_ts": row[6],
                "label_ts": label_ts,
                "is_fraud": is_fraud,
                "chargeback_code": chargeback_code,
                "joined_ts": joined_ts
            }

            conn.execute("""
                INSERT OR REPLACE INTO labeled_outcomes
                (txn_id, payer_id, score, action, route, latency_ms, decision_ts, label_ts, is_fraud, chargeback_code, joined_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                joined_record["txn_id"],
                joined_record["payer_id"],
                joined_record["score"],
                joined_record["action"],
                joined_record["route"],
                joined_record["latency_ms"],
                joined_record["decision_ts"],
                joined_record["label_ts"],
                joined_record["is_fraud"],
                joined_record["chargeback_code"],
                joined_record["joined_ts"]
            ))
            conn.commit()

        self._export_to_parquet()
        return joined_record

    def compute_operational_metrics(self) -> dict:
        """
        Computes realized production metrics:
        - False Discovery Rate (Declines that were not fraud)
        - Missed Fraud Rate (Approved transactions that resulted in chargebacks)
        - Route breakdown accuracy
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM labeled_outcomes", conn)

        if df.empty:
            return {"total_labeled": 0}

        total = len(df)
        fraud_cases = df[df["is_fraud"] == 1]
        legit_cases = df[df["is_fraud"] == 0]

        # Approved transactions that were fraud -> False Negatives (Missed Fraud)
        approved_fraud = df[(df["action"] == "APPROVE") & (df["is_fraud"] == 1)]
        missed_fraud_rate = len(approved_fraud) / max(1, len(fraud_cases))

        # Declined transactions that were legit -> False Positives (False Discovery)
        declined_legit = df[(df["action"] == "DECLINE") & (df["is_fraud"] == 0)]
        declines = df[df["action"] == "DECLINE"]
        false_discovery_rate = len(declined_legit) / max(1, len(declines))

        # Route breakdown
        route_perf = {}
        for r, group in df.groupby("route"):
            route_perf[r] = {
                "count": len(group),
                "fraud_count": int(group["is_fraud"].sum()),
                "avg_score": round(float(group["score"].mean()), 4)
            }

        return {
            "total_labeled": total,
            "total_fraud": len(fraud_cases),
            "missed_fraud_count": len(approved_fraud),
            "missed_fraud_rate_pct": round(missed_fraud_rate * 100.0, 2),
            "false_discovery_count": len(declined_legit),
            "false_discovery_rate_pct": round(false_discovery_rate * 100.0, 2),
            "route_breakdown": route_perf
        }

    def _export_to_parquet(self):
        """Flushes joined ground-truth dataset to Parquet for model retraining."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM labeled_outcomes", conn)
        if not df.empty:
            df.to_parquet(FEEDBACK_PARQUET, index=False)
