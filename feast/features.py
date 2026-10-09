from datetime import timedelta
from feast import Entity, FeatureView, Field, PushSource, FileSource, ValueType
from feast.types import Int64, Float32

# Entity definition with explicit ValueType
payer = Entity(
    name="payer_id",
    join_keys=["payer_id"],
    value_type=ValueType.STRING,
    description="Unique identifier for the payment initiator",
)

# Batch offline file source
batch_source = FileSource(
    path="data/features.parquet",
    timestamp_field="event_ts",
)

# Push source for real-time streaming velocity features
push_source = PushSource(
    name="velocity_push",
    batch_source=batch_source,
)

# Feature view definition
velocity = FeatureView(
    name="velocity",
    entities=[payer],
    ttl=timedelta(days=2),
    schema=[
        Field(name="cnt_1m", dtype=Int64),
        Field(name="cnt_10m", dtype=Int64),
        Field(name="cnt_1h", dtype=Int64),
        Field(name="amt_sum_24h", dtype=Float32),
        Field(name="secs_since_last", dtype=Float32),
        Field(name="distinct_payees_1h", dtype=Int64),
    ],
    online=True,
    source=push_source,
)
