# Adaptive Dual-Path Payment Fraud Detection Architecture
## End-to-End Implementation Checklist & Roadmap

**Repository:** `dual-path-fraud`  
**Current Status Snapshot:** October 10, 2026  
**Latest Confirmed Checkpoint:** Phase 1 & Phase 2 Fully Completed & Verified (1.1 through 2.4).  
**Primary Performance Targets:**
- End-to-end scoring latency: **$p99.9 \le 50\text{ ms}$** under dynamic burst workloads.
- Model quality retention: **Within 2–3%** (relative PR-AUC / Recall@1% FPR) of the always-deep baseline.
- Streaming feature freshness: **$p99 < 200\text{ ms}$** (event-time to Redis visibility).
- Online feature store fetch: **$p99.9 < 5\text{ ms}$** at $2\times$ target TPS.

---

### Latency Budget Hypothesis (50 ms Scoring Path Budget)
| Stage | Budget | Implementation & Notes |
| :--- | :--- | :--- |
| **Gateway + Router Overhead** | $\le 2\text{ ms}$ | Stateless request parsing, load factor calculation |
| **Online Feature Fetch (Redis)** | $3\text{–}5\text{ ms}$ | Pre-aggregated velocity features (pipeline/MGET, measured $p99.9 = 0.186\text{ ms}$) |
| **Fast Model (In-Process)** | $1\text{–}3\text{ ms}$ | LightGBM/XGBoost via ONNX/Treelite/lleaves |
| **Deep Model (Triton GPU)** | $10\text{–}25\text{ ms}$ | Dynamic batching (preferred: 8, 16, 32; delay $\le 2\text{ ms}$) |
| **Network, Serialization & Headroom**| $10\text{ ms}$ | gRPC / buffer pooling / safety margin |
| **Hard Deadline for Deep Path** | **$\sim 35\text{ ms}$** | `asyncio.wait_for(timeout=0.035)` $\to$ fallback to fast score |

---

### Legend
- `[x]` **Completed & Verified**
- `[/]` **In Progress / Partially Implemented**
- `[ ]` **Not Started**

---

## Phase 1: Streaming Infrastructure & Feature Pipeline Foundation

### 1.1 Local Infrastructure & Base Plumbing
- [x] **Repository Structure & Directories**
  - Directories created: `infrastructure/`, `flink-job/`, `simulator/`, `feast/`, `tests/`, `data/`.
- [x] **Docker Compose Streaming Setup** (`infrastructure/docker-compose.yml`)
  - Apache Kafka 3.8.0 container (`fraud-kafka`, port 9092/29092, KRaft mode).
  - Redis 7 container (`fraud-redis`, port 6379, `--appendonly no --save ""`).
  - Apache Flink 1.19.0 JobManager (`fraud-flink-jobmanager`, port 8081).
  - Apache Flink 1.19.0 TaskManager (`fraud-flink-taskmanager`, 4 slots).
- [x] **Kafka Topics Initialization**
  - Topic `txn.raw`: 12 partitions, key: `payer_id`.
  - Topic `txn.features`: 12 partitions.
- [x] **Maven & Flink Project Setup** (`flink-job/pom.xml`)
  - Java 11 build target, Flink 1.19.0 dependencies, Jackson Databind, Flink Kafka connector, `flink-connector-base`, `flink-test-utils`.
- [x] **Domain Models**
  - `Transaction.java`: `txnId`, `tsEvent`, `payerId`, `payeeId`, `deviceId`, `amount`, `currency`, `channel`, `merchantCategory`, `ipGeo`, `label`.
  - `FeatureRow.java`: Feature schema container for computed velocity features.
- [x] **Producer & Synthetic Simulator**
  - `TransactionProducer.java` & `TransactionSerializer.java`.
  - `TransactionSimulator.java`: Synthetic transactions generator targeting `txn.raw`.
- [x] **Flink Streaming Ingestion & Deployment** (`FraudFeatureJob.java`)
  - Ingest from `txn.raw` via KafkaSource, deserialization, keying by `payerId`.

---

### 1.2 Stateful Velocity Feature Engineering (`VelocityFeatureFunction.java`)
- [x] **Base Feature State Implementation**
  - `secsSinceLast`: Per-payer `ValueState<Long>` tracking previous transaction timestamp with `-1` first-txn convention.
  - `cnt1m`: MapState minute-bucket approximation for 1-minute transaction count.
  - `cnt10m`: MapState minute-bucket approximation for 10-minute transaction count.
  - `cnt1h`: MapState minute-bucket approximation for 1-hour transaction count.
  - `amtSum24h`: MapState minute-bucket tracking amount sums across 1440 minutes.
- [x] **Step 7.16: Payee-State Representation Correction**
  - [x] Replaced composite key with `payeeLastSeen` MapState (`payeeId -> last event timestamp`).
  - [x] Null-safe and whitespace-safe payee handling.
  - [x] Exact 1-hour rolling window boundary ($currentTs - lastSeen \le 3600\text{ s}$).
- [x] **Event-Time Watermarks & Timer-Based State Cleanup**
  - [x] Watermark generation strategy: `BoundedOutOfOrderness(Duration.ofSeconds(5))` with timestamp assigner on `txn.getTsEvent()`.
  - [x] Registered event-time cleanup timer: `context.timerService().registerEventTimeTimer((maxMinute + 1441) * 60_000)`.
  - [x] Implemented `onTimer(...)` callback to cleanly drop expired count, amount, and payee buckets when payers go inactive.
  - [x] Monotonic `lastTimestampState` update preventing out-of-order state regression.

---

### 1.3 Feature Correctness & Unit Testing (`VelocityFeatureFunctionTest.java`)
- [x] **Deterministic Unit Tests for `VelocityFeatureFunction`**
  - [x] Test 1: Single payer first transaction (`secsSinceLast = -1`, counts = 1, amount = amount, distinct payees = 1).
  - [x] Test 2: Rapid successive transactions within same minute (incrementing counts, accurate amount sum, positive `secsSinceLast`).
  - [x] Test 3: Transactions across multiple minute boundaries (verifying 1m expires while 10m and 1h retain).
  - [x] Test 4: Distinct payee deduplication (repeated transactions to payee A count as 1; payment to payee B increments to 2).
  - [x] Test 5: Hourly boundary and payee expiration (65 min later; payee expires from 1h window).
  - [x] Test 6: 25-hour gap (verify 24h sum and 1h counts reset to current txn values).
  - [x] Test 7: Out-of-order timestamps (verifying monotonicity and non-negative `secsSinceLast`).
- [x] JVM `--add-opens` configured in `maven-surefire-plugin` for Java 17+ Kryo compatibility.
- [x] All 6 unit tests passing (`mvn test` $\to$ **BUILD SUCCESS**).

---

### 1.4 Complete Flink Pipeline & Kafka Output
- [x] **Kafka Feature Sink Integration**
  - [x] Attached `KafkaSink<String>` using JSON serialization to `FraudFeatureJob.java`.
  - [x] Configured output stream to topic `txn.features`.
  - [x] Built shaded JAR (`mvn clean package -DskipTests`).
  - [x] Deployed updated job (`be4a7325222f1d08ca3d85e0de69a6f3`) to JobManager.
- [x] **End-to-End Streaming Verification**
  - [x] Verified records flow from `txn.raw` through Flink into `txn.features` via `tests/test_stream_pipeline.py`.

---

### 1.5 Online Feature Store & Feast Integration
- [x] **Feast Feature Store Configuration** (`feast/feature_store.yaml`)
  - [x] Project configuration: Redis online store (`localhost:6379`), File offline store (`data/features.parquet`).
  - [x] Defined Entity (`payer_id`) and Feature View (`velocity`) in `feast/features.py`.
  - [x] Defined `PushSource(name="velocity_push")`.
  - [x] Deployed schema with `feast apply`.
- [x] **Feature Pusher Micro-Batch Service** (`feature_pusher.py`)
  - [x] Consumer of `txn.features` (micro-batches of 100–200 rows).
  - [x] Pipelined `HSET` to direct Redis hashes (`payer:<payerId>`) with 48h TTL.
  - [x] Push to Feast: `store.push("velocity_push", df_batch, to=PushMode.ONLINE_AND_OFFLINE)`.
- [x] **Online Retrieval Benchmark (Decision Gate)**
  - [x] Executed `tests/benchmark_online_retrieval.py` (5,000 requests per strategy):
    - Direct Redis HMGET: $p50 = 0.036\text{ ms}$, $p99.9 = 0.172\text{ ms}$.
    - Direct Redis HGETALL: $p50 = 0.034\text{ ms}$, $p99.9 = 0.063\text{ ms}$.
    - Feast Python SDK: $p50 = 0.123\text{ ms}$, $p99.9 = 1.233\text{ ms}$.
  - [x] Confirmed Direct Redis hash lookup satisfies the $p99.9 < 5.0\text{ ms}$ SLO with sub-0.2ms latency.

---

### 1.6 Offline Feature Persistence & Parity Audit
- [x] **Offline Feature Sink**
  - [x] Persisted feature stream to Parquet files in `data/features.parquet`.
  - [x] Validated "as of previous event for entity" semantics.
- [x] **Online/Offline Feature Parity Audit**
  - [x] Built and executed deterministic replay test `tests/test_feature_parity.py`.
  - [x] Audited all 6 velocity features against Redis online hashes.
  - [x] **100% PARITY CONFIRMED** (zero train/serve feature skew).

---

### 1.7 Phase 1 Scale & Performance Benchmarking (Exit Criteria)
- [x] **Replay & Scale Test** (`tests/benchmark_phase1_scale.py`)
  - [x] Replayed 10,000 burst events at **$9,145\text{ TPS}$**.
  - [x] Monitored Flink backpressure metric: **`status: ok` (zero backpressure)**.
- [x] **Feature Freshness Benchmark**
  - [x] Measured end-to-end streaming freshness: **$p50 = 101.0\text{ ms}$, $p99 = 119.5\text{ ms}$** ($\le 200\text{ ms}$ SLO).
- [x] **Online Feature Fetch Benchmark**
  - [x] Benchmarked Redis fetch under $2\times$ target load (5,000 requests): **$p99.9 = 0.186\text{ ms}$** ($\le 5\text{ ms}$ SLO).
- [x] **Phase 1 Exit Criteria Sign-off**
  - [x] All 4 exit criteria met and documented in `walkthrough.md`. Ready for Phase 2!

---

## Phase 2: Dual Models & Scoring Service Router

### 2.1 Dataset Preparation & Evaluation Strategy
- [x] **Benchmark datasets:** Synthetic UPI simulator with strict entity proxies and velocity features.
- [x] **Strict temporal train/val/test splitting:** Generated 50,000 transactions partitioned temporally into `train.parquet`, `val.parquet`, and `test.parquet`.
- [x] **Label isolation:** Labels isolated strictly from online feature schema.
- [x] **Feature matrix contract:** Assembled training feature matrix preserving the "as of previous event" contract with zero leakage.

### 2.2 Model Training & Calibration
- [x] **Fast Model Path:** LightGBM shallow decision trees trained and evaluated ($PR\text{-}AUC = 0.1344$, in-process latency $p99.9 = 1.054\text{ ms} < 3\text{ ms}$).
- [x] **Deep Model Path:** PyTorch Tabular DNN with entity embeddings trained ($PR\text{-}AUC = 0.1751$, $+30.29\%$ relative performance margin over fast model).
- [x] **Probability Calibration:** Isotonic regression fitted on temporal validation split with well-calibrated probabilities.
- [x] **Artifacts Saved:** Models and preprocessors serialized to `ml/artifacts/fast_model.pkl`, `deep_model.pth`, `calibrators.pkl`, and `preprocessor.pkl`.

### 2.3 Router Logic & Fallback Engine
- [x] **Adaptive Router Implementation (`ml/router.py`):**
  - Uncertainty band $[\tau_{lo}, \tau_{hi}]$ with high-value bypass ($amt > \$2,000$) and EWMA load factor shedding.
  - Vectorized Pareto sweep evaluated across threshold grid.
  - Optimal operating point $[\tau_{lo}=0.02, \tau_{hi}=0.70]$ routes only **$37.8\%$** to deep path while achieving **$104.35\%$** of deep baseline PR-AUC (surpassing the 2–3% quality retention SLO).
- [x] **Hard Deadline & Fast-Score Fallback Engine:**
  - Fast model always runs first to guarantee an immediate score.
  - Asynchronous $35\text{ ms}$ deadline on deep path via `asyncio.wait_for`.
  - Fallback engine catches simulated delays ($100\text{ ms}$) and returns fast score in $40.71\text{ ms} \le 50\text{ ms}$.

### 2.4 Scoring Service Implementation
- [x] **FastAPI Real-Time Scoring Microservice (`scoring_service/app.py`):**
  - Startup model warmup preventing cold-start spikes.
  - Async direct Redis `HMGET` online feature lookup ($p99.9 \le 0.186\text{ ms}$).
  - In-process fast inference + non-blocking background/concurrent deep path.
  - Live metric tracking (load factor, in-flight, TPS, deep share, fallbacks).
- [x] **Decision Telemetry Kafka Stream:**
  - Scoring decisions and audit telemetry published to Kafka topic `txn.decisions`.
- [x] **Comprehensive Verification Suite (`tests/test_phase2_suite.py`):**
  - All 4 tests verified and passing. Phase 2 fully validated!

---

## Phase 3: Load Testing, Benchmarking & SLO Validation

### 3.1 Open-Model Load Test Harness (k6 / wrk2)
- [ ] Open arrival rate test scenarios: steady 1k TPS, step ramp, burst spike (1k $\to$ 10k), key skew, failure injection.

### 3.2 Metrics Collection & HdrHistogram
- [ ] HdrHistogram recording $\ge 1\text{M}$ samples per stage ($p50, p95, p99, p99.9$).
- [ ] Prometheus + Grafana dashboards for latency percentiles and fallback rate.

### 3.3 Target Evaluation Matrix
| Configuration | TPS Sustained | $p99$ Latency | $p99.9$ Latency | Deep Share % | PR-AUC | Recall@1%FPR | Fallback % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Always-Deep Baseline** | Measured | Measured | Measured | 100% | Baseline | Baseline | 0% |
| **Always-Fast Baseline** | Measured | Measured | Measured | 0% | Measured | Measured | 0% |
| **Adaptive Dual-Path** | Target Peak | $\le 25\text{ ms}$ | **$\le 50\text{ ms}$** | $10\text{–}20\%$ | $\ge 0.97 \times \text{Deep}$ | $\ge 0.97 \times \text{Deep}$ | $< 0.1\%$ |

---

## Phase 4: Operational Hardening (Next)
- [ ] Delayed-label feedback pipeline (`txn.labels` chargebacks joined to `txn.decisions`).
- [ ] Feature and model drift monitoring.
- [ ] Canary and shadow-mode rollout strategy.
