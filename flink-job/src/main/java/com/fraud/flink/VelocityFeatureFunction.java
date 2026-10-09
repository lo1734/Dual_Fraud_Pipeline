package com.fraud.flink;

import com.fraud.model.FeatureRow;
import com.fraud.model.Transaction;

import org.apache.flink.api.common.state.MapState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Computes per-payer velocity features using minute buckets.
 *
 * Window rule: a bucket is inside an N-minute window when
 * 0 <= currentMinute - bucketMinute < N.
 *
 * Event-time timers ensure state cleanup even if a payer goes inactive.
 */
public class VelocityFeatureFunction
        extends KeyedProcessFunction<String, Transaction, FeatureRow> {

    private static final long MINUTE_MS = 60_000L;
    private static final long HOUR_MS = 60 * MINUTE_MS;

    // How long each kind of state must be kept (= its longest window).
    private static final long COUNT_RETENTION_MINUTES = 60;       // cnt1m, cnt10m, cnt1h
    private static final long AMOUNT_RETENTION_MINUTES = 24 * 60; // amtSum24h

    private transient ValueState<Long> lastTimestampState;        // newest event ts seen for this payer
    private transient MapState<Long, Long> minuteBuckets;         // minute -> txn count
    private transient MapState<Long, Double> amountMinuteBuckets; // minute -> amount sum
    private transient MapState<String, Long> payeeLastSeen;       // payeeId -> last event ts

    @Override
    public void open(Configuration parameters) throws Exception {
        lastTimestampState = getRuntimeContext().getState(
                new ValueStateDescriptor<>("last-transaction-timestamp", Long.class));

        minuteBuckets = getRuntimeContext().getMapState(
                new MapStateDescriptor<>("transaction-minute-buckets", Long.class, Long.class));

        amountMinuteBuckets = getRuntimeContext().getMapState(
                new MapStateDescriptor<>("transaction-amount-minute-buckets", Long.class, Double.class));

        payeeLastSeen = getRuntimeContext().getMapState(
                new MapStateDescriptor<>("payee-last-seen", String.class, Long.class));
    }

    @Override
    public void processElement(
            Transaction transaction,
            Context context,
            Collector<FeatureRow> out) throws Exception {

        long currentTimestamp = transaction.getTsEvent();
        Long previousTimestamp = lastTimestampState.value();

        // Convention: secsSinceLast is -1 if this is the first transaction for this payer.
        long secsSinceLast = -1;
        if (previousTimestamp != null) {
            secsSinceLast = Math.max(0, (currentTimestamp - previousTimestamp) / 1000);
        }

        // Newest event seen so far; pruning is relative to this so a late event
        // cannot cause still-needed buckets to be deleted.
        long maxTimestamp = (previousTimestamp == null)
                ? currentTimestamp
                : Math.max(previousTimestamp, currentTimestamp);
        long maxMinute = maxTimestamp / MINUTE_MS;
        long currentMinute = currentTimestamp / MINUTE_MS;

        // Register event-time timer for state cleanup (fires when watermarks pass retention window)
        context.timerService().registerEventTimeTimer((maxMinute + AMOUNT_RETENTION_MINUTES + 1) * MINUTE_MS);

        // 1. Record this transaction in state.
        Long bucketCount = minuteBuckets.get(currentMinute);
        minuteBuckets.put(currentMinute, (bucketCount == null ? 0L : bucketCount) + 1);

        Double bucketAmount = amountMinuteBuckets.get(currentMinute);
        amountMinuteBuckets.put(currentMinute,
                (bucketAmount == null ? 0.0 : bucketAmount) + transaction.getAmount());

        String payeeId = transaction.getPayeeId();
        if (payeeId != null && !payeeId.trim().isEmpty()) {
            Long seen = payeeLastSeen.get(payeeId);
            if (seen == null || currentTimestamp > seen) {
                payeeLastSeen.put(payeeId, currentTimestamp);
            }
        }

        // 2. Transaction counts (single pass) + prune expired count buckets.
        long cnt1m = 0, cnt10m = 0, cnt1h = 0;
        List<Long> expiredCountBuckets = new ArrayList<>();

        for (Map.Entry<Long, Long> entry : minuteBuckets.entries()) {
            long bucketMinute = entry.getKey();
            if (bucketMinute <= maxMinute - COUNT_RETENTION_MINUTES) {
                expiredCountBuckets.add(bucketMinute);
                continue;
            }
            long age = currentMinute - bucketMinute;
            if (age < 0) continue; // bucket is in the future relative to an out-of-order event
            if (age < 1)  cnt1m  += entry.getValue();
            if (age < 10) cnt10m += entry.getValue();
            if (age < 60) cnt1h  += entry.getValue();
        }
        for (Long minute : expiredCountBuckets) {
            minuteBuckets.remove(minute);
        }

        // 3. 24h amount sum + prune expired amount buckets.
        double amtSum24h = 0.0;
        List<Long> expiredAmountBuckets = new ArrayList<>();

        for (Map.Entry<Long, Double> entry : amountMinuteBuckets.entries()) {
            long bucketMinute = entry.getKey();
            if (bucketMinute <= maxMinute - AMOUNT_RETENTION_MINUTES) {
                expiredAmountBuckets.add(bucketMinute);
                continue;
            }
            long age = currentMinute - bucketMinute;
            if (age >= 0 && age < AMOUNT_RETENTION_MINUTES) {
                amtSum24h += entry.getValue();
            }
        }
        for (Long minute : expiredAmountBuckets) {
            amountMinuteBuckets.remove(minute);
        }

        // 4. Distinct payees in the last hour + prune payees not seen for > 1h.
        long distinctPayees1h = 0;
        List<String> expiredPayees = new ArrayList<>();

        for (Map.Entry<String, Long> entry : payeeLastSeen.entries()) {
            long lastSeen = entry.getValue();
            if (lastSeen <= maxTimestamp - HOUR_MS) {
                expiredPayees.add(entry.getKey());
                continue;
            }
            if (lastSeen > currentTimestamp - HOUR_MS && lastSeen <= currentTimestamp) {
                distinctPayees1h++;
            }
        }
        for (String payee : expiredPayees) {
            payeeLastSeen.remove(payee);
        }

        // 5. Emit the feature row.
        out.collect(new FeatureRow(
                transaction.getTxnId(),
                currentTimestamp,
                transaction.getPayerId(),
                cnt1m,
                cnt10m,
                cnt1h,
                amtSum24h,
                secsSinceLast,
                distinctPayees1h
        ));

        // Keep the newest timestamp so older out-of-order events don't move state backwards.
        lastTimestampState.update(maxTimestamp);
    }

    @Override
    public void onTimer(long timestamp, OnTimerContext context, Collector<FeatureRow> out) throws Exception {
        Long maxTs = lastTimestampState.value();
        if (maxTs == null) return;
        long maxMinute = maxTs / MINUTE_MS;

        List<Long> expiredCountBuckets = new ArrayList<>();
        for (Map.Entry<Long, Long> entry : minuteBuckets.entries()) {
            if (entry.getKey() <= maxMinute - COUNT_RETENTION_MINUTES) {
                expiredCountBuckets.add(entry.getKey());
            }
        }
        for (Long m : expiredCountBuckets) {
            minuteBuckets.remove(m);
        }

        List<Long> expiredAmountBuckets = new ArrayList<>();
        for (Map.Entry<Long, Double> entry : amountMinuteBuckets.entries()) {
            if (entry.getKey() <= maxMinute - AMOUNT_RETENTION_MINUTES) {
                expiredAmountBuckets.add(entry.getKey());
            }
        }
        for (Long m : expiredAmountBuckets) {
            amountMinuteBuckets.remove(m);
        }

        List<String> expiredPayees = new ArrayList<>();
        for (Map.Entry<String, Long> entry : payeeLastSeen.entries()) {
            if (entry.getValue() <= maxTs - HOUR_MS) {
                expiredPayees.add(entry.getKey());
            }
        }
        for (String p : expiredPayees) {
            payeeLastSeen.remove(p);
        }
    }
}