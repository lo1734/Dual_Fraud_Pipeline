package com.fraud.model;

public class FeatureRow {

    private String txnId;
    private long eventTs;
    private String payerId;

    private long cnt1m;
    private long cnt10m;
    private long cnt1h;

    private double amtSum24h;
    private long secsSinceLast;
    private long distinctPayees1h;

    public FeatureRow() {
    }

    public FeatureRow(
            String txnId,
            long eventTs,
            String payerId,
            long cnt1m,
            long cnt10m,
            long cnt1h,
            double amtSum24h,
            long secsSinceLast,
            long distinctPayees1h
    ) {
        this.txnId = txnId;
        this.eventTs = eventTs;
        this.payerId = payerId;
        this.cnt1m = cnt1m;
        this.cnt10m = cnt10m;
        this.cnt1h = cnt1h;
        this.amtSum24h = amtSum24h;
        this.secsSinceLast = secsSinceLast;
        this.distinctPayees1h = distinctPayees1h;
    }

    public String getTxnId() {
        return txnId;
    }

    public long getEventTs() {
        return eventTs;
    }

    public String getPayerId() {
        return payerId;
    }

    public long getCnt1m() {
        return cnt1m;
    }

    public long getCnt10m() {
        return cnt10m;
    }

    public long getCnt1h() {
        return cnt1h;
    }

    public double getAmtSum24h() {
        return amtSum24h;
    }

    public long getSecsSinceLast() {
        return secsSinceLast;
    }

    public long getDistinctPayees1h() {
        return distinctPayees1h;
    }

    @Override
    public String toString() {
        return "FeatureRow{" +
                "txnId='" + txnId + '\'' +
                ", eventTs=" + eventTs +
                ", payerId='" + payerId + '\'' +
                ", cnt1m=" + cnt1m +
                ", cnt10m=" + cnt10m +
                ", cnt1h=" + cnt1h +
                ", amtSum24h=" + amtSum24h +
                ", secsSinceLast=" + secsSinceLast +
                ", distinctPayees1h=" + distinctPayees1h +
                '}';
    }
}