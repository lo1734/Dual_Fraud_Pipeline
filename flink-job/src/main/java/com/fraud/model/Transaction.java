package com.fraud.model;

public class Transaction {

    private String txnId;
    private long tsEvent;

    private String payerId;
    private String payeeId;
    private String deviceId;

    private double amount;
    private String currency;
    private String channel;
    private String merchantCategory;
    private String ipGeo;

    // Offline only
    private Integer label;

    public Transaction() {
    }

    public Transaction(
            String txnId,
            long tsEvent,
            String payerId,
            String payeeId,
            String deviceId,
            double amount,
            String currency,
            String channel,
            String merchantCategory,
            String ipGeo,
            Integer label) {

        this.txnId = txnId;
        this.tsEvent = tsEvent;
        this.payerId = payerId;
        this.payeeId = payeeId;
        this.deviceId = deviceId;
        this.amount = amount;
        this.currency = currency;
        this.channel = channel;
        this.merchantCategory = merchantCategory;
        this.ipGeo = ipGeo;
        this.label = label;
    }

    public String getTxnId() {
        return txnId;
    }

    public long getTsEvent() {
        return tsEvent;
    }

    public String getPayerId() {
        return payerId;
    }

    public String getPayeeId() {
        return payeeId;
    }

    public String getDeviceId() {
        return deviceId;
    }

    public double getAmount() {
        return amount;
    }

    public String getCurrency() {
        return currency;
    }

    public String getChannel() {
        return channel;
    }

    public String getMerchantCategory() {
        return merchantCategory;
    }

    public String getIpGeo() {
        return ipGeo;
    }

    public Integer getLabel() {
        return label;
    }

    public void setTxnId(String txnId) {
        this.txnId = txnId;
    }

    public void setTsEvent(long tsEvent) {
        this.tsEvent = tsEvent;
    }

    public void setPayerId(String payerId) {
        this.payerId = payerId;
    }

    public void setPayeeId(String payeeId) {
        this.payeeId = payeeId;
    }

    public void setDeviceId(String deviceId) {
        this.deviceId = deviceId;
    }

    public void setAmount(double amount) {
        this.amount = amount;
    }

    public void setCurrency(String currency) {
        this.currency = currency;
    }

    public void setChannel(String channel) {
        this.channel = channel;
    }

    public void setMerchantCategory(String merchantCategory) {
        this.merchantCategory = merchantCategory;
    }

    public void setIpGeo(String ipGeo) {
        this.ipGeo = ipGeo;
    }

    public void setLabel(Integer label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return "Transaction{" +
                "txnId='" + txnId + '\'' +
                ", tsEvent=" + tsEvent +
                ", payerId='" + payerId + '\'' +
                ", payeeId='" + payeeId + '\'' +
                ", deviceId='" + deviceId + '\'' +
                ", amount=" + amount +
                ", currency='" + currency + '\'' +
                ", channel='" + channel + '\'' +
                ", merchantCategory='" + merchantCategory + '\'' +
                ", ipGeo='" + ipGeo + '\'' +
                ", label=" + label +
                '}';
    }
}
