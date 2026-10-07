package com.fraud;

import com.fraud.model.Transaction;

public class TransactionTest {

    public static void main(String[] args) {

        Transaction txn = new Transaction(
                "TXN-001",
                System.currentTimeMillis(),
                "P001",
                "P900",
                "DEVICE-001",
                2500.00,
                "INR",
                "UPI",
                "GROCERY",
                "IN-AP",
                null
        );

        System.out.println(txn);
    }
}
