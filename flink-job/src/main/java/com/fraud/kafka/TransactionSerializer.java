package com.fraud.kafka;

import com.fraud.model.Transaction;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.kafka.common.serialization.Serializer;

public class TransactionSerializer implements Serializer<Transaction> {
    private final ObjectMapper objectMapper = new ObjectMapper();

    public byte[] serialize(String topic, Transaction transaction){
        if(transaction == null) return null;

        try{
            return objectMapper.writeValueAsBytes(transaction);
        }catch (Exception e){
            throw new RuntimeException(
                    "Failed to serialize transaction.",
                    e
            );
        }
    }

}
