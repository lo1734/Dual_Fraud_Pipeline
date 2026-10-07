package com.fraud.simulator;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;

import com.fraud.model.Transaction;
import com.fraud.kafka.TransactionSerializer;
import java.util.*;

public class TransactionSimulator {
    private static final String BOOTSTRAP_SERVERS = "localhost:9092";
    private static final String TOPIC = "txn.raw";
    private static final Random RANDOM = new Random();

    private static final String[] CHANNELS = {
            "UPI",
            "Mobile",
            "Web"
    };
    private static final String[] MERCHANT_CATEGORIES = {
            "GROCERY",
            "RESTAURANT",
            "FUEL",
            "SHOPPING",
            "TRAVEL"
    };

    private static final String[] IP_GEOS = {
            "IN-AP",
            "IN-TN",
            "IN-KA",
            "IN-MH",
            "IN-DL"
    };


    public static void main(String[] args){
        System.out.println("Starting Transaction Simulator...");

        Properties properties = new Properties();

        properties.put(
                ProducerConfig.BOOTSTRAP_SERVERS_CONFIG,
                BOOTSTRAP_SERVERS
        );

        properties.put(
                ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG,
                "org.apache.kafka.common.serialization.StringSerializer"
        );

        properties.put(
                ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG,
                TransactionSerializer.class.getName()
        );

        try (KafkaProducer<String, Transaction> producer =
                     new KafkaProducer<>(properties)) {

            for (int i = 0; i < 10; i++) {

                Transaction txn = generateTransaction();

                ProducerRecord<String, Transaction> record =
                        new ProducerRecord<>(
                                TOPIC,
                                txn.getPayerId(),
                                txn
                        );

                producer.send(
                        record,
                        (metadata, exception) -> {

                            if (exception != null) {

                                exception.printStackTrace();

                            } else {

                                System.out.println(
                                        "Sent transaction: "
                                                + txn.getTxnId()
                                                + " | Payer: "
                                                + txn.getPayerId()
                                                + " | Partition: "
                                                + metadata.partition()
                                                + " | Offset: "
                                                + metadata.offset()
                                );
                            }
                        }
                );
            }

            producer.flush();

            System.out.println(
                    "Finished sending 10 transactions."
            );
        }
    }

    private static Transaction generateTransaction(){
        String txnId = UUID.randomUUID().toString();
        long timestamp = System.currentTimeMillis();
        String payerId = "P" + (RANDOM.nextInt(100) + 1);
        String payeeId = "P"+(RANDOM.nextInt(1000)+1);
        String deviceId = "Device-"+(RANDOM.nextInt(50)+1);
        double amount =
                10 + (RANDOM.nextDouble() * 9990);

        String currency = "INR";

        String channel =
                CHANNELS[RANDOM.nextInt(CHANNELS.length)];

        String merchantCategory =
                MERCHANT_CATEGORIES[
                        RANDOM.nextInt(MERCHANT_CATEGORIES.length)
                        ];

        String ipGeo =
                IP_GEOS[RANDOM.nextInt(IP_GEOS.length)];

        Integer label = null;

        return new Transaction(
                txnId,
                timestamp,
                payerId,
                payeeId,
                deviceId,
                amount,
                currency,
                channel,
                merchantCategory,
                ipGeo,
                label
        );
    }


}
