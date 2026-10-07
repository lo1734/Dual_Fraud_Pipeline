package com.fraud.kafka;

import com.fraud.model.Transaction;
import org.apache.kafka.clients.Metadata;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.serialization.StringSerializer;

import java.util.Properties;

public class TransactionProducer {
    private static final String BOOTSTRAP_SERVERS =
            "localhost:9092";

    private static final String TOPIC =
            "txn.raw";

    public static void main(String[] args){

        Properties properties = new Properties();
        properties.put(
                ProducerConfig.BOOTSTRAP_SERVERS_CONFIG,
                BOOTSTRAP_SERVERS
        );
        properties.put(
                ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG,
                StringSerializer.class.getName()
        );
        properties.put(
                ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG,
                TransactionSerializer.class.getName()
        );
        try(KafkaProducer<String, Transaction> producer=
            new KafkaProducer<>(properties)
        ){
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

            ProducerRecord<String, Transaction> record =
                    new ProducerRecord<>(
                            TOPIC,
                            txn.getPayerId(),
                            txn
                    );
            producer.send(
                    record,
                    (metadata,exception)->{
                        if(exception!=null){
                            exception.printStackTrace();
                        }else{
                            System.out.println("Transaction sent successfully!");

                            System.out.println("Topic: " + metadata.topic());

                            System.out.println("Partition: " + metadata.partition());

                            System.out.println("Offset: " + metadata.offset());
                        }
                    }


            );
            producer.flush();
        }
    }

}
