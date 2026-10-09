package com.fraud.flink;

import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.base.DeliveryGuarantee;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fraud.model.Transaction;

import java.io.File;
import java.time.Duration;

public class FraudFeatureJob {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    public static void main(String[] args) throws Exception {

        System.out.println("Fraud Feature Job starting...");
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 12 partitions matches Kafka topic partitions
        env.setParallelism(1);

        String defaultBootstrapServers = new File("/opt/flink").exists() ? "kafka:29092" : "localhost:9092";
        String bootstrapServers = args.length > 0 ? args[0] :
                System.getenv().getOrDefault("KAFKA_BOOTSTRAP_SERVERS", defaultBootstrapServers);

        System.out.println("Connecting to Kafka at: " + bootstrapServers);

        KafkaSource<String> source = KafkaSource.<String>builder()
                .setBootstrapServers(bootstrapServers)
                .setTopics("txn.raw")
                .setGroupId("fraud-feature-group")
                .setStartingOffsets(OffsetsInitializer.earliest())
                .setValueOnlyDeserializer(new SimpleStringSchema())
                .build();

        var transactionJsonStream = env.fromSource(
                source,
                WatermarkStrategy.noWatermarks(),
                "Kafka Transaction Raw Source"
        );

        // Deserialization and Event-Time Watermarking with 5-second bounded out-of-orderness
        var transactions = transactionJsonStream
                .map(json -> OBJECT_MAPPER.readValue(json, Transaction.class))
                .assignTimestampsAndWatermarks(
                        WatermarkStrategy.<Transaction>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                                .withTimestampAssigner((txn, recordTimestamp) -> txn.getTsEvent())
                );

        var keyedTransactions = transactions.keyBy(Transaction::getPayerId);

        var featureRows = keyedTransactions.process(new VelocityFeatureFunction());

        // Sink to Kafka txn.features topic
        KafkaSink<String> kafkaSink = KafkaSink.<String>builder()
                .setBootstrapServers(bootstrapServers)
                .setRecordSerializer(
                        KafkaRecordSerializationSchema.builder()
                                .setTopic("txn.features")
                                .setValueSerializationSchema(new SimpleStringSchema())
                                .build()
                )
                .setDeliveryGuarantee(DeliveryGuarantee.AT_LEAST_ONCE)
                .build();

        var featureJsonStream = featureRows.map(OBJECT_MAPPER::writeValueAsString);
        featureJsonStream.sinkTo(kafkaSink).name("Kafka txn.features Sink");

        // Also print for observability in container logs
        featureRows.print();

        env.execute("Fraud Feature Job");
    }
}
