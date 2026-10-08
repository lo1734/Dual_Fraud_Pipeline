package com.fraud.flink;

import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.api.common.serialization.SimpleStringSchema;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fraud.model.Transaction;

public class FraudFeatureJob {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    public static void main(String[] args) throws Exception {

        System.out.println(
                "Fraud Feature Job starting..."
        );
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        env.setParallelism(1);

        String defaultBootstrapServers = new java.io.File("/opt/flink").exists() ? "kafka:29092" : "localhost:9092";
        String bootstrapServers = args.length > 0 ? args[0] :
                System.getenv().getOrDefault("KAFKA_BOOTSTRAP_SERVERS", defaultBootstrapServers);

        System.out.println("Connecting to Kafka at: " + bootstrapServers);

        KafkaSource<String> source =
                KafkaSource.<String>builder()
                                .setBootstrapServers(bootstrapServers)
                                .setTopics("txn.raw")
                                .setGroupId("fraud-feature-group")
                                .setStartingOffsets(OffsetsInitializer.earliest())
                                .setValueOnlyDeserializer(new SimpleStringSchema())
                                .build();

        var transactionJsonStream =
                env.fromSource(
                        source,
                        WatermarkStrategy.noWatermarks(),
                        "Kafka Transaction Source"
                );
        var transactions =
                transactionJsonStream
                        .map(json -> OBJECT_MAPPER.readValue(
                                json,
                                Transaction.class
                        ));

//        var transactions =
//                env.fromSource(
//                        source,
//                        WatermarkStrategy.noWatermarks(),
//                        "Kafka Transaction Source"
//                );
        transactions.print();
        env.execute("Fraud Feature Job");
    }
}
