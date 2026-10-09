package com.fraud.flink;

import com.fraud.model.FeatureRow;
import com.fraud.model.Transaction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.streaming.util.KeyedOneInputStreamOperatorTestHarness;
import org.apache.flink.streaming.util.ProcessFunctionTestHarnesses;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class VelocityFeatureFunctionTest {

    private KeyedOneInputStreamOperatorTestHarness<String, Transaction, FeatureRow> testHarness;

    @BeforeEach
    public void setup() throws Exception {
        VelocityFeatureFunction function = new VelocityFeatureFunction();
        testHarness = ProcessFunctionTestHarnesses.forKeyedProcessFunction(
                function,
                Transaction::getPayerId,
                BasicTypeInfo.STRING_TYPE_INFO
        );
        testHarness.open();
    }

    @AfterEach
    public void teardown() throws Exception {
        if (testHarness != null) {
            testHarness.close();
        }
    }

    private Transaction createTxn(String txnId, long ts, String payer, String payee, double amount) {
        return new Transaction(
                txnId,
                ts,
                payer,
                payee,
                "DEV-1",
                amount,
                "INR",
                "UPI",
                "RETAIL",
                "IN-KA",
                null
        );
    }

    @Test
    public void testFirstTransaction() throws Exception {
        long t0 = 1_000_000L;
        Transaction t1 = createTxn("TXN-1", t0, "U1", "M1", 500.0);

        testHarness.processElement(t1, t0);

        List<FeatureRow> output = testHarness.extractOutputValues();
        assertEquals(1, output.size());

        FeatureRow f1 = output.get(0);
        assertEquals("TXN-1", f1.getTxnId());
        assertEquals("U1", f1.getPayerId());
        assertEquals(1, f1.getCnt1m());
        assertEquals(1, f1.getCnt10m());
        assertEquals(1, f1.getCnt1h());
        assertEquals(500.0, f1.getAmtSum24h(), 1e-6);
        assertEquals(-1, f1.getSecsSinceLast(), "First transaction must have secsSinceLast = -1");
        assertEquals(1, f1.getDistinctPayees1h());
    }

    @Test
    public void testSuccessiveTransactionsSameMinute() throws Exception {
        long t0 = 1_000_000L; // Minute 16
        Transaction t1 = createTxn("TXN-1", t0, "U1", "M1", 500.0);
        Transaction t2 = createTxn("TXN-2", t0 + 15_000, "U1", "M1", 300.0); // 15s later, same minute

        testHarness.processElement(t1, t0);
        testHarness.processElement(t2, t0 + 15_000);

        List<FeatureRow> output = testHarness.extractOutputValues();
        assertEquals(2, output.size());

        FeatureRow f2 = output.get(1);
        assertEquals("TXN-2", f2.getTxnId());
        assertEquals(2, f2.getCnt1m());
        assertEquals(2, f2.getCnt10m());
        assertEquals(2, f2.getCnt1h());
        assertEquals(800.0, f2.getAmtSum24h(), 1e-6);
        assertEquals(15, f2.getSecsSinceLast());
        assertEquals(1, f2.getDistinctPayees1h(), "Same payee repeated should have distinct count = 1");
    }

    @Test
    public void testMinuteBoundaryAndDistinctPayees() throws Exception {
        long t0 = 60_000L * 100; // Minute 100
        Transaction t1 = createTxn("TXN-1", t0, "U1", "M1", 100.0);
        // t2 is 2 minutes later (Minute 102), to a new payee M2
        Transaction t2 = createTxn("TXN-2", t0 + 120_000L, "U1", "M2", 200.0);

        testHarness.processElement(t1, t0);
        testHarness.processElement(t2, t0 + 120_000L);

        List<FeatureRow> output = testHarness.extractOutputValues();
        assertEquals(2, output.size());

        FeatureRow f2 = output.get(1);
        assertEquals(1, f2.getCnt1m(), "t1 is 2 minutes ago, so cnt1m should only count t2");
        assertEquals(2, f2.getCnt10m(), "t1 is within 10 minutes");
        assertEquals(2, f2.getCnt1h(), "t1 is within 1 hour");
        assertEquals(300.0, f2.getAmtSum24h(), 1e-6);
        assertEquals(120, f2.getSecsSinceLast());
        assertEquals(2, f2.getDistinctPayees1h(), "Distinct payees M1 and M2 within 1 hour should be 2");
    }

    @Test
    public void testHourlyBoundaryAndPayeeExpiration() throws Exception {
        long t0 = 60_000L * 100; // Minute 100
        Transaction t1 = createTxn("TXN-1", t0, "U1", "M1", 100.0);
        // t2 is 65 minutes later (Minute 165), to payee M2
        Transaction t2 = createTxn("TXN-2", t0 + 65 * 60_000L, "U1", "M2", 250.0);

        testHarness.processElement(t1, t0);
        testHarness.processElement(t2, t0 + 65 * 60_000L);

        List<FeatureRow> output = testHarness.extractOutputValues();
        assertEquals(2, output.size());

        FeatureRow f2 = output.get(1);
        assertEquals(1, f2.getCnt1m());
        assertEquals(1, f2.getCnt10m());
        assertEquals(1, f2.getCnt1h(), "t1 was 65 minutes ago, so cnt1h should only count t2");
        assertEquals(350.0, f2.getAmtSum24h(), 1e-6, "24h window retains both transactions");
        assertEquals(65 * 60, f2.getSecsSinceLast());
        assertEquals(1, f2.getDistinctPayees1h(), "M1 was 65 min ago so only M2 is in 1h window");
    }

    @Test
    public void test24HourExpiration() throws Exception {
        long t0 = 60_000L * 100;
        Transaction t1 = createTxn("TXN-1", t0, "U1", "M1", 500.0);
        // t2 is 25 hours later
        Transaction t2 = createTxn("TXN-2", t0 + 25 * 3600_000L, "U1", "M2", 150.0);

        testHarness.processElement(t1, t0);
        testHarness.processElement(t2, t0 + 25 * 3600_000L);

        List<FeatureRow> output = testHarness.extractOutputValues();
        assertEquals(2, output.size());

        FeatureRow f2 = output.get(1);
        assertEquals(1, f2.getCnt1m());
        assertEquals(1, f2.getCnt10m());
        assertEquals(1, f2.getCnt1h());
        assertEquals(150.0, f2.getAmtSum24h(), 1e-6, "t1 was 25h ago, so 24h sum should be only t2 amount");
        assertEquals(25 * 3600, f2.getSecsSinceLast());
        assertEquals(1, f2.getDistinctPayees1h());
    }

    @Test
    public void testOutOfOrderEvents() throws Exception {
        long t0 = 60_000L * 100;
        Transaction t1 = createTxn("TXN-1", t0 + 10_000, "U1", "M1", 200.0); // 10s into minute
        Transaction tLate = createTxn("TXN-LATE", t0 + 2_000, "U1", "M2", 100.0); // 2s into minute (arrives late)

        testHarness.processElement(t1, t0 + 10_000);
        testHarness.processElement(tLate, t0 + 2_000);

        List<FeatureRow> output = testHarness.extractOutputValues();
        assertEquals(2, output.size());

        FeatureRow fLate = output.get(1);
        assertEquals("TXN-LATE", fLate.getTxnId());
        assertEquals(2, fLate.getCnt1m());
        assertEquals(300.0, fLate.getAmtSum24h(), 1e-6);
        // At event time of tLate (t0 + 2s), M1 (t0 + 10s) is in the future, so only M2 is counted
        assertEquals(1, fLate.getDistinctPayees1h());
        // For late event, currentTimestamp < previousTimestamp so secsSinceLast is 0
        assertEquals(0, fLate.getSecsSinceLast());
    }
}
