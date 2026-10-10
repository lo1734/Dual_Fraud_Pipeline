import os
import sys
import time
import asyncio
import numpy as np
from httpx import AsyncClient, ASGITransport

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WORKSPACE_DIR)
sys.path.insert(0, os.path.join(WORKSPACE_DIR, "scoring_service"))

from scoring_service.app import app, startup_event, shutdown_event, state

def test_chaos_scenarios():
    print("=" * 80)
    print("STEP 5: CHAOS & FAILURE INJECTION SUITE")
    print("=" * 80)

    async def _run_async_tests():
        await startup_event()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:

            # -------------------------------------------------------------
            # Scenario 1: Deep Path / Triton Hard Failure Injection
            # -------------------------------------------------------------
            print("\n[Chaos 1/3] Injecting Deep Path Hard Failure / Triton Outage (150ms delay)...")
            req = {
                "txnId": "CHAOS-DEEP-FAIL-01",
                "tsEvent": int(time.time() * 1000),
                "payerId": "P_CHAOS_01",
                "payeeId": "M1",
                "amount": 25000.0, # High value triggers deep path
                "channel": "UPI",
                "merchantCategory": "ELECTRONICS",
                "ipGeo": "IN-KA"
            }

            t0 = time.perf_counter()
            # test_delay_ms=150ms exceeds 35ms deadline
            res = await client.post("/score?test_delay_ms=150.0", json=req)
            elapsed = (time.perf_counter() - t0) * 1000.0

            assert res.status_code == 200
            data = res.json()
            print(f"Fallback response received in {elapsed:.2f} ms: route={data['route']}, score={data['score']}")
            assert data["route"] == "fast_fallback", f"Expected fast_fallback, got {data['route']}"
            assert elapsed < 50.0, f"SLA breach during deep outage! Took {elapsed} ms"
            print("✅ Scenario 1 PASS: Deep path outage automatically caught by fallback engine within 50ms SLA!")

            # -------------------------------------------------------------
            # Scenario 2: Redis Connection Failure / Cold Entity Cache Miss
            # -------------------------------------------------------------
            print("\n[Chaos 2/3] Simulating Redis Connection Interruption / Cold Entity...")
            # Temporarily replace redis instance with a broken client
            orig_redis = state["redis"]
            class BrokenRedis:
                async def hmget(self, *args, **kwargs):
                    raise ConnectionResetError("Simulated Redis Connection Drop")

            state["redis"] = BrokenRedis()
            t0 = time.perf_counter()
            res_redis = await client.post("/score", json=req)
            elapsed_redis = (time.perf_counter() - t0) * 1000.0
            state["redis"] = orig_redis # restore

            assert res_redis.status_code == 200
            data_redis = res_redis.json()
            print(f"Resilient response under Redis outage in {elapsed_redis:.2f} ms: score={data_redis['score']}")
            assert elapsed_redis < 20.0
            print("✅ Scenario 2 PASS: Scoring service cleanly handles Redis disconnection without crashing!")

            # -------------------------------------------------------------
            # Scenario 3: Extreme Burst Surge & Emergency Load Shedding
            # -------------------------------------------------------------
            print("\n[Chaos 3/3] Simulating Extreme Traffic Burst (LF >= 0.95)...")
            # Force high load state
            state["load_state"].tps = 10000.0
            state["load_state"].inflight = 64

            # Low/medium value uncertain transaction
            req_burst = {
                "txnId": "CHAOS-BURST-01",
                "tsEvent": int(time.time() * 1000),
                "payerId": "P_BURST_01",
                "payeeId": "M2",
                "amount": 200.0, # Normal value
                "channel": "UPI",
                "merchantCategory": "RETAIL",
                "ipGeo": "IN-KA"
            }

            res_burst = await client.post("/score", json=req_burst)
            data_burst = res_burst.json()
            print(f"Emergency load response: route={data_burst['route']}, loadFactor={data_burst['loadFactor']}")
            assert data_burst["route"] == "fast", "Under extreme burst, non-critical traffic must shed to fast path"
            print("✅ Scenario 3 PASS: Emergency load factor shedding protects system from GPU saturation!")

        await shutdown_event()

    asyncio.run(_run_async_tests())
    print("\n" + "=" * 80)
    print("🎉 ALL CHAOS & FAILURE INJECTION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_chaos_scenarios()
