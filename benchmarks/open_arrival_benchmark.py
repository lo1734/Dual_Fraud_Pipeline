import os
import sys
import time
import math
import random
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from hdrh.histogram import HdrHistogram

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)
sys.path.insert(0, os.path.join(WORKSPACE_DIR, "ml"))

from ml.models import TabularDeepModel
from ml.router import RouterConfig, LoadState, route

CHANNELS = ["UPI", "IMPS", "NETBANKING", "CARD"]
MERCHANTS = ["RETAIL", "GROCERY", "GAMING", "CRYPTO", "ELECTRONICS"]
GEOS = ["IN-KA", "IN-MH", "IN-DL", "IN-TN", "IN-WB"]

def generate_request(idx: int):
    # Skewed entity distribution (Zipf / Pareto-like distribution)
    payer_id = f"P{int(np.random.pareto(1.5) * 50) % 2000}"
    payee_id = f"M{int(np.random.pareto(1.5) * 100) % 5000}"
    # 5% high value transactions to stress the high-value bypass logic
    amt = float(random.uniform(20000, 50000)) if random.random() < 0.05 else float(random.uniform(50, 2000))
    return {
        "txnId": f"BENCH-{idx}-{int(time.time()*1000)}",
        "tsEvent": int(time.time() * 1000),
        "payerId": payer_id,
        "payeeId": payee_id,
        "deviceId": f"DEV-{random.randint(1, 1000)}",
        "amount": round(amt, 2),
        "currency": "INR",
        "channel": random.choice(CHANNELS),
        "merchantCategory": random.choice(MERCHANTS),
        "ipGeo": random.choice(GEOS)
    }

class OpenArrivalBenchmarker:
    """
    Open-arrival rate generator using Poisson arrival processes.
    Measures true uncoordinated queueing latency without coordinated omission.
    """
    def __init__(self, target_url: str = "http://127.0.0.1:8000/score"):
        self.target_url = target_url
        # HdrHistogram: values from 1 to 100,000 microseconds (0.001 ms to 100 ms), 3 sig figs
        self.hist_total = HdrHistogram(1, 200000, 3)
        self.hist_fast = HdrHistogram(1, 200000, 3)
        self.hist_deep = HdrHistogram(1, 200000, 3)
        self.hist_fallback = HdrHistogram(1, 200000, 3)
        self.routes_count = {"fast": 0, "deep": 0, "fast_fallback": 0, "deep_audit": 0, "error": 0}

    def reset_histograms(self):
        self.hist_total = HdrHistogram(1, 200000, 3)
        self.hist_fast = HdrHistogram(1, 200000, 3)
        self.hist_deep = HdrHistogram(1, 200000, 3)
        self.hist_fallback = HdrHistogram(1, 200000, 3)
        self.routes_count = {"fast": 0, "deep": 0, "fast_fallback": 0, "deep_audit": 0, "error": 0}

    async def _send_request(self, session: aiohttp.ClientSession, payload: dict, scheduled_start: float):
        actual_send = time.perf_counter()
        try:
            timeout = aiohttp.ClientTimeout(total=0.060) # 60ms timeout to observe true latency
            async with session.post(self.target_url, json=payload, timeout=timeout) as resp:
                status = resp.status
                data = await resp.json() if status == 200 else {}
                finish = time.perf_counter()
                latency_us = int(max(1, (finish - actual_send) * 1_000_000))
                self.hist_total.record_value(min(200000, latency_us))
                
                route_type = data.get("route", "error")
                self.routes_count[route_type] = self.routes_count.get(route_type, 0) + 1
                if route_type == "fast":
                    self.hist_fast.record_value(min(200000, latency_us))
                elif route_type in ("deep", "deep_audit"):
                    self.hist_deep.record_value(min(200000, latency_us))
                elif route_type == "fast_fallback":
                    self.hist_fallback.record_value(min(200000, latency_us))
        except Exception:
            finish = time.perf_counter()
            latency_us = int(max(1, (finish - actual_send) * 1_000_000))
            self.hist_total.record_value(min(200000, latency_us))
            self.routes_count["error"] += 1

    async def run_scenario(self, scenario_name: str, stages: list[tuple[float, float]]):
        """
        stages is a list of (duration_seconds, target_tps).
        Generates arrivals following a Poisson distribution.
        """
        self.reset_histograms()
        print(f"\n================================================================================")
        print(f"RUNNING SCENARIO: {scenario_name}")
        print(f"Stages: {stages}")
        print(f"================================================================================")

        connector = aiohttp.TCPConnector(limit=4000, ttl_dns_cache=300)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            req_idx = 0
            start_wall = time.perf_counter()
            current_time = start_wall

            for duration, target_tps in stages:
                stage_end = current_time + duration
                if target_tps <= 0:
                    await asyncio.sleep(duration)
                    current_time = time.perf_counter()
                    continue

                mean_interval = 1.0 / target_tps
                while current_time < stage_end:
                    req_idx += 1
                    payload = generate_request(req_idx)
                    now = time.perf_counter()
                    if current_time > now:
                        await asyncio.sleep(current_time - now)
                    # Task spawned at arrival time
                    t = asyncio.create_task(self._send_request(session, payload, current_time))
                    tasks.append(t)

                    # Poisson inter-arrival delta: -ln(U) * mean_interval
                    delta = -math.log(max(1e-9, random.random())) * mean_interval
                    current_time = max(time.perf_counter(), current_time + delta)

            print(f"Dispatched {len(tasks)} requests. Awaiting completions...")
            await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.perf_counter() - start_wall
        return self._format_results(scenario_name, len(tasks), elapsed)

    def _format_results(self, scenario_name: str, total_dispatched: int, duration_sec: float) -> dict:
        total_rec = self.hist_total.get_total_count()
        if total_rec == 0:
            return {"scenario": scenario_name, "error": "No requests recorded"}

        p50 = self.hist_total.get_value_at_percentile(50.0) / 1000.0
        p90 = self.hist_total.get_value_at_percentile(90.0) / 1000.0
        p95 = self.hist_total.get_value_at_percentile(95.0) / 1000.0
        p99 = self.hist_total.get_value_at_percentile(99.0) / 1000.0
        p99_9 = self.hist_total.get_value_at_percentile(99.9) / 1000.0
        achieved_tps = total_rec / duration_sec

        deep_routes = self.routes_count.get("deep", 0) + self.routes_count.get("deep_audit", 0)
        fallbacks = self.routes_count.get("fast_fallback", 0)
        fast_routes = self.routes_count.get("fast", 0)

        deep_pct = (deep_routes / total_rec) * 100.0 if total_rec else 0.0
        fb_pct = (fallbacks / total_rec) * 100.0 if total_rec else 0.0

        res = {
            "scenario": scenario_name,
            "dispatched": total_dispatched,
            "recorded": total_rec,
            "duration_sec": round(duration_sec, 2),
            "achieved_tps": round(achieved_tps, 1),
            "p50_ms": round(p50, 2),
            "p90_ms": round(p90, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "p99_9_ms": round(p99_9, 2),
            "fast_count": fast_routes,
            "deep_count": deep_routes,
            "deep_share_pct": round(deep_pct, 2),
            "fallback_count": fallbacks,
            "fallback_pct": round(fb_pct, 3),
            "error_count": self.routes_count.get("error", 0),
            "slo_pass": (p99_9 <= 50.0)
        }

        print(f"\n--- {scenario_name} SUMMARY RESULTS ---")
        print(f"Achieved TPS: {res['achieved_tps']} | Duration: {res['duration_sec']}s | Samples: {res['recorded']}")
        print(f"Latency Percentiles: p50={res['p50_ms']}ms | p90={res['p90_ms']}ms | p95={res['p95_ms']}ms | p99={res['p99_ms']}ms | p99.9={res['p99_9_ms']}ms")
        print(f"Routing Distribution: Fast={fast_routes} | Deep={deep_routes} ({res['deep_share_pct']}%) | Fallbacks={fallbacks} ({res['fallback_pct']}%)")
        print(f"SLO Target Check: p99.9 <= 50.0 ms -> {'✅ PASSED' if res['slo_pass'] else '❌ FAILED'}")
        return res
