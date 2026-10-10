import os
import sys
import time
import asyncio
import aiohttp
import numpy as np
import onnxruntime as ort

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "ml", "artifacts")

class TritonClient:
    """
    Asynchronous client for Triton Inference Server:
    - Queries queue depth from Triton :8002/metrics
    - Sends inference requests to Triton HTTP/gRPC endpoints
    - Seamlessly falls back to local in-process ONNX Runtime if Triton server is offline
    """
    def __init__(self,
                 triton_http_url: str = "http://localhost:8003",
                 triton_metrics_url: str = "http://localhost:8002/metrics",
                 model_name: str = "deep_fraud",
                 model_version: str = "1"):
        self.triton_http_url = triton_http_url
        self.triton_metrics_url = triton_metrics_url
        self.model_name = model_name
        self.model_version = model_version
        self.infer_url = f"{self.triton_http_url}/v2/models/{self.model_name}/versions/{self.model_version}/infer"

        # Local fallback in-process ONNX Runtime session
        onnx_path = os.path.join(ARTIFACTS_DIR, "deep_model.onnx")
        if os.path.exists(onnx_path):
            self.ort_session = ort.InferenceSession(onnx_path)
        else:
            self.ort_session = None

        self._triton_available = False
        self._last_health_check = 0.0
        self.queue_depth = 0

    async def check_health(self, session: aiohttp.ClientSession) -> bool:
        now = time.monotonic()
        if now - self._last_health_check < 2.0:
            return self._triton_available

        self._last_health_check = now
        try:
            health_url = f"{self.triton_http_url}/v2/health/ready"
            async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=0.5)) as resp:
                self._triton_available = (resp.status == 200)
        except Exception:
            self._triton_available = False

        return self._triton_available

    async def poll_metrics(self, session: aiohttp.ClientSession) -> int:
        """Polls Triton Prometheus metrics to calculate current queue depth."""
        try:
            async with session.get(self.triton_metrics_url, timeout=aiohttp.ClientTimeout(total=0.5)) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    # Parse queue metrics: nv_inference_pending_request_count or queue duration
                    for line in text.splitlines():
                        if "nv_inference_pending_request_count" in line and not line.startswith("#"):
                            val = float(line.split()[-1])
                            self.queue_depth = int(val)
                            return self.queue_depth
        except Exception:
            pass
        return self.queue_depth

    async def infer(self, session: aiohttp.ClientSession, X_cat: np.ndarray, X_num: np.ndarray) -> float:
        """
        Runs deep model inference:
        1. Attempts remote Triton Inference Server if ready.
        2. Falls back to in-process ONNX Runtime session if Triton is offline.
        """
        is_ready = await self.check_health(session)
        if is_ready:
            try:
                payload = {
                    "inputs": [
                        {
                            "name": "x_cat",
                            "shape": list(X_cat.shape),
                            "datatype": "INT64",
                            "data": X_cat.tolist()
                        },
                        {
                            "name": "x_num",
                            "shape": list(X_num.shape),
                            "datatype": "FP32",
                            "data": X_num.tolist()
                        }
                    ],
                    "outputs": [{"name": "logits"}]
                }
                async with session.post(self.infer_url, json=payload, timeout=aiohttp.ClientTimeout(total=0.035)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logits = data["outputs"][0]["data"][0]
                        raw_prob = 1.0 / (1.0 + np.exp(-logits))
                        return float(raw_prob)
            except Exception:
                pass  # Fall back to local ONNX runtime

        # Local in-process ONNX Runtime execution (fast fallback)
        if self.ort_session is not None:
            ort_inputs = {
                "x_cat": X_cat.astype(np.int64),
                "x_num": X_num.astype(np.float32)
            }
            logits = self.ort_session.run(None, ort_inputs)[0]
            raw_prob = 1.0 / (1.0 + np.exp(-logits[0]))
            return float(raw_prob)

        # Final safety dummy fallback
        return 0.5
