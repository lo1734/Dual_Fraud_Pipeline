import http from 'k6/http';
import { check, sleep } from 'k6';

// Open arrival rate executor configuration
export const options = {
  scenarios: {
    // 1. Steady State: 1,000 TPS
    steady_1k: {
      executor: 'constant-arrival-rate',
      rate: 1000,
      timeUnit: '1s',
      duration: '10s',
      preAllocatedVUs: 100,
      maxVUs: 1000,
    },
    // 2. Burst Spikes: 1,000 TPS -> 5,000 TPS
    burst_spike: {
      executor: 'ramping-arrival-rate',
      startRate: 1000,
      timeUnit: '1s',
      stages: [
        { target: 1000, duration: '5s' },
        { target: 5000, duration: '5s' }, // dynamic burst
        { target: 1000, duration: '5s' }, // recovery
      ],
      preAllocatedVUs: 200,
      maxVUs: 2000,
      startTime: '12s',
    },
  },
  thresholds: {
    // End-to-end SLO: 99.9% of requests below 50ms
    'http_req_duration': ['p(99)<25', 'p(99.9)<50'],
  },
};

const CHANNELS = ['UPI', 'IMPS', 'NETBANKING', 'CARD'];
const MERCHANTS = ['RETAIL', 'GROCERY', 'GAMING', 'CRYPTO', 'ELECTRONICS'];
const GEOS = ['IN-KA', 'IN-MH', 'IN-DL', 'IN-TN', 'IN-WB'];

export default function () {
  const payerNum = Math.floor(Math.random() * 2000);
  const payeeNum = Math.floor(Math.random() * 5000);
  const amount = (Math.random() < 0.05) ? (20000 + Math.random() * 30000) : (50 + Math.random() * 2000);

  const payload = JSON.stringify({
    txnId: `TXN-${__VU}-${__ITER}-${Date.now()}`,
    tsEvent: Date.now(),
    payerId: `P${payerNum}`,
    payeeId: `M${payeeNum}`,
    deviceId: `DEV-${Math.floor(Math.random() * 1000)}`,
    amount: amount,
    currency: 'INR',
    channel: CHANNELS[Math.floor(Math.random() * CHANNELS.length)],
    merchantCategory: MERCHANTS[Math.floor(Math.random() * MERCHANTS.length)],
    ipGeo: GEOS[Math.floor(Math.random() * GEOS.length)],
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
    timeout: '50ms', // enforce client timeout
  };

  const res = http.post('http://127.0.0.1:8000/score', payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
  });
}
