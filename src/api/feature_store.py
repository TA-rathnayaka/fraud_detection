"""
In-memory, per-card incremental feature store mirroring the causal features built in
notebooks/03_feature_engineering.ipynb -- but computed online, one transaction at a
time, instead of vectorized over a static table. This is the logic a real streaming
feature store (e.g. Redis-backed, updated by a Kafka consumer) runs per incoming event.

Velocity (per-card and per-merchant) / amount-anomaly / time-of-day-anomaly /
category-novelty / tenure features update immediately on every transaction -- they're
observable the instant a transaction happens. Merchant and category fraud-rate encodings are bootstrapped once from historical
labeled data at service startup and would be refreshed periodically by a batch job in
production (chargeback labels lag real transactions by days to weeks) -- they are NOT
updated per-request here, since an incoming transaction's true label isn't known at
scoring time.

Known scaling limitation: this is a single-process, lock-protected in-memory store, fine
for this project's demo throughput. Horizontally scaling the API would mean moving this
state to Redis (sorted sets per card for the windowed velocity queries).
"""

import math
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class FeatureStore:
    def __init__(self, merchant_fraud_rate: dict, category_fraud_rate: dict,
                 global_fraud_rate: float):
        self._lock = threading.Lock()
        self._history: dict[str, deque] = defaultdict(deque)  # user_id -> deque[(ts, amount)]
        self._count: dict[str, int] = defaultdict(int)
        self._sum: dict[str, float] = defaultdict(float)
        self._sumsq: dict[str, float] = defaultdict(float)
        self._categories: dict[str, set] = defaultdict(set)
        self._first_seen: dict[str, datetime] = {}
        self._last_ts: dict[str, datetime] = {}
        self._merchant_history: dict[str, deque] = defaultdict(deque)  # merchant_id -> deque[ts]
        self._sum_hour_sin: dict[str, float] = defaultdict(float)
        self._sum_hour_cos: dict[str, float] = defaultdict(float)

        self.merchant_fraud_rate = merchant_fraud_rate
        self.category_fraud_rate = category_fraud_rate
        self.global_fraud_rate = global_fraud_rate

    @property
    def n_users(self) -> int:
        return len(self._first_seen)

    def bootstrap_user(self, user_id: str, timestamp: datetime, amount: float) -> None:
        """Seed a card's history from historical data at startup."""
        self._ingest(user_id, timestamp, amount, category=None)

    def bootstrap_merchant(self, merchant_id: str, timestamp: datetime) -> None:
        """Seed a merchant's trailing-1h transaction timestamps from historical data."""
        dq = self._merchant_history[merchant_id]
        dq.append(timestamp)
        cutoff = timestamp - timedelta(hours=1)
        while dq and dq[0] < cutoff:
            dq.popleft()

    def _ingest(self, user_id: str, timestamp: datetime, amount: float, category) -> None:
        dq = self._history[user_id]
        dq.append((timestamp, amount))
        cutoff = timestamp - timedelta(days=7)
        while dq and dq[0][0] < cutoff:
            dq.popleft()

        self._count[user_id] += 1
        self._sum[user_id] += amount
        self._sumsq[user_id] += amount ** 2
        hour = timestamp.hour
        self._sum_hour_sin[user_id] += math.sin(2 * math.pi * hour / 24)
        self._sum_hour_cos[user_id] += math.cos(2 * math.pi * hour / 24)
        if category is not None:
            self._categories[user_id].add(category)
        if user_id not in self._first_seen:
            self._first_seen[user_id] = timestamp
        self._last_ts[user_id] = timestamp

    def compute_and_update(self, txn: dict) -> dict:
        """
        Compute this transaction's feature vector from state accumulated strictly
        before it (causal), then fold this transaction into that state. Mirrors
        notebook 03's cumcount / (cumsum - self) pattern, just incremental.
        """
        user_id = txn["user_id"]
        timestamp = txn["timestamp"]
        amount = txn["amount"]
        category = txn["category"]
        merchant_id = txn["merchant_id"]

        with self._lock:
            dq = self._history[user_id]
            cutoff_1h = timestamp - timedelta(hours=1)
            cutoff_24h = timestamp - timedelta(hours=24)
            cutoff_7d = timestamp - timedelta(days=7)

            while dq and dq[0][0] < cutoff_7d:
                dq.popleft()

            count_1h = sum(1 for ts, _ in dq if ts >= cutoff_1h)
            sum_1h = sum(a for ts, a in dq if ts >= cutoff_1h)
            count_24h = sum(1 for ts, _ in dq if ts >= cutoff_24h)
            sum_24h = sum(a for ts, a in dq if ts >= cutoff_24h)
            count_7d = len(dq)
            sum_7d = sum(a for _, a in dq)

            prior_n = self._count[user_id]
            if prior_n > 0:
                prior_mean = self._sum[user_id] / prior_n
                prior_var = max(self._sumsq[user_id] / prior_n - prior_mean ** 2, 1e-6)
                amount_zscore_user = (amount - prior_mean) / math.sqrt(prior_var)
            else:
                amount_zscore_user = 0.0

            hour_sin_now = math.sin(2 * math.pi * timestamp.hour / 24)
            hour_cos_now = math.cos(2 * math.pi * timestamp.hour / 24)
            if prior_n > 0:
                prior_mean_hsin = self._sum_hour_sin[user_id] / prior_n
                prior_mean_hcos = self._sum_hour_cos[user_id] / prior_n
                hour_similarity_to_user = hour_sin_now * prior_mean_hsin + hour_cos_now * prior_mean_hcos
            else:
                hour_similarity_to_user = 0.0

            is_new_category = 1 if category not in self._categories[user_id] else 0

            first_seen = self._first_seen.get(user_id, timestamp)
            card_tenure_days = (timestamp - first_seen).days

            last_ts = self._last_ts.get(user_id)
            time_since_prev_txn_sec = (
                (timestamp - last_ts).total_seconds() if last_ts is not None else 30 * 24 * 3600
            )

            # this event counts toward its own trailing window -- matches the batch
            # pipeline's inclusive `.rolling()` window over a sorted timestamp index
            count_1h += 1
            sum_1h += amount
            count_24h += 1
            sum_24h += amount
            count_7d += 1
            sum_7d += amount

            merchant_dq = self._merchant_history[merchant_id]
            merchant_cutoff_1h = timestamp - timedelta(hours=1)
            while merchant_dq and merchant_dq[0] < merchant_cutoff_1h:
                merchant_dq.popleft()
            # +1 for this transaction itself -- same inclusive-window convention as the
            # per-card velocity counts above.
            merchant_txn_count_1h = len(merchant_dq) + 1
            merchant_dq.append(timestamp)

            distance_from_home_km = haversine_km(
                txn["home_lat"], txn["home_lon"], txn["merchant_lat"], txn["merchant_lon"]
            )
            merchant_fraud_rate_prior = self.merchant_fraud_rate.get(
                merchant_id, self.global_fraud_rate
            )
            category_fraud_rate_prior = self.category_fraud_rate.get(
                category, self.global_fraud_rate
            )

            hour, dow = timestamp.hour, timestamp.weekday()
            features = {
                "amount": amount,
                "hour_sin": math.sin(2 * math.pi * hour / 24),
                "hour_cos": math.cos(2 * math.pi * hour / 24),
                "dow_sin": math.sin(2 * math.pi * dow / 7),
                "dow_cos": math.cos(2 * math.pi * dow / 7),
                "is_weekend": 1 if dow >= 5 else 0,
                "card_tenure_days": card_tenure_days,
                "user_txn_count_1h": count_1h,
                "user_amount_sum_1h": sum_1h,
                "user_txn_count_24h": count_24h,
                "user_amount_sum_24h": sum_24h,
                "user_txn_count_7d": count_7d,
                "user_amount_sum_7d": sum_7d,
                "merchant_txn_count_1h": merchant_txn_count_1h,
                "time_since_prev_txn_sec": time_since_prev_txn_sec,
                "amount_zscore_user": amount_zscore_user,
                "hour_similarity_to_user": hour_similarity_to_user,
                "is_new_category_for_user": is_new_category,
                "distance_from_home_km": distance_from_home_km,
                "merchant_fraud_rate_prior": merchant_fraud_rate_prior,
                "category_fraud_rate_prior": category_fraud_rate_prior,
            }

            self._ingest(user_id, timestamp, amount, category)

        return features
