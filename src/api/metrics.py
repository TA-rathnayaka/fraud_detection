from prometheus_client import Counter, Histogram

SCORE_LATENCY = Histogram(
    "fraud_score_latency_seconds",
    "Latency of /score requests, end-to-end (feature computation + inference + SHAP)",
    buckets=(0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1),
)

PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total scored transactions by decision",
    ["decision"],
)

SCORE_DISTRIBUTION = Histogram(
    "fraud_score_distribution",
    "Distribution of predicted fraud probabilities",
    buckets=[i / 20 for i in range(21)],
)
