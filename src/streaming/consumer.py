"""
Consumes the live transaction feed from Kafka, scores each transaction against the
FastAPI real-time scoring service, republishes the result to a 'scored_transactions'
topic, and logs fraud alerts. This is the piece that turns the batch-trained model into
a streaming pipeline: producer -> Kafka -> consumer -> scoring API -> Kafka -> (anywhere
downstream that wants scored events -- a case-management queue, a data warehouse sink).

Usage:
    python -m src.streaming.consumer
"""
import json
import logging
import os
import time

import httpx
from kafka import KafkaConsumer, KafkaProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("consumer")

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
IN_TOPIC = os.environ.get("TRANSACTIONS_TOPIC", "transactions")
OUT_TOPIC = os.environ.get("SCORED_TOPIC", "scored_transactions")
API_URL = os.environ.get("SCORING_API_URL", "http://localhost:8000")


def main():
    consumer = KafkaConsumer(
        IN_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="latest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="fraud-scoring-consumer",
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    client = httpx.Client(base_url=API_URL, timeout=5.0)

    logger.info("Waiting for the scoring API to be ready at %s ...", API_URL)
    for _ in range(60):
        try:
            if client.get("/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(2)
    else:
        raise RuntimeError(f"Scoring API at {API_URL} never became ready")

    logger.info("Consuming from '%s', scoring via %s, publishing to '%s'",
                IN_TOPIC, API_URL, OUT_TOPIC)

    n_scored, n_flagged, n_matches_true_label = 0, 0, 0
    for msg in consumer:
        txn = msg.value
        true_label = txn.pop("_true_label", None)

        try:
            resp = client.post("/score", json=txn)
            resp.raise_for_status()
            result = resp.json()
        except httpx.HTTPError as e:
            logger.error("scoring failed for %s: %s", txn.get("transaction_id"), e)
            continue

        n_scored += 1
        flagged = result["decision"] != "approve"
        n_flagged += flagged
        if true_label is not None and flagged == bool(true_label):
            n_matches_true_label += 1

        producer.send(OUT_TOPIC, value={**result, "true_label": true_label})

        if flagged:
            top = result["top_reasons"][0]
            logger.warning(
                "FLAGGED %s  score=%.4f  amount=$%.2f  top_reason=%s (%+.2f)  [true_label=%s]",
                txn["transaction_id"], result["score"], txn["amount"],
                top["feature"], top["shap_contribution"], true_label,
            )

        if n_scored % 200 == 0:
            logger.info("scored=%d flagged=%d agreement_with_true_label=%.1f%%",
                        n_scored, n_flagged,
                        100 * n_matches_true_label / n_scored if n_scored else 0)


if __name__ == "__main__":
    main()
