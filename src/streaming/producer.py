"""
Replays real transactions from the held-out test split onto a Kafka topic, simulating a
live transaction feed. Historical timestamps are preserved in the payload (so downstream
features/logic see realistic data) but messages are published at a compressed,
configurable wall-clock cadence -- the real test window spans months, nobody's waiting
that long for a demo.

Usage:
    python -m src.streaming.producer --limit 2000 --delay 0.05
"""
import argparse
import json
import logging
import os
import time

import pandas as pd
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("producer")

DATA_DIR = os.environ.get("DATA_DIR", "data")
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.environ.get("TRANSACTIONS_TOPIC", "transactions")


def load_stream_df(limit: int | None) -> pd.DataFrame:
    test = pd.read_parquet(os.path.join(DATA_DIR, "processed", "test.parquet"))
    raw = pd.read_parquet(
        os.path.join(DATA_DIR, "raw", "transactions.parquet"),
        columns=["transaction_id", "home_lat", "home_lon", "merchant_lat", "merchant_lon"],
    )
    df = test.merge(raw, on="transaction_id", how="left").sort_values("timestamp")
    if limit:
        df = df.head(limit)
    return df


def to_payload(row) -> dict:
    return {
        "transaction_id": row.transaction_id,
        "timestamp": row.timestamp.isoformat(),
        "user_id": row.user_id,
        "amount": float(row.amount),
        "merchant_id": row.merchant_id,
        "category": row.category,
        "home_lat": float(row.home_lat),
        "home_lon": float(row.home_lon),
        "merchant_lat": float(row.merchant_lat),
        "merchant_lon": float(row.merchant_lon),
        # not sent to /score (not part of TransactionIn) -- carried for the consumer's
        # own bookkeeping/accuracy stats only, since real labels lag in production
        "_true_label": int(row.is_fraud),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "2000")))
    parser.add_argument("--delay", type=float, default=float(os.environ.get("DELAY_SECONDS", "0.05")))
    args = parser.parse_args()

    df = load_stream_df(args.limit)
    logger.info("Replaying %d transactions to topic '%s' (%.3fs between messages)",
                len(df), TOPIC, args.delay)

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )

    sent = 0
    for row in df.itertuples(index=False):
        payload = to_payload(row)
        producer.send(TOPIC, key=payload["user_id"], value=payload)
        sent += 1
        if sent % 200 == 0:
            logger.info("sent %d/%d", sent, len(df))
        time.sleep(args.delay)

    producer.flush()
    logger.info("Done -- sent %d transactions", sent)


if __name__ == "__main__":
    main()
