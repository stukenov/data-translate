#!/usr/bin/env python3
"""Download quality instruction datasets and merge into a unified format.

Unified schema:
  - instruction: str  (the task/question)
  - input: str        (optional context, empty string if none)
  - output: str       (the response/answer)
  - source: str       (original dataset name)
"""

import logging

from datasets import Dataset, concatenate_datasets, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_dolly() -> Dataset:
    """databricks/databricks-dolly-15k — 15K human-written instructions."""
    logger.info("Loading databricks-dolly-15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    logger.info(f"  Loaded {len(ds)} rows")

    def transform(row):
        return {
            "instruction": row["instruction"],
            "input": row.get("context", "") or "",
            "output": row["response"],
            "source": "dolly-15k",
        }

    return ds.map(transform, remove_columns=ds.column_names)


def load_alpaca_cleaned() -> Dataset:
    """yahma/alpaca-cleaned — 52K cleaned Stanford Alpaca."""
    logger.info("Loading alpaca-cleaned...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    logger.info(f"  Loaded {len(ds)} rows")

    def transform(row):
        return {
            "instruction": row["instruction"],
            "input": row.get("input", "") or "",
            "output": row["output"],
            "source": "alpaca-cleaned",
        }

    return ds.map(transform, remove_columns=ds.column_names)


def load_no_robots() -> Dataset:
    """HuggingFaceH4/no_robots — 10K human-written instructions."""
    logger.info("Loading no_robots...")
    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    logger.info(f"  Loaded {len(ds)} rows")

    def transform(row):
        messages = row.get("messages", [])
        instruction = ""
        output = ""
        for msg in messages:
            if msg["role"] == "user":
                instruction = msg["content"]
            elif msg["role"] == "assistant":
                output = msg["content"]
        return {
            "instruction": instruction,
            "input": "",
            "output": output,
            "source": "no_robots",
        }

    return ds.map(transform, remove_columns=ds.column_names)


def main():
    datasets = []

    for loader in [load_dolly, load_alpaca_cleaned, load_no_robots]:
        try:
            ds = loader()
            datasets.append(ds)
        except Exception as e:
            logger.error(f"Failed to load {loader.__name__}: {e}")

    if not datasets:
        logger.error("No datasets loaded!")
        return

    logger.info("Merging datasets...")
    merged = concatenate_datasets(datasets)
    logger.info(f"Total rows: {len(merged)}")

    # Filter out empty instructions or outputs
    before = len(merged)
    merged = merged.filter(lambda row: bool(row["instruction"].strip()) and bool(row["output"].strip()))
    logger.info(f"After filtering empty rows: {len(merged)} (removed {before - len(merged)})")

    # Shuffle
    merged = merged.shuffle(seed=42)

    # Save
    out_path = "./data/instruct_en_merged"
    merged.save_to_disk(out_path)
    logger.info(f"Saved to {out_path}")

    # Also save as parquet for easy inspection
    merged.to_parquet("./data/instruct_en_merged.parquet")
    logger.info("Saved parquet: ./data/instruct_en_merged.parquet")

    # Stats
    logger.info("--- Stats ---")
    for src in ["dolly-15k", "alpaca-cleaned", "no_robots"]:
        count = len(merged.filter(lambda r: r["source"] == src))
        logger.info(f"  {src}: {count}")

    # Preview
    logger.info("--- Preview ---")
    for i in range(3):
        row = merged[i]
        logger.info(f"  [{row['source']}] {row['instruction'][:80]}...")


if __name__ == "__main__":
    main()
