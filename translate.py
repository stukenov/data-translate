#!/usr/bin/env python3
"""CLI entry point for data-translate — mass translate HuggingFace datasets via Google Translate."""

import argparse
import logging
import os
import sys

import yaml

from translator.checkpoint import CheckpointManager
from translator.core import BatchTranslator
from translator.hf_utils import load_hf_dataset, push_dataset, save_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Translate HuggingFace datasets via Google Translate")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name (e.g. squad)")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset/config name")
    parser.add_argument("--columns", type=str, help="Comma-separated column names to translate")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--src-lang", type=str, default="en", help="Source language (default: en)")
    parser.add_argument("--dest-lang", type=str, default="kk", help="Target language (default: kk)")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for translation (default: 20)")
    parser.add_argument("--max-workers", type=int, default=3, help="Parallel workers (default: 3)")
    parser.add_argument("--retry-attempts", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--output-format", type=str, default="parquet", choices=["parquet", "jsonl"])
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push (e.g. user/dataset-kk)")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token for push")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint (default: true)")
    parser.add_argument("--push-every", type=int, default=1000, help="Push to HF Hub every N rows (default: 1000)")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint, start fresh")
    return parser.parse_args()


def load_config(args) -> dict:
    """Merge YAML config with CLI args. CLI args take precedence."""
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    # CLI overrides
    if args.dataset:
        config["dataset"] = args.dataset
    if args.subset:
        config["subset"] = args.subset
    if args.columns:
        config["columns"] = [c.strip() for c in args.columns.split(",")]
    if args.split != "train" or "split" not in config:
        config.setdefault("split", args.split)
    if args.push_to_hub:
        config["push_to_hub"] = args.push_to_hub
    if args.hf_token:
        config["hf_token"] = args.hf_token

    config.setdefault("src_lang", args.src_lang)
    config.setdefault("dest_lang", args.dest_lang)
    config.setdefault("batch_size", args.batch_size)
    config.setdefault("max_workers", args.max_workers)
    config.setdefault("retry_attempts", args.retry_attempts)
    config.setdefault("retry_delay", args.retry_delay)
    config.setdefault("checkpoint_every", args.checkpoint_every)
    config.setdefault("output_dir", args.output_dir)
    config.setdefault("output_format", args.output_format)
    config.setdefault("push_every", args.push_every)
    config.setdefault("fresh", args.fresh)

    # HF token from env if not set
    if "hf_token" not in config:
        config["hf_token"] = os.environ.get("HF_TOKEN")

    if "dataset" not in config:
        print("Error: --dataset or config.dataset is required")
        sys.exit(1)
    if "columns" not in config:
        print("Error: --columns or config.columns is required")
        sys.exit(1)

    return config


def main():
    args = parse_args()
    config = load_config(args)

    dataset_name = config["dataset"]
    columns = config["columns"]
    split = config["split"]

    logger.info(f"Dataset: {dataset_name} (split={split})")
    logger.info(f"Columns: {columns}")
    logger.info(f"Direction: {config['src_lang']} -> {config['dest_lang']}")

    # Load dataset
    logger.info("Loading dataset from HuggingFace...")
    ds = load_hf_dataset(dataset_name, split, config.get("subset"))
    logger.info(f"Loaded {len(ds)} rows")

    # Validate columns
    for col in columns:
        if col not in ds.column_names:
            print(f"Error: column '{col}' not found. Available: {ds.column_names}")
            sys.exit(1)

    # Checkpoint
    checkpoint = CheckpointManager(config["output_dir"], dataset_name, split)
    if config.get("fresh"):
        checkpoint.clear()
        logger.info("Starting fresh (checkpoint cleared)")

    # Translate
    translator = BatchTranslator(
        src_lang=config["src_lang"],
        dest_lang=config["dest_lang"],
        batch_size=config["batch_size"],
        max_workers=config["max_workers"],
        retry_attempts=config["retry_attempts"],
        retry_delay=config["retry_delay"],
        checkpoint_every=config["checkpoint_every"],
    )

    hub_repo = config.get("push_to_hub")
    hf_token = config.get("hf_token")
    push_every = config.get("push_every", 1000)

    def on_checkpoint(rows_done, total):
        """Called every checkpoint_every rows. Push partial results to HF Hub."""
        if not hub_repo or rows_done % push_every != 0:
            return
        try:
            from datasets import Dataset as DS
            partial = {}
            for col in ds.column_names:
                if col in columns:
                    partial[col] = [translated[col][j] or "" for j in range(rows_done)]
                else:
                    partial[col] = [ds[j][col] for j in range(rows_done)]
            partial_ds = DS.from_dict(partial)
            logger.info(f"Pushing checkpoint ({rows_done}/{total}) to {hub_repo}...")
            push_dataset(partial_ds, hub_repo, split, hf_token)
            logger.info(f"Checkpoint push done ({rows_done} rows)")
        except Exception as e:
            logger.error(f"Checkpoint push failed: {e}")

    translated = translator.translate_dataset(ds, columns, checkpoint, on_checkpoint=on_checkpoint)

    # Build final dataset
    new_data = {}
    for col in ds.column_names:
        if col in columns:
            new_data[col] = translated[col]
        else:
            new_data[col] = ds[col]

    from datasets import Dataset
    translated_ds = Dataset.from_dict(new_data)

    # Save locally
    out_path = save_dataset(translated_ds, config["output_dir"], dataset_name, split, config.get("output_format", "parquet"))
    logger.info(f"Saved to {out_path}")

    # Final push to Hub
    if hub_repo:
        logger.info(f"Final push to HuggingFace Hub: {hub_repo}")
        push_dataset(translated_ds, hub_repo, split, hf_token)
        logger.info("Push complete!")

    logger.info("Done!")


if __name__ == "__main__":
    main()
