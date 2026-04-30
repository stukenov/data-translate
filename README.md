# data-translate

Mass translate HuggingFace datasets to any language via Google Translate.

Features:
- Translates any HuggingFace dataset columns to a target language
- Checkpointing — resume interrupted jobs automatically
- Periodic push to HuggingFace Hub during translation
- Handles long texts by splitting into chunks
- Configurable via YAML or CLI arguments

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Translate SQuAD questions and contexts from English to Kazakh
python translate.py --dataset squad --columns question,context --dest-lang kk

# Use a config file
python translate.py --config config.example.yaml
```

## Configuration

Copy and edit the example config:

```bash
cp config.example.yaml config.yaml
```

```yaml
dataset: squad              # HuggingFace dataset name or local path
columns:
  - question
  - context
split: train
src_lang: en                # source language
dest_lang: kk               # target language
batch_size: 20
max_workers: 3
retry_attempts: 5
retry_delay: 2.0
checkpoint_every: 500
output_dir: ./output
push_to_hub: user/dataset   # optional: push to HuggingFace Hub
```

CLI arguments override config file values. Run `python translate.py --help` for all options.

## Building Instruction Datasets

`build_dataset.py` downloads and merges several English instruction datasets (Dolly, Alpaca, No Robots) into a unified format for translation:

```bash
python build_dataset.py
```

Output is saved to `data/instruct_en_merged/`.

## Pushing to HuggingFace Hub

Set your token via environment variable:

```bash
export HF_TOKEN=hf_...
```

Or pass it via CLI:

```bash
python translate.py --config config.yaml --push-to-hub your-username/dataset-translated --hf-token hf_...
```

## Resuming

Checkpoints are saved to `output/.checkpoints/`. Translation automatically resumes from the last checkpoint. Use `--fresh` to start over.

## License

MIT
