<p align="center">
  <h1 align="center">data-translate</h1>
  <p align="center">
    <strong>Translate entire HuggingFace datasets to any language in one command.</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &middot;
    <a href="#features">Features</a> &middot;
    <a href="#configuration">Configuration</a> &middot;
    <a href="#examples">Examples</a>
  </p>
  <p align="center">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg">
    <img alt="Python" src="https://img.shields.io/badge/python-3.10+-3776ab.svg">
    <img alt="HuggingFace" src="https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow.svg">
    <img alt="Google Translate" src="https://img.shields.io/badge/engine-Google%20Translate-4285F4.svg">
  </p>
</p>

---

Building NLP datasets for low-resource languages is painful. Most high-quality instruction datasets exist only in English. **data-translate** lets you take any HuggingFace dataset, pick the columns you want, and translate the entire thing — with automatic checkpointing, crash recovery, and direct push to the Hub.

> Built for researchers and engineers working on multilingual AI, low-resource language NLP, and cross-lingual dataset creation.

## Features

| Feature | Description |
|---|---|
| **Any dataset, any language** | Works with any HuggingFace dataset and any of 100+ languages supported by Google Translate |
| **Crash-proof** | JSON checkpointing saves progress every N rows — resume exactly where you stopped |
| **Live Hub sync** | Push partial results to HuggingFace Hub during translation, not just at the end |
| **Long text handling** | Automatically splits texts >4500 chars into chunks, translates, and reassembles |
| **Parallel workers** | Configurable thread pool for concurrent translation batches |
| **Infinite retries** | Exponential backoff with unlimited retries — the job never dies |
| **YAML + CLI** | Configure via file or command line; CLI args override YAML values |
| **Dataset builder** | Bonus script to merge Dolly + Alpaca + No Robots into a unified instruction dataset |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Translate SQuAD from English to Kazakh
python translate.py --dataset squad --columns question,context --dest-lang kk

# Translate with a config file
python translate.py --config config.example.yaml
```

That's it. Output lands in `./output/` as Parquet.

## Examples

**Translate a Q&A dataset to Turkish:**
```bash
python translate.py \
  --dataset squad \
  --columns question,context \
  --dest-lang tr \
  --batch-size 25 \
  --checkpoint-every 1000
```

**Translate instruction data to Kazakh and push to Hub:**
```bash
python translate.py \
  --dataset yahma/alpaca-cleaned \
  --columns instruction,input,output \
  --dest-lang kk \
  --push-to-hub your-username/alpaca-cleaned-kk
```

**Build a merged English instruction dataset, then translate it:**
```bash
# Step 1: merge Dolly + Alpaca + No Robots
python build_dataset.py

# Step 2: translate the merged dataset
python translate.py \
  --dataset ./data/instruct_en_merged \
  --columns instruction,output \
  --dest-lang kk
```

## Configuration

Copy the example and edit:

```bash
cp config.example.yaml config.yaml
```

```yaml
dataset: squad                  # HuggingFace dataset name or local path
columns:                        # columns to translate
  - question
  - context
split: train                    # dataset split
src_lang: en                    # source language
dest_lang: kk                   # target language (any Google Translate code)
batch_size: 20                  # texts per translation batch
max_workers: 3                  # parallel threads
retry_attempts: 5               # retries before switching to 5-min backoff
retry_delay: 2.0                # initial retry delay (doubles each attempt)
checkpoint_every: 500           # save progress every N rows
output_dir: ./output            # local output directory
output_format: parquet          # parquet or jsonl
push_to_hub: user/dataset-kk   # optional: HuggingFace Hub repo
push_every: 1000                # push to Hub every N rows
```

All options are also available as CLI flags. Run `python translate.py --help` for the full list.

## HuggingFace Hub Integration

```bash
# Option 1: environment variable
export HF_TOKEN=hf_...
python translate.py --config config.yaml

# Option 2: CLI flag
python translate.py --config config.yaml --hf-token hf_...
```

Partial results are pushed every `push_every` rows, so your Hub dataset stays up to date even during long-running jobs.

## Checkpointing & Resume

Translation state is saved to `output/.checkpoints/`. If the process crashes or is interrupted:

```bash
# Just run the same command again — it resumes automatically
python translate.py --config config.yaml

# Or start fresh
python translate.py --config config.yaml --fresh
```

## Project Structure

```
data-translate/
├── translate.py           # CLI entry point
├── build_dataset.py       # Merge instruction datasets (Dolly, Alpaca, No Robots)
├── config.example.yaml    # Example configuration
├── requirements.txt       # Python dependencies
└── translator/
    ├── core.py            # BatchTranslator — translation engine
    ├── checkpoint.py      # CheckpointManager — crash recovery
    └── hf_utils.py        # HuggingFace load/save/push utilities
```

## Supported Languages

Any language pair supported by [Google Translate](https://cloud.google.com/translate/docs/languages) — 100+ languages including:

`af` `sq` `am` `ar` `hy` `az` `eu` `be` `bn` `bs` `bg` `ca` `zh` `hr` `cs` `da` `nl` `en` `et` `fi` `fr` `gl` `ka` `de` `el` `gu` `ht` `ha` `he` `hi` `hu` `is` `id` `ga` `it` `ja` `jv` `kn` `kk` `km` `ko` `ku` `ky` `lo` `la` `lv` `lt` `lb` `mk` `mg` `ms` `ml` `mt` `mi` `mr` `mn` `my` `ne` `no` `ny` `or` `ps` `fa` `pl` `pt` `pa` `ro` `ru` `sm` `gd` `sr` `st` `sn` `sd` `si` `sk` `sl` `so` `es` `su` `sw` `sv` `tl` `tg` `ta` `tt` `te` `th` `tr` `tk` `uk` `ur` `ug` `uz` `vi` `cy` `xh` `yi` `yo` `zu`

## Contributing

Contributions are welcome! Feel free to open issues and pull requests.

## License

[MIT](LICENSE)
