from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk


def load_hf_dataset(dataset_name: str, split: str, subset: str | None = None) -> Dataset:
    """Load a dataset from HuggingFace Hub or local disk."""
    local_path = Path(dataset_name)
    if local_path.exists():
        return load_from_disk(str(local_path))
    kwargs = {"path": dataset_name, "split": split}
    if subset:
        kwargs["name"] = subset
    return load_dataset(**kwargs)


def save_dataset(dataset: Dataset, output_dir: str, dataset_name: str, split: str, fmt: str = "parquet"):
    """Save translated dataset locally as parquet or jsonl."""
    safe_name = dataset_name.replace("/", "__")
    out_path = Path(output_dir) / safe_name
    out_path.mkdir(parents=True, exist_ok=True)

    file_path = out_path / f"{split}.{fmt}"
    if fmt == "parquet":
        dataset.to_parquet(str(file_path))
    elif fmt == "jsonl":
        dataset.to_json(str(file_path))
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return file_path


def push_dataset(dataset: Dataset, repo_id: str, split: str, token: str | None = None):
    """Push translated dataset to HuggingFace Hub."""
    dataset.push_to_hub(repo_id, split=split, token=token)
