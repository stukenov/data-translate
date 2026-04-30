import json
import os
from pathlib import Path


class CheckpointManager:
    """Saves and resumes translation progress so crashed jobs can continue."""

    def __init__(self, output_dir: str, dataset_name: str, split: str):
        safe_name = dataset_name.replace("/", "__")
        self.checkpoint_dir = Path(output_dir) / ".checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / f"{safe_name}_{split}.json"
        self._state = self._load()

    def _load(self) -> dict:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r") as f:
                return json.load(f)
        return {"translated_rows": {}, "last_index": 0}

    def save(self):
        with open(self.checkpoint_file, "w") as f:
            json.dump(self._state, f)

    @property
    def last_index(self) -> int:
        return self._state["last_index"]

    def get_translated(self, row_index: int, column: str) -> str | None:
        key = f"{row_index}:{column}"
        return self._state["translated_rows"].get(key)

    def set_translated(self, row_index: int, column: str, text: str):
        key = f"{row_index}:{column}"
        self._state["translated_rows"][key] = text

    def update_last_index(self, index: int):
        self._state["last_index"] = index

    def is_done(self, row_index: int, column: str) -> bool:
        return self.get_translated(row_index, column) is not None

    def clear(self):
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        self._state = {"translated_rows": {}, "last_index": 0}
