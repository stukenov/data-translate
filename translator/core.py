import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from deep_translator import GoogleTranslator
from tqdm import tqdm

from translator.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class BatchTranslator:
    """Translates dataset columns using deep-translator with batching, retries, and checkpointing."""

    def __init__(
        self,
        src_lang: str = "en",
        dest_lang: str = "kk",
        batch_size: int = 20,
        max_workers: int = 3,
        retry_attempts: int = 5,
        retry_delay: float = 2.0,
        checkpoint_every: int = 500,
    ):
        self.src_lang = src_lang
        self.dest_lang = dest_lang
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.checkpoint_every = checkpoint_every

    def _create_translator(self) -> GoogleTranslator:
        return GoogleTranslator(source=self.src_lang, target=self.dest_lang)

    def _translate_single(self, text: str, worker_id: int) -> str:
        """Translate a single text, splitting into chunks if > 4500 chars."""
        if len(text) <= 4500:
            return self._translate_with_retry([text], worker_id)[0]

        # Split long text by paragraphs, then by sentences
        chunks = []
        current = ""
        for para in text.split("\n"):
            if len(current) + len(para) + 1 > 4500:
                if current:
                    chunks.append(current)
                current = para[:4500] if len(para) > 4500 else para
            else:
                current = current + "\n" + para if current else para
        if current:
            chunks.append(current)

        translated_chunks = []
        for chunk in chunks:
            result = self._translate_with_retry([chunk], worker_id)
            translated_chunks.append(result[0])
        return "\n".join(translated_chunks)

    def _translate_with_retry(self, texts: list[str], worker_id: int) -> list[str]:
        """Translate with retry logic. Never crashes."""
        translator = self._create_translator()
        attempt = 0
        while True:
            try:
                results = translator.translate_batch(texts)
                return [r or "" for r in results]
            except Exception as e:
                attempt += 1
                if attempt <= self.retry_attempts:
                    delay = self.retry_delay * (2 ** attempt)
                else:
                    delay = 300
                logger.warning(f"Worker {worker_id} attempt {attempt} failed: {e}. Retrying in {delay:.0f}s")
                time.sleep(delay)
                translator = self._create_translator()

    def _translate_batch_sync(self, texts: list[str], worker_id: int) -> list[str]:
        """Translate a batch of texts. Handles long texts by splitting."""
        results = []
        for text in texts:
            results.append(self._translate_single(text, worker_id))
        return results

    def translate_dataset(self, dataset, columns: list[str], checkpoint: CheckpointManager, on_checkpoint=None) -> dict:
        """Translate specified columns of a HF dataset. Returns dict of {column: [translated_texts]}."""
        total_rows = len(dataset)
        start_index = checkpoint.last_index
        if start_index > 0:
            logger.info(f"Resuming from row {start_index}/{total_rows}")

        translated = {col: [None] * total_rows for col in columns}

        # Restore already-translated rows from checkpoint
        for i in range(start_index):
            for col in columns:
                cached = checkpoint.get_translated(i, col)
                if cached is not None:
                    translated[col][i] = cached

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                self._translate_all(dataset, columns, translated, checkpoint, start_index, total_rows, on_checkpoint)
            )
        finally:
            loop.close()

        return translated

    async def _translate_all(self, dataset, columns, translated, checkpoint, start_index, total_rows, on_checkpoint=None):
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        loop = asyncio.get_event_loop()

        pbar = tqdm(total=total_rows * len(columns), initial=start_index * len(columns), desc="Translating")
        rows_since_checkpoint = 0

        for i in range(start_index, total_rows):
            tasks = []
            for col in columns:
                if checkpoint.is_done(i, col):
                    translated[col][i] = checkpoint.get_translated(i, col)
                    pbar.update(1)
                    continue

                text = dataset[i][col]
                if not text or not isinstance(text, str) or not text.strip():
                    translated[col][i] = text or ""
                    checkpoint.set_translated(i, col, text or "")
                    pbar.update(1)
                    continue

                tasks.append((col, text))

            # Batch translate all columns for this row
            if tasks:
                cols_batch, texts_batch = zip(*tasks)
                # Split into sub-batches if needed
                for batch_start in range(0, len(texts_batch), self.batch_size):
                    batch_end = min(batch_start + self.batch_size, len(texts_batch))
                    batch_texts = list(texts_batch[batch_start:batch_end])
                    batch_cols = cols_batch[batch_start:batch_end]

                    worker_id = i % self.max_workers
                    results = await loop.run_in_executor(
                        executor, self._translate_batch_sync, batch_texts, worker_id
                    )

                    for col, result_text in zip(batch_cols, results):
                        translated[col][i] = result_text
                        checkpoint.set_translated(i, col, result_text)
                        pbar.update(1)

            checkpoint.update_last_index(i + 1)
            rows_since_checkpoint += 1

            if rows_since_checkpoint >= self.checkpoint_every:
                checkpoint.save()
                if on_checkpoint:
                    on_checkpoint(i + 1, total_rows)
                rows_since_checkpoint = 0

        # Final save
        checkpoint.save()
        pbar.close()
        executor.shutdown(wait=False)
