from collections.abc import Sequence

import numpy as np
from fastembed import TextEmbedding
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from board_game_rules.config import Settings


class Embeddings:
    def __init__(self, settings: Settings) -> None:
        metadata = next(
            (
                m
                for m in TextEmbedding.list_supported_models()
                if m["model"] == settings.embedding_model
            ),
            None,
        )
        if metadata is None or not metadata["sources"]["hf"]:
            raise ValueError(f"No supported embedding/tokenizer model: {settings.embedding_model}")
        self.dimension = int(metadata["dim"])
        self.model_name = settings.embedding_model
        tokenizer = AutoTokenizer.from_pretrained(
            metadata["sources"]["hf"], trust_remote_code=False
        )
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError("Embedding model did not supply a compatible tokenizer.")
        self.tokenizer = tokenizer
        self.max_tokens = int(self.tokenizer.model_max_length)
        if self.max_tokens > 1_000_000:
            raise ValueError("Embedding tokenizer does not declare a usable context limit.")
        if settings.chunk_tokens > self.max_tokens:
            raise ValueError("Chunk token budget exceeds the embedding model limit.")
        self.model = TextEmbedding(model_name=self.model_name, threads=4)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def validate_text(self, text: str) -> None:
        if not text.strip():
            raise ValueError("Cannot embed blank text.")
        if self.count_tokens(text) > self.max_tokens:
            raise ValueError("Embedding input exceeds model context; truncation is not allowed.")

    def passages(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("Cannot embed an empty batch.")
        for text in texts:
            self.validate_text(text)
        vectors = [vector.tolist() for vector in self.model.passage_embed(texts)]
        self.validate_vectors(vectors, len(texts))
        return vectors

    def query(self, text: str) -> list[float]:
        self.validate_text(text)
        vectors = [vector.tolist() for vector in self.model.query_embed(text)]
        self.validate_vectors(vectors, 1)
        return vectors[0]

    def validate_vectors(self, vectors: list[list[float]], expected: int) -> None:
        if len(vectors) != expected:
            raise ValueError("Embedding vector count does not match input count.")
        for vector in vectors:
            if len(vector) != self.dimension or not np.isfinite(vector).all():
                raise ValueError("Embedding returned an invalid vector.")
