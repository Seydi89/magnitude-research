from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Dependency-free smoke-test backend; not the final semantic experiment."""

    name = "hashing-word-char"

    def __init__(self, dimensions: int = 1024):
        self.word = HashingVectorizer(
            n_features=dimensions,
            alternate_sign=False,
            ngram_range=(1, 2),
            norm="l2",
        )
        self.char = HashingVectorizer(
            analyzer="char_wb",
            n_features=dimensions,
            alternate_sign=False,
            ngram_range=(3, 5),
            norm="l2",
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        word = self.word.transform(texts).toarray()
        char = self.char.transform(texts).toarray()
        result = np.hstack([word, char])
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install -e '.[semantic]'"
            ) from exc
        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        )


def make_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sentence-transformer":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown embedding backend: {backend}")


def stable_text_key(texts: list[str], embedder_name: str) -> str:
    payload = embedder_name + "\0" + "\0".join(texts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

