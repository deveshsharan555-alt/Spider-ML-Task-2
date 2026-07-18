"""
Embedding generation.

"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np


class Embedder(ABC):
    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        ...

    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        return self.transform([text])[0]

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...


class TfidfEmbedder(Embedder):
    

    name = "tfidf"

    def __init__(self, max_features: int = 20000, ngram_range=(1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit() before transform().")
        mat = self.vectorizer.transform(texts)
        from sklearn.preprocessing import normalize

        mat = normalize(mat)
        return mat.toarray().astype(np.float32)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._fitted = True


class SentenceTransformerEmbedder(Embedder):
    """Dense semantic embeddings. Requires `pip install sentence-transformers`."""

    name = "sentence-transformer"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dep

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: List[str]) -> None:
        # No training needed for a pretrained sentence encoder.
        pass

    def transform(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    def save(self, path: str) -> None:
        Path(path).write_text(self.model_name)

    def load(self, path: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = Path(path).read_text().strip()
        self.model = SentenceTransformer(self.model_name)


def get_embedder(backend: str = "tfidf") -> Embedder:
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend in ("sentence-transformer", "st", "dense"):
        try:
            return SentenceTransformerEmbedder()
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run `pip install sentence-transformers` or use backend='tfidf'."
            ) from e
    raise ValueError(f"Unknown embedding backend: {backend}")
