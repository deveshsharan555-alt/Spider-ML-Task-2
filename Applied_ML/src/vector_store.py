"""
Vector store / retrieval index.

"""
from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.chunking import Chunk


class NumpyVectorStore:
    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        assert len(chunks) == vectors.shape[0]
        self.chunks = chunks
        self.vectors = vectors

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        if self.vectors is None or len(self.chunks) == 0:
            return []
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(query_vector.reshape(1, -1), self.vectors)[0]
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectors": self.vectors,
                    "chunks": [asdict(c) for c in self.chunks],
                },
                f,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectors = data["vectors"]
        self.chunks = [Chunk(**c) for c in data["chunks"]]


class FaissVectorStore:
    

    def __init__(self):
        self.index = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        import faiss  # optional dep

        self.chunks = chunks
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # vectors must already be L2-normalized
        self.index.add(vectors.astype(np.float32))

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        if self.index is None:
            return []
        scores, idx = self.index.search(query_vector.reshape(1, -1).astype(np.float32), top_k)
        results = []
        for score, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            results.append((self.chunks[i], float(score)))
        return results

    def save(self, path: str) -> None:
        import faiss

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".meta", "wb") as f:
            pickle.dump([asdict(c) for c in self.chunks], f)

    def load(self, path: str) -> None:
        import faiss

        self.index = faiss.read_index(path + ".faiss")
        with open(path + ".meta", "rb") as f:
            self.chunks = [Chunk(**c) for c in pickle.load(f)]


def get_vector_store(backend: str = "numpy"):
    if backend == "numpy":
        return NumpyVectorStore()
    if backend == "faiss":
        return FaissVectorStore()
    raise ValueError(f"Unknown vector store backend: {backend}")
