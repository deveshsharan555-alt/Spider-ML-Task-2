"""
Reranking module.

"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from src.chunking import Chunk

_WORD = re.compile(r"[a-zA-Z]+")


def _lexical_overlap(query: str, text: str) -> float:
    q_terms = set(w.lower() for w in _WORD.findall(query))
    t_terms = set(w.lower() for w in _WORD.findall(text))
    if not q_terms:
        return 0.0
    return len(q_terms & t_terms) / len(q_terms)


@dataclass
class RankedChunk:
    chunk: Chunk
    retrieval_score: float
    rerank_score: float



class HeuristicReranker:
    """Dependency-free reranker: blends retrieval score with lexical overlap."""

    def score(self, query: str, chunks: List[Chunk], retrieval_scores: List[float]) -> List[float]:
        scores = []
        for chunk, r_score in zip(chunks, retrieval_scores):
            overlap = _lexical_overlap(query, chunk.text)
            # Weighted blend: semantic similarity still dominant, lexical overlap
            # rewards passages that literally contain the medical terms asked about.
            scores.append(0.7 * r_score + 0.3 * overlap)
        return scores


def rerank(
    query: str,
    candidates: List[Tuple[Chunk, float]],
    top_k: int = 5,
    diversify_sources: bool = True,
) -> List[RankedChunk]:
    chunks = [c for c, _ in candidates]
    retrieval_scores = [s for _, s in candidates]

    rerank_scores = HeuristicReranker().score(query, chunks, retrieval_scores)

    ranked = [
        RankedChunk(chunk=c, retrieval_score=r, rerank_score=s)
        for c, r, s in zip(chunks, retrieval_scores, rerank_scores)
    ]
    ranked.sort(key=lambda rc: rc.rerank_score, reverse=True)

    if not diversify_sources:
        return ranked[:top_k]

    # Greedy selection that mildly favors source diversity once relevance is close,
    # so a hypertension answer can show MedQuAD + WHO + NICE agreeing, not just
    # five near-duplicate MedQuAD passages.
    selected: List[RankedChunk] = []
    seen_sources = set()
    remaining = list(ranked)
    while remaining and len(selected) < top_k:
        best_idx, best_val = 0, -1.0
        for i, rc in enumerate(remaining):
            bonus = 0.05 if rc.chunk.source not in seen_sources else 0.0
            val = rc.rerank_score + bonus
            if val > best_val:
                best_val, best_idx = val, i
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        seen_sources.add(chosen.chunk.source)
    return selected
