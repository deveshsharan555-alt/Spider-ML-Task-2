"""
Confidence estimation.


"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.reranker import RankedChunk

MIN_RELEVANCE_THRESHOLD = 0.15  # below this, a chunk doesn't really "count" as evidence


@dataclass
class ConfidenceResult:
    score: float          # 0.0 - 1.0
    label: str            # "High" | "Medium" | "Low" | "Insufficient evidence"
    rationale: str


def estimate_confidence(ranked_chunks: List[RankedChunk]) -> ConfidenceResult:
    relevant = [rc for rc in ranked_chunks if rc.rerank_score >= MIN_RELEVANCE_THRESHOLD]

    if not relevant:
        return ConfidenceResult(
            score=0.0,
            label="Insufficient evidence",
            rationale="No retrieved passage cleared the minimum relevance threshold "
                       "for this query.",
        )

    top_score = relevant[0].rerank_score
    scores = [rc.rerank_score for rc in relevant]
    margin = top_score - (scores[1] if len(scores) > 1 else 0.0)
    distinct_sources = len({rc.chunk.source for rc in relevant})
    coverage = min(len(relevant) / 3.0, 1.0)  # saturates at 3+ supporting chunks

    agreement_bonus = min((distinct_sources - 1) * 0.15, 0.3)

    raw = 0.5 * top_score + 0.15 * min(margin * 2, 1.0) + 0.2 * coverage + agreement_bonus
    raw = max(0.0, min(raw, 1.0))

    if raw >= 0.66:
        label = "High"
    elif raw >= 0.4:
        label = "Medium"
    else:
        label = "Low"

    rationale_parts = [
        f"top match relevance {top_score:.2f}",
        f"{len(relevant)} supporting passage(s)",
        f"from {distinct_sources} distinct source(s)",
    ]
    if distinct_sources > 1:
        rationale_parts.append("independent sources agree")
    else:
        rationale_parts.append("only one source found — not independently corroborated")

    return ConfidenceResult(score=round(raw, 2), label=label, rationale="; ".join(rationale_parts))


def detect_conflict(ranked_chunks: List[RankedChunk]) -> bool:
    
    import re

    numbers_by_source: dict[str, set[str]] = {}
    for rc in ranked_chunks[:5]:
        nums = set(re.findall(r"\d{2,3}/\d{2,3}|\d+\s?mg|\d+\s?mmHg", rc.chunk.text))
        if nums:
            numbers_by_source.setdefault(rc.chunk.source, set()).update(nums)

    all_number_sets = list(numbers_by_source.values())
    if len(all_number_sets) < 2:
        return False
    
    
    common = set.intersection(*all_number_sets) if all_number_sets else set()
    return len(common) == 0
