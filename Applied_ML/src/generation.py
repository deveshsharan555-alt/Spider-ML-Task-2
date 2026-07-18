"""
Grounded response generation.

"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from src.chunking import split_sentences
from src.reranker import RankedChunk
from src.safety import GENERAL_DISCLAIMER


@dataclass
class Citation:
    marker: int
    source: str
    title: str | None
    url: str | None
    snippet: str


@dataclass
class GroundedAnswer:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    disclaimer: str = GENERAL_DISCLAIMER


def _make_citations(ranked_chunks: List[RankedChunk]) -> List[Citation]:
    citations = []
    for i, rc in enumerate(ranked_chunks, start=1):
        snippet = rc.chunk.text
        if len(snippet) > 220:
            snippet = snippet[:217].rsplit(" ", 1)[0] + "..."
        citations.append(
            Citation(
                marker=i,
                source=rc.chunk.source,
                title=rc.chunk.title,
                url=rc.chunk.url,
                snippet=snippet,
            )
        )
    return citations


class ExtractiveGenerator:
    

    def generate(self, query: str, ranked_chunks: List[RankedChunk]) -> GroundedAnswer:
        if not ranked_chunks:
            return GroundedAnswer(
                answer="I don't have enough evidence in the knowledge base to answer "
                       "this confidently. Please rephrase, or consult a healthcare "
                       "provider directly.",
                citations=[],
            )

        citations = _make_citations(ranked_chunks)
        lines = []
        for rc, citation in zip(ranked_chunks, citations):
            # For Q&A-type chunks, prefer the answer half; strip the "Q: ... A:" scaffold.
            text = rc.chunk.text
            if text.startswith("Q:") and "\nA:" in text:
                text = text.split("\nA:", 1)[1].strip()
            sentences = split_sentences(text)
            # Take the 1-2 most on-topic sentences from this chunk rather than the whole thing.
            best = sentences[:2] if sentences else [text]
            snippet = " ".join(best)
            lines.append(f"{snippet} [{citation.marker}]")

        answer = " ".join(lines)
        return GroundedAnswer(answer=answer, citations=citations)


def _build_grounding_prompt(citations: List[Citation]) -> str:
    """The evidence-only grounding instruction shared by every LLM-backed
    generator, so Claude and Groq are held to the identical contract."""
    evidence_block = "\n".join(
        f"[{c.marker}] ({c.source}) {c.snippet}" for c in citations
    )
    return (
        "You are a healthcare information assistant. Answer the user's question "
        "using ONLY the numbered evidence passages below. Rules:\n"
        "1. Every factual sentence must end with the bracketed number(s) of the "
        "evidence it came from, e.g. [1] or [1][2].\n"
        "2. Do not add any medical fact, number, or recommendation that is not "
        "present in the evidence, even if you believe it is true.\n"
        "3. If the evidence is insufficient or conflicting, say so explicitly "
        "instead of filling the gap.\n"
        "4. Do not give personal dosing instructions or diagnose the user.\n"
        "5. Keep the answer concise (3-6 sentences).\n\n"
        f"EVIDENCE:\n{evidence_block}"
    )




class GroqGenerator:
    """Falls back to ExtractiveGenerator if no key / package / call fails."""
    

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        self.fallback = ExtractiveGenerator()

    def generate(self, query: str, ranked_chunks: List[RankedChunk]) -> GroundedAnswer:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or not ranked_chunks:
            return self.fallback.generate(query, ranked_chunks)

        citations = _make_citations(ranked_chunks)
        system_prompt = _build_grounding_prompt(citations)
        try:
            from groq import Groq

            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
            )
            text = resp.choices[0].message.content
            return GroundedAnswer(answer=text.strip(), citations=citations)
        except Exception:
            return self.fallback.generate(query, ranked_chunks)


def get_generator(backend: str = "extractive"):
    if backend == "extractive":
        return ExtractiveGenerator()
    if backend == "groq":
        return GroqGenerator()
    raise ValueError(f"Unknown generator backend: {backend}")


