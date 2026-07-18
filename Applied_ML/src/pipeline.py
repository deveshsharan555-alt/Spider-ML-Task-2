"""
End-to-end pipeline 
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.ingest import load_default_knowledge_base, Document
from src.chunking import chunk_documents, Chunk
from src.embeddings import get_embedder, Embedder
from src.vector_store import get_vector_store
from src.reranker import rerank, RankedChunk
from src.confidence import estimate_confidence, detect_conflict, ConfidenceResult, MIN_RELEVANCE_THRESHOLD
from src.generation import get_generator, GroundedAnswer
from src.safety import check_safety, SafetyFlag, GENERAL_DISCLAIMER

_COMMON_MISSPELLINGS = {
    "hipertension": "hypertension",
    "hypertention": "hypertension",
    "diabetis": "diabetes",
    "diabeties": "diabetes",
    "cholestrol": "cholesterol",
    "cholestral": "cholesterol",
    "hart": "heart",
    "strok": "stroke",
    "presure": "pressure",
    "presure.": "pressure.",
}


def normalize_query(query: str) -> str:
    """Light, dependency-free spelling normalization for common medical terms.
    (Swap in a real spellchecker / SymSpell if higher recall is needed.)"""
    words = query.split()
    fixed = [_COMMON_MISSPELLINGS.get(w.lower().strip("?.,"), w) for w in words]
    return " ".join(fixed)


VAGUE_QUERY_PATTERNS = [
    r"^\s*(help|hi|hello|health|sick|not feeling well)\s*\??\s*$",
]


def is_vague(query: str) -> bool:
    q = query.strip()
    if len(q.split()) <= 2 and not any(ch.isdigit() for ch in q):
        return any(re.match(p, q, flags=re.IGNORECASE) for p in VAGUE_QUERY_PATTERNS) or len(q.split()) == 1
    return False


CLARIFICATION_MESSAGE = (
    "Could you say a bit more about what you'd like to know? For example, a "
    "specific condition (e.g. 'hypertension', 'type 2 diabetes'), a symptom, "
    "or a topic like diet, exercise, or medication."
)


@dataclass
class AssistantResponse:
    query: str
    safety_flag: str
    answer: str
    confidence: Optional[ConfidenceResult] = None
    citations: list = field(default_factory=list)
    conflict_detected: bool = False
    disclaimer: str = GENERAL_DISCLAIMER
    retrieved_count: int = 0


class HealthcareRAGPipeline:
    def __init__(
        self,
        embedding_backend: str = "tfidf",
        vector_store_backend: str = "numpy",
        generator_backend: str = "extractive",
        top_n_retrieve: int = 15,
        top_k_final: int = 5,
    ):
        self.embedder: Embedder = get_embedder(embedding_backend)
        self.vector_store = get_vector_store(vector_store_backend)
        self.generator = get_generator(generator_backend)
        self.top_n_retrieve = top_n_retrieve
        self.top_k_final = top_k_final
        self._built = False

    def build(self, documents: Optional[List[Document]] = None) -> None:
        documents = documents if documents is not None else load_default_knowledge_base()
        if not documents:
            raise RuntimeError(
                "No documents loaded. Check data/raw/ for sample_medquad.csv / "
                "medquad.csv / guidelines.json."
            )
        chunks: List[Chunk] = chunk_documents(documents)
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.transform(texts)
        self.vector_store.build(chunks, vectors)
        self._built = True
        self.num_documents = len(documents)
        self.num_chunks = len(chunks)

    def retrieve(self, query: str) -> List[tuple]:
        if not self._built:
            raise RuntimeError("Call .build() before querying the pipeline.")
        query_vec = self.embedder.embed_query(query)
        return self.vector_store.search(query_vec, top_k=self.top_n_retrieve)

    def answer(self, query: str) -> AssistantResponse:
        raw_query = query
        safety = check_safety(raw_query)
        if safety.flag != SafetyFlag.OK:
            return AssistantResponse(
                query=raw_query,
                safety_flag=safety.flag.value,
                answer=safety.message or "",
                confidence=None,
                citations=[],
                disclaimer="",
            )

        if is_vague(raw_query):
            return AssistantResponse(
                query=raw_query,
                safety_flag=SafetyFlag.OK.value,
                answer=CLARIFICATION_MESSAGE,
                confidence=None,
                citations=[],
                disclaimer="",
            )

        query = normalize_query(raw_query)
        candidates = self.retrieve(query)
        ranked: List[RankedChunk] = rerank(
            query,
            candidates,
            top_k=self.top_k_final,
        )
        confidence = estimate_confidence(ranked)
        conflict = detect_conflict(ranked)

        if confidence.label == "Insufficient evidence":
            return AssistantResponse(
                query=raw_query,
                safety_flag=SafetyFlag.OK.value,
                answer=(
                    "I couldn't find reliable evidence in the knowledge base to answer "
                    "this question. Rather than guess, I'll say so directly — please "
                    "consult a healthcare provider or trusted medical source, or try "
                    "rephrasing your question."
                ),
                confidence=confidence,
                citations=[],
                conflict_detected=False,
                retrieved_count=len(candidates),
            )

        # Only feed genuinely relevant passages to the generator — a chunk that
        # scraped into the top-k purely for source diversity but is barely
        # relevant should not get woven into the answer as if it were evidence.
        relevant_ranked = [rc for rc in ranked if rc.rerank_score >= MIN_RELEVANCE_THRESHOLD] or ranked[:1]
        grounded: GroundedAnswer = self.generator.generate(query, relevant_ranked)

        answer_text = grounded.answer
        if conflict:
            answer_text += (
                "\n\nNote: sources in the evidence differ on specific numeric "
                "thresholds for this topic — see citations below for each source's "
                "own figures rather than treating one as universally correct."
            )

        return AssistantResponse(
            query=raw_query,
            safety_flag=SafetyFlag.OK.value,
            answer=answer_text,
            confidence=confidence,
            citations=grounded.citations,
            conflict_detected=conflict,
            disclaimer=grounded.disclaimer,
            retrieved_count=len(candidates),
        )
