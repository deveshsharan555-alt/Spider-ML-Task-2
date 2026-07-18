"""
Text preprocessing / chunking.


"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from src.ingest import Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return sentences


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    title: str | None
    url: str | None
    doc_type: str


def chunk_document(
    doc: Document,
    max_sentences: int = 5,
    overlap_sentences: int = 1,
) -> List[Chunk]:
    """Sentence-window chunking with overlap. Short docs -> a single chunk."""
    sentences = split_sentences(doc.text)
    if len(sentences) <= max_sentences:
        return [
            Chunk(
                chunk_id=f"{doc.id}_c0",
                doc_id=doc.id,
                text=doc.text,
                source=doc.source,
                title=doc.title,
                url=doc.url,
                doc_type=doc.doc_type,
            )
        ]

    chunks: List[Chunk] = []
    step = max(max_sentences - overlap_sentences, 1)
    idx = 0
    i = 0
    while idx < len(sentences):
        window = sentences[idx: idx + max_sentences]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.id}_c{i}",
                doc_id=doc.id,
                text=" ".join(window),
                source=doc.source,
                title=doc.title,
                url=doc.url,
                doc_type=doc.doc_type,
            )
        )
        idx += step
        i += 1
    return chunks


def chunk_documents(docs: List[Document], **kwargs) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, **kwargs))
    return chunks
