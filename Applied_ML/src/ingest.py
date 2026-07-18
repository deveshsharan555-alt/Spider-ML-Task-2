"""
Document ingestion pipeline.

"""
from __future__ import annotations

import csv
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class Document:
    """A single, source-attributed piece of evidence in the knowledge base."""
    id: str
    text: str
    source: str                 # e.g. "MedQuAD (NHLBI)", "WHO Hypertension Guidelines"
    title: Optional[str] = None
    url: Optional[str] = None
    doc_type: str = "qa"        # "qa" | "guideline"
    metadata: dict = field(default_factory=dict)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def load_medquad_csv(path: str) -> List[Document]:
    """Load a MedQuAD-style CSV with columns: question, answer, source, url."""
    docs: List[Document] = []
    p = Path(path)
    if not p.exists():
        return docs
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = _clean(row.get("question", ""))
            answer = _clean(row.get("answer", ""))
            if not answer:
                continue
            source = _clean(row.get("source", "MedQuAD"))
            url = row.get("url", "") or None
            # Store question+answer together so retrieval can match on either,
            # but keep the answer as the primary evidence text.
            text = f"Q: {question}\nA: {answer}" if question else answer
            docs.append(
                Document(
                    id=f"medquad_{uuid.uuid4().hex[:10]}",
                    text=text,
                    source=source,
                    title=question or None,
                    url=url,
                    doc_type="qa",
                    metadata={"question": question, "answer": answer},
                )
            )
    return docs


def load_guidelines_json(path: str) -> List[Document]:
    """Load WHO / CDC / NICE / nutrition / exercise style guideline snippets."""
    docs: List[Document] = []
    p = Path(path)
    if not p.exists():
        return docs
    items = json.loads(p.read_text(encoding="utf-8"))
    for item in items:
        docs.append(
            Document(
                id=item.get("id") or f"guideline_{uuid.uuid4().hex[:10]}",
                text=_clean(item["text"]),
                source=item.get("source", "Guideline"),
                title=item.get("title"),
                url=item.get("url"),
                doc_type="guideline",
                metadata={},
            )
        )
    return docs


def build_knowledge_base(
    medquad_paths: Iterable[str] = (),
    guideline_paths: Iterable[str] = (),
) -> List[Document]:
    """Merge every configured source into one knowledge base (list of Documents)."""
    docs: List[Document] = []
    for path in medquad_paths:
        docs.extend(load_medquad_csv(path))
    for path in guideline_paths:
        docs.extend(load_guidelines_json(path))
    return docs


DEFAULT_SOURCES = {
    "medquad_paths": ["data/raw/sample_medquad.csv", "data/raw/medquad.csv"],
    "guideline_paths": ["data/raw/guidelines.json"],
}


def load_default_knowledge_base() -> List[Document]:
    """Convenience loader used by the pipeline/app: loads every file that exists."""
    return build_knowledge_base(**DEFAULT_SOURCES)


if __name__ == "__main__":
    kb = load_default_knowledge_base()
    print(f"Loaded {len(kb)} documents")
    by_source = {}
    for d in kb:
        by_source[d.source] = by_source.get(d.source, 0) + 1
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count}")
