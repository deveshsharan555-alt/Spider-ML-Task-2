"""
Healthcare Information Assistant — core logic.

"""

import os
import re
import csv
import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import difflib
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------
# Settings 
# ---------------------------------------------------------------
EMBEDDING = "st"     
GENERATOR = "groq"   
MEDQUAD_FILES = ["data/raw/sample_medquad.csv", "data/raw/medquad.csv"]
GUIDELINE_FILES = ["data/raw/guidelines.json"]
MIN_RELEVANCE = 0.15  
CLARIFY_MESSAGE = (
    "Could you say a bit more about what you'd like to know? For example, a "
    "specific condition (e.g. 'hypertension', 'type 2 diabetes'), a symptom, "
    "or a topic like diet, exercise, or medication."
)
NO_EVIDENCE_MESSAGE = (
    "I couldn't find reliable evidence in the knowledge base to answer this "
    "question. Please consult a healthcare provider or trusted medical "
    "source, or try rephrasing your question."
)


# ---------------------------------------------------------------
# 1. Load documents and split them into short chunks
# ---------------------------------------------------------------

def load_documents():
    docs = []
    for path in MEDQUAD_FILES:
        p = Path(path)
        if not p.exists():
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                question = row.get("question", "").strip()
                answer = row.get("answer", "").strip()
                if not answer:
                    continue
                text = f"Q: {question}\nA: {answer}" if question else answer
                docs.append({
                    "text": text,
                    "source": row.get("source", "MedQuAD"),
                    "title": question or None,
                    "url": row.get("url") or None,
                })
    for path in GUIDELINE_FILES:
        p = Path(path)
        if not p.exists():
            continue
        for item in json.loads(p.read_text(encoding="utf-8")):
            docs.append({
                "text": item["text"],
                "source": item.get("source", "Guideline"),
                "title": item.get("title"),
                "url": item.get("url"),
            })
    return docs


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_documents(docs, max_sentences=5):
    
    chunks = []
    for doc in docs:
        sentences = split_sentences(doc["text"])
        if len(sentences) <= max_sentences:
            chunks.append(doc)
            continue
        step = max_sentences - 1
        for i in range(0, len(sentences), step):
            window = sentences[i:i + max_sentences]
            chunks.append({**doc, "text": " ".join(window)})
    return chunks


# ---------------------------------------------------------------
# 2. Turn text into vectors
# ---------------------------------------------------------------

_st_model = None  # loaded once and reused


def _sentence_transformer_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def embed_texts(texts):
    
    model = _sentence_transformer_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text):
    vectors = embed_texts([text])
    return vectors[0]


# ---------------------------------------------------------------
# 3. Build the index
# ---------------------------------------------------------------

def build_index():
    docs = load_documents()

    if not docs:
        raise RuntimeError("No documents found. Check the data/raw/ folder.")

    chunks = chunk_documents(docs)

    vectors = embed_texts([c["text"] for c in chunks])

    return {
        "chunks": chunks,
        "vectors": vectors,
        "num_documents": len(docs),
    }


# ---------------------------------------------------------------
# 4. Retrieve the most relevant chunks, then rerank them
# ---------------------------------------------------------------

WORD = re.compile(r"[a-zA-Z]+")


def lexical_overlap(query, text):
    q_words = set(w.lower() for w in WORD.findall(query))
    t_words = set(w.lower() for w in WORD.findall(text))
    return len(q_words & t_words) / len(q_words) if q_words else 0.0


def retrieve_and_rerank(query, index, top_n=15, top_k=5):
    query_vector = embed_query(query)
    similarities = cosine_similarity(query_vector.reshape(1, -1), index["vectors"])[0]
    top_indexes = np.argsort(-similarities)[:top_n]

    # Blend cosine similarity with plain keyword overlap — rewards passages
    # that actually contain the medical terms asked about.
    candidates = []
    for i in top_indexes:
        chunk = index["chunks"][i]
        score = 0.7 * float(similarities[i]) + 0.3 * lexical_overlap(query, chunk["text"])
        candidates.append((chunk, score))
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    
    # Prefer a mix of sources once relevance is close, so the answer can show
    # MedQuAD *and* WHO *and* NICE agreeing instead of citing one source 5x.
    selected, seen_sources = [], set()
    remaining = list(candidates)
    while remaining and len(selected) < top_k:
        best_i, best_value = 0, -1
        for i, (chunk, score) in enumerate(remaining):
            bonus = 0.05 if chunk["source"] not in seen_sources else 0.0
            if score + bonus > best_value:
                best_value, best_i = score + bonus, i
        chunk, score = remaining.pop(best_i)
        selected.append((chunk, score))
        seen_sources.add(chunk["source"])
    return selected


# ---------------------------------------------------------------
# 5. Confidence estimation
# ---------------------------------------------------------------

def estimate_confidence(ranked):
    relevant = [(c, s) for c, s in ranked if s >= MIN_RELEVANCE]
    if not relevant:
        return {
            "label": "Insufficient evidence",
            "score": 0.0,
            "rationale": "No retrieved passage cleared the minimum relevance threshold.",
        }

    scores = [s for _, s in relevant]
    top_score = scores[0]
    margin = top_score - (scores[1] if len(scores) > 1 else 0.0)
    sources = {c["source"] for c, _ in relevant}
    coverage = min(len(relevant) / 3, 1.0)
    agreement_bonus = min((len(sources) - 1) * 0.15, 0.3)

    raw = 0.5 * top_score + 0.15 * min(margin * 2, 1.0) + 0.2 * coverage + agreement_bonus
    raw = max(0.0, min(raw, 1.0))
    label = "High" if raw >= 0.66 else "Medium" if raw >= 0.4 else "Low"
    agreement_text = "independent sources agree" if len(sources) > 1 else "only one source found"

    return {
        "label": label,
        "score": round(raw, 2),
        "rationale": (
            f"top match relevance {top_score:.2f}; {len(relevant)} supporting passage(s); "
            f"from {len(sources)} distinct source(s); {agreement_text}"
        ),
    }


# ---------------------------------------------------------------
# 6. Safety checks
# ---------------------------------------------------------------

EMERGENCY_PATTERNS = [
    r"\bchest pain\b", r"\bcan'?t breathe\b", r"\bnot breathing\b", r"\bsevere bleeding\b",
    r"\bface (is )?drooping\b", r"\bslurred speech\b", r"\bunconscious\b",
    r"\boverdos(e|ed|ing)\b", r"\bsevere allergic reaction\b", r"\bpassed out\b",
]
SELF_HARM_PATTERNS = [
    r"\bkill myself\b", r"\bsuicid", r"\bend my life\b", r"\bself[- ]harm\b", r"\bwant to die\b",
]
UNSAFE_PATTERNS = [
    r"\bhow many (pills|mg|milligrams)\b", r"\bwhat dose\b", r"\bwhat dosage\b",
    r"\bhow much .*(should i take|can i take)\b",
    r"\bdo i have (cancer|diabetes|a tumor|hiv)\b", r"\bam i having a heart attack\b",
    r"\bcan i (stop|skip|double) (taking )?my medication\b",
]
SAFETY_MESSAGES = {
    "emergency": (
        "⚠️ What you're describing may be a medical emergency. This assistant cannot "
        "provide emergency care. Please call your local emergency number (e.g. 911 in "
        "the US, 999 in the UK, 112 in the EU,112 in India) or go to the nearest emergency department "
        "right now. If you can, ask someone nearby to stay with you until help arrives."
    ),
    "self_harm": (
        "I'm concerned about what you shared. You deserve immediate support from someone "
        "who can help right now. If you are in the US, you can call or text 988 (Suicide "
        "& Crisis Lifeline), available 24/7. If you are elsewhere, please contact your "
        "local emergency number or a crisis line in your country."
    ),
    "unsafe": (
        "I can't provide personal dosing instructions, diagnose a condition, or tell you "
        "whether to change a prescribed medication — that requires a clinician who knows "
        "your medical history. I can share general, source-cited information about a "
        "condition or medication class instead. Please consult a doctor or pharmacist for "
        "anything specific to your situation."
    ),
}


def check_safety(query):
    if any(re.search(p, query, re.I) for p in SELF_HARM_PATTERNS):
        return "self_harm", SAFETY_MESSAGES["self_harm"]
    if any(re.search(p, query, re.I) for p in EMERGENCY_PATTERNS):
        return "emergency", SAFETY_MESSAGES["emergency"]
    if any(re.search(p, query, re.I) for p in UNSAFE_PATTERNS):
        return "unsafe", SAFETY_MESSAGES["unsafe"]
    return "ok", None


def is_vague(query):
    return len(query.strip().split()) <= 1

# ---------------------------------------------------------------
# Handling misspellings
# ---------------------------------------------------------------
 
 
def build_vocabulary(chunks, min_length=4, min_count=2):
    
    counts = {}
    for chunk in chunks:
        for word in WORD.findall(chunk["text"].lower()):
            if len(word) < min_length or word in ENGLISH_STOP_WORDS:
                continue
            counts[word] = counts.get(word, 0) + 1
    return {word for word, count in counts.items() if count >= min_count}
 
 
def correct_word(word, vocabulary):
    lower = word.lower()
    if lower in vocabulary or lower in ENGLISH_STOP_WORDS or len(lower) < 4:
        return word
    matches = difflib.get_close_matches(lower, vocabulary, n=1, cutoff=0.84)
    return matches[0] if matches else word
 
 
def normalize_query(query, vocabulary):
    fixed_words = []
    for word in query.split():
        core = word.strip("?.,!")
        corrected = correct_word(core, vocabulary)
        fixed_words.append(corrected if corrected != core else word)
    return " ".join(fixed_words)


# ---------------------------------------------------------------
# 7. Generate the final answer with citations
# ---------------------------------------------------------------

def make_citations(ranked):
    citations = []
    for i, (chunk, _) in enumerate(ranked, start=1):
        snippet = chunk["text"]
        if len(snippet) > 220:
            snippet = snippet[:217].rsplit(" ", 1)[0] + "..."
        citations.append({
            "marker": i, "source": chunk["source"],
            "title": chunk["title"], "url": chunk["url"], "snippet": snippet,
        })
    return citations



def groq_answer(query, ranked):
    
    citations = make_citations(ranked)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found.")

    evidence = "\n".join(f"[{c['marker']}] ({c['source']}) {c['snippet']}" for c in citations)
    prompt = (
        "You are a healthcare information assistant. Answer using ONLY the evidence "
        "below. Cite every sentence with its bracketed number(s). Don't add facts not "
        "in the evidence. If the evidence is insufficient or conflicting, say so. "
        "Don't give dosing advice or diagnose. Keep it to 3-6 sentences.\n\n"
        f"EVIDENCE:\n{evidence}"
    )
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
        )
        return response.choices[0].message.content.strip(), citations
    except Exception as e:
        raise RuntimeError(f"Groq request failed : {e}")


# ---------------------------------------------------------------
# 8. Put it all together
# ---------------------------------------------------------------

class Assistant:
    def build(self):
        self.EMBEDDING = EMBEDDING
        self.GENERATOR = GENERATOR

        self.index = build_index()
        self.num_documents = self.index["num_documents"]
        self.num_chunks = len(self.index["chunks"])
        self.vocabulary = build_vocabulary(self.index["chunks"])

    def ask(self, query):
        flag, message = check_safety(query)
        if flag != "ok":
            return {"safety_flag": flag, "answer": message, "confidence": None, "citations": []}

        if is_vague(query):
            return {"safety_flag": "ok", "answer": CLARIFY_MESSAGE, "confidence": None, "citations": []}
        
        
        query = normalize_query(query, self.vocabulary)

        ranked = retrieve_and_rerank(query, self.index)
        confidence = estimate_confidence(ranked)
        if confidence["label"] == "Insufficient evidence":
            return {"safety_flag": "ok", "answer": NO_EVIDENCE_MESSAGE,
                    "confidence": confidence, "citations": []}

        relevant = [(c, s) for c, s in ranked if s >= MIN_RELEVANCE] or ranked[:1]
        answer, citations = groq_answer(query, relevant)
        return {"safety_flag": "ok", "answer": answer, "confidence": confidence, "citations": citations}
