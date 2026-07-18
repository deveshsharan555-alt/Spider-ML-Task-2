"""
Streamlit user interface for the Healthcare Information Assistant.

"""
from __future__ import annotations

import streamlit as st

from src.pipeline import HealthcareRAGPipeline

st.set_page_config(page_title="Healthcare Information Assistant", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Building knowledge base (ingest -> chunk -> embed -> index)...")
def load_pipeline(embedding_backend: str, generator_backend: str):
    pipeline = HealthcareRAGPipeline(
        embedding_backend=embedding_backend,
        generator_backend=generator_backend,

    )
    pipeline.build()
    return pipeline


CONFIDENCE_COLORS = {
    "High": "🟢",
    "Medium": "🟡",
    "Low": "🟠",
    "Insufficient evidence": "🔴",
}

with st.sidebar:
    st.header("⚙️ Configuration")
    embedding_backend = st.selectbox(
        "Embedding backend", ["tfidf", "st"],
        help="tfidf = scikit-learn TF-IDF (no download). "
             "st = sentence-transformers dense embeddings (better semantic recall, "
             "requires `pip install sentence-transformers`).",
    )
    generator_backend = st.selectbox(
        "Generation backend", ["extractive", "groq"],
        help="extractive = zero-dependency, sentences copied verbatim from evidence "
             "(fully hallucination-proof by construction). "
             "groq = free alternative using Llama 3.3 70B (needs GROQ_API_KEY from console.groq.com). "
             "Switching this is instant — it never rebuilds the index.",
    )
    
    
    st.divider()
    st.caption(
        "This assistant retrieves evidence from MedQuAD Q&A pairs and WHO / CDC / NICE / "
        "nutrition / exercise guideline snippets, then generates an answer that is grounded "
        "in — and cites — that evidence only. It is not a substitute for professional "
        "medical advice."
    )

pipeline = load_pipeline(embedding_backend, generator_backend)

st.title("🩺 Healthcare Information Assistant")
st.caption(
    f"Knowledge base: {pipeline.num_documents} documents → {pipeline.num_chunks} chunks "
    f"(MedQuAD + WHO/CDC/NICE/nutrition/exercise guidelines)"
)

if "history" not in st.session_state:
    st.session_state.history = []

EXAMPLES = [
    "What lifestyle changes help hypertension?",
    "What is the DASH diet?",
    "What are the warning signs of a stroke?",
    "How much exercise do adults need for heart health?",
]
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state.pending_query = ex

query = st.chat_input("Ask a health question, e.g. 'What lifestyle changes help hypertension?'")
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    with st.spinner("Retrieving evidence and generating a grounded answer..."):
        response = pipeline.answer(query)
    st.session_state.history.append(response)

for response in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(response.query)
    with st.chat_message("assistant"):
        if response.safety_flag != "ok" and not response.citations and response.confidence is None:
            st.warning(response.answer)
        else:
            st.write(response.answer)

        if response.confidence:
            icon = CONFIDENCE_COLORS.get(response.confidence.label, "")
            st.markdown(
                f"**Confidence:** {icon} {response.confidence.label} "
                f"({response.confidence.score:.2f}) — {response.confidence.rationale}"
            )
            if response.conflict_detected:
                st.info("⚖️ Sources present differing numeric guidance on this topic — "
                         "see individual citations below.")

        if response.citations:
            with st.expander(f"📚 Sources ({len(response.citations)})", expanded=False):
                for c in response.citations:
                    title = f"**{c.title}** — " if c.title else ""
                    st.markdown(f"**[{c.marker}]** {title}*{c.source}*")
                    st.caption(c.snippet)
                    if c.url:
                        st.markdown(f"[{c.url}]({c.url})")
                    st.divider()

        if response.disclaimer:
            st.caption(response.disclaimer)
