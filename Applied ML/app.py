"""Streamlit web app for the Healthcare Information Assistant."""

import streamlit as st
from rag_core import Assistant

st.set_page_config(page_title="Healthcare Information Assistant", page_icon="🩺")


@st.cache_resource(show_spinner="Building knowledge base...")
def load_bot():
    bot = Assistant()
    bot.build()
    return bot


bot = load_bot()

with st.sidebar:
    st.header("⚙️ Configuration")

    st.markdown("### Current Settings")
    st.write(f"**Embedding:** {bot.EMBEDDING}")
    st.write(f"**Generator:** {bot.GENERATOR}")

    st.divider()

    st.caption(
        "This assistant retrieves evidence from MedQuAD and trusted medical "
        "guidelines, then answers only using the retrieved evidence. "
        "It is not a substitute for professional medical advice."
    )

st.title("🩺 Healthcare Information Assistant")

st.caption(
    f"Knowledge base: {bot.num_documents} documents → {bot.num_chunks} chunks "
    "(MedQuAD + WHO/CDC/NICE guidelines)"
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

for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state.pending_query = example

query = st.chat_input(
    "Ask a health question, e.g. 'What lifestyle changes help hypertension?'"
)

if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")


if query:
    with st.spinner("Retrieving evidence and generating a grounded answer..."):
        result = bot.ask(query)

    result["query"] = query
    st.session_state.history.append(result)

CONFIDENCE_ICONS = {"High": "🟢", "Medium": "🟡", "Low": "🟠", "Insufficient evidence": "🔴"}

for r in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(r["query"])
    with st.chat_message("assistant"):
        st.write(r["answer"])

        if r["confidence"]:
            c = r["confidence"]
            icon = CONFIDENCE_ICONS.get(c["label"], "")
            st.markdown(f"**Confidence:** {icon} {c['label']} ({c['score']}) — {c['rationale']}")

        if r["citations"]:
            with st.expander(f"📚 Sources ({len(r['citations'])})", expanded=False):
                for c in r["citations"]:
                    st.markdown(f"**[{c['marker']}]** *{c['source']}*")
                    st.caption(c["snippet"])

                    if c["url"]:
                        st.markdown(f"[{c['url']}]({c['url']})")

                    st.divider()
