"""
streamlit_app.py
----------------
MacroThread — Economic Research Assistant
Clean, minimal Streamlit UI with query input,
answer display, and source citations panel.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import os
from pathlib import Path

import streamlit as st

# Make sure imports from sibling directories work
sys.path.append(str(Path(__file__).parent.parent))

from generation.chain import ask
from retrieval.retriever import get_collection, load_all_chunks, build_bm25_index

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MacroThread",
    page_icon="🧵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Clean font and spacing */
    .main { max-width: 800px; }

    /* Header */
    .mt-header {
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 2rem;
    }
    .mt-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.5px;
    }
    .mt-subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }

    /* Answer box */
    .mt-answer {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        line-height: 1.7;
        color: #1f2937;
        font-size: 0.95rem;
        margin: 1.5rem 0;
    }

    /* Source card */
    .mt-source {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
    }
    .mt-source-label {
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.2rem;
    }
    .mt-source-meta {
        color: #6b7280;
        font-size: 0.8rem;
    }
    .mt-source-link {
        color: #2563eb;
        font-size: 0.8rem;
        text-decoration: none;
    }

    /* Badge */
    .mt-badge {
        display: inline-block;
        background: #f3f4f6;
        color: #374151;
        border-radius: 4px;
        padding: 0.1rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.3rem;
    }
    .mt-badge-both { background: #dcfce7; color: #166534; }
    .mt-badge-semantic { background: #dbeafe; color: #1e40af; }
    .mt-badge-bm25 { background: #fef9c3; color: #854d0e; }

    /* Footer */
    .mt-footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
        color: #9ca3af;
        font-size: 0.8rem;
        text-align: center;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load Resources (cached) ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():
    """
    Load ChromaDB collection and BM25 index once at startup.
    Streamlit caches this so it doesn't reload on every query.
    This is important — rebuilding the BM25 index on every query
    would add ~2 seconds of latency each time.
    """
    collection = get_collection()
    all_chunks = load_all_chunks()
    bm25_index, all_chunks = build_bm25_index(all_chunks)
    return collection, bm25_index, all_chunks


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="mt-header">
    <div class="mt-title">🧵 MacroThread</div>
    <div class="mt-subtitle">
        Ask research questions grounded in Federal Reserve speeches and FOMC minutes
    </div>
</div>
""", unsafe_allow_html=True)


# ── API Key Input ─────────────────────────────────────────────────────────────

# Check for key in environment first — if running locally with .env it's already set
api_key_from_env = os.getenv("OPENAI_API_KEY")

if not api_key_from_env:
    with st.expander("🔑 OpenAI API Key", expanded=True):
        api_key_input = st.text_input(
            "Enter your OpenAI API key",
            type="password",
            placeholder="sk-...",
            help="Your key is never stored. Get one at platform.openai.com"
        )
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
        else:
            st.info("Add your OpenAI API key to get started.")
            st.stop()


# ── Load Resources ────────────────────────────────────────────────────────────

try:
    collection, bm25_index, all_chunks = load_resources()
except Exception as e:
    st.error(f"Failed to load knowledge base: {e}")
    st.info("Make sure you have run embed.py first.")
    st.stop()

# Show corpus stats quietly
total_vectors = collection.count()
st.caption(f"{total_vectors:,} vectors across {len(all_chunks):,} chunks · Federal Reserve corpus")


# ── Query Input ───────────────────────────────────────────────────────────────

st.markdown("### Ask a research question")

# Example queries to help users get started
examples = [
    "How has the Fed described the relationship between inflation and employment?",
    "What did Powell say about interest rates in 2024?",
    "How has FOMC language around inflation changed since 2022?",
    "What are the Fed's views on quantitative tightening?",
]

with st.expander("💡 Example questions"):
    for example in examples:
        if st.button(example, key=example, use_container_width=True):
            st.session_state["query"] = example

query = st.text_area(
    label="Question",
    value=st.session_state.get("query", ""),
    placeholder="e.g. How has the Fed's language on inflation evolved since 2022?",
    height=90,
    label_visibility="collapsed"
)

# Filters in a clean collapsible section
with st.expander("🔍 Filters (optional)"):
    col1, col2 = st.columns(2)
    with col1:
        doc_type = st.selectbox(
            "Document type",
            options=["All", "speech", "minutes"],
            index=0
        )
        doc_type = None if doc_type == "All" else doc_type

    with col2:
        year_options = ["All"] + [str(y) for y in range(2026, 2021, -1)]
        year = st.selectbox("Year", options=year_options, index=0)
        year = None if year == "All" else year

    speaker = st.text_input(
        "Speaker (partial name)",
        placeholder="e.g. Powell, Waller, Bowman"
    )
    speaker = speaker if speaker.strip() else None

top_k = st.slider(
    "Number of sources to retrieve",
    min_value=3,
    max_value=10,
    value=5,
    help="More sources = more context for the LLM but slower response"
)

submit = st.button("Ask MacroThread", type="primary", use_container_width=True)


# ── Run Query ─────────────────────────────────────────────────────────────────

if submit and query.strip():
    with st.spinner("Searching knowledge base and generating answer..."):
        try:
            result = ask(
                query=query.strip(),
                top_k=top_k,
                doc_type=doc_type,
                year=year,
                speaker=speaker,
                collection=collection,
                bm25_index=bm25_index,
                all_chunks=all_chunks,
            )

            # ── Answer ────────────────────────────────────────────────────────
            st.markdown("### Answer")
            st.markdown(
                f'<div class="mt-answer">{result["answer"]}</div>',
                unsafe_allow_html=True
            )

            # Model + chunk info
            st.caption(
                f"Generated by {result['model']} · "
                f"{result['chunks_used']} chunks retrieved"
            )

            # ── Sources Panel ─────────────────────────────────────────────────
            if result["sources"]:
                st.markdown("### Sources")

                for src in result["sources"]:
                    # Retrieval method badge
                    method = src.get("method", "")
                    badge_class = f"mt-badge-{method}" if method else "mt-badge"
                    badge = f'<span class="mt-badge {badge_class}">{method or "retrieved"}</span>'

                    # Doc type badge
                    doc_badge = f'<span class="mt-badge">{src["doc_type"].upper()}</span>'

                    st.markdown(f"""
                    <div class="mt-source">
                        <div class="mt-source-label">
                            [Source {src['source_num']}] {src['speaker'][:60]}
                        </div>
                        <div class="mt-source-meta">
                            {doc_badge}
                            {badge}
                            {src['date']}
                        </div>
                        <div style="margin-top: 0.3rem;">
                            <a class="mt-source-link"
                               href="{src['source_url']}"
                               target="_blank">
                                View original document ↗
                            </a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.info("Check your API key and make sure embed.py has been run.")

elif submit and not query.strip():
    st.warning("Please enter a research question.")


# ── Empty State ───────────────────────────────────────────────────────────────

if not submit:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #9ca3af;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧵</div>
        <div>Ask a question to search the Federal Reserve corpus</div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="mt-footer">
    MacroThread · Built on public Federal Reserve data ·
    <a href="https://github.com/yourusername/macrothread"
       style="color: #9ca3af;">GitHub</a>
</div>
""", unsafe_allow_html=True)