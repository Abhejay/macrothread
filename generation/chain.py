"""
chain.py
--------
The generation layer of MacroThread.
Takes a user query, retrieves relevant chunks, and passes
them to GPT-4o-mini to generate a grounded answer with citations.

This is the file that makes MacroThread actually answer questions.

Usage:
    python chain.py "How has the Fed described inflation since 2022?"
    python chain.py "What did Powell say about employment in 2024?" --doc-type speech
    python chain.py "Rate decisions in 2023" --year 2023 --doc-type minutes
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from retrieval.retriever import (
    retrieve,
    format_context_for_llm,
    get_collection,
    load_all_chunks,
    build_bm25_index,
)

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

DEFAULT_MODEL   = "gpt-4o-mini"
DEFAULT_TOP_K   = 5
MAX_TOKENS      = 1024


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MacroThread, an economic research assistant with deep knowledge \
of central bank communications, monetary policy, and macroeconomic analysis.

You answer questions strictly based on the source documents provided in the context.
Your goal is to help researchers, analysts, and economists understand how central banks \
think and communicate about the economy.

RULES:
1. Base your answer ONLY on the provided context. Do not use outside knowledge.
2. Always cite your sources using the [Source N] labels from the context.
3. If the context does not contain enough information to answer, say so clearly.
4. Be precise with dates, figures, and attributions.
5. Write in clear, professional prose suitable for economic research.
6. Never speculate or make predictions — only report what the sources say.

CITATION FORMAT:
When referencing a source, use inline citations like this:
"The Fed noted that inflation remained elevated [Source 1], while employment \
conditions stayed robust [Source 2]."

Always end your answer with a "Sources" section listing each cited source with:
- Source number
- Document type (SPEECH or MINUTES)
- Date
- Speaker/Institution
- URL
"""


# ── Generation ────────────────────────────────────────────────────────────────

def build_user_prompt(query: str, context: str) -> str:
    """
    Build the user message that gets sent to the LLM.
    Combines the research question with the retrieved context.
    """
    return f"""RESEARCH QUESTION:
{query}

CONTEXT FROM FEDERAL RESERVE DOCUMENTS:
{context}

Please answer the research question based on the context above.
Cite sources inline using [Source N] notation and list them at the end."""


def generate_answer(
    query:   str,
    chunks:  list[dict],
    model:   str = DEFAULT_MODEL,
    client:  OpenAI = None,
) -> dict:
    """
    Generate a grounded answer from retrieved chunks.

    Args:
        query:   the user's research question
        chunks:  retrieved chunks from retriever.py
        model:   OpenAI model to use
        client:  OpenAI client (created if not passed)

    Returns:
        dict with keys:
            answer:    the LLM's response text
            sources:   list of source metadata dicts
            query:     original query
            model:     model used
            chunks_used: number of chunks passed to LLM
    """
    if not chunks:
        return {
            "answer": "No relevant documents found in the knowledge base for this query. "
                      "Try rephrasing or removing filters.",
            "sources": [],
            "query": query,
            "model": model,
            "chunks_used": 0
        }

    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        client = OpenAI(api_key=api_key)

    # Format context for the LLM
    context = format_context_for_llm(chunks)
    user_prompt = build_user_prompt(query, context)

    # Call the LLM
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.1,  # low temperature = more factual, less creative
    )

    answer = response.choices[0].message.content

    # Build sources list from chunks
    sources = []
    for i, chunk in enumerate(chunks, 1):
        sources.append({
            "source_num":  i,
            "doc_type":    chunk.get("doc_type", ""),
            "date":        chunk.get("date", ""),
            "speaker":     chunk.get("speaker", ""),
            "institution": chunk.get("institution", ""),
            "source_url":  chunk.get("source_url", ""),
            "score":       chunk.get("score", 0),
            "method":      chunk.get("retrieval_method", ""),
        })

    return {
        "answer":      answer,
        "sources":     sources,
        "query":       query,
        "model":       model,
        "chunks_used": len(chunks),
    }


# ── Full RAG Pipeline ─────────────────────────────────────────────────────────

def ask(
    query:         str,
    top_k:         int        = DEFAULT_TOP_K,
    doc_type:      str | None = None,
    year:          str | None = None,
    speaker:       str | None = None,
    model:         str        = DEFAULT_MODEL,
    collection                = None,
    bm25_index                = None,
    all_chunks:    list | None = None,
) -> dict:
    """
    The main end-to-end RAG function.
    Retrieves relevant chunks then generates a grounded answer.

    This is the single function your Streamlit UI will call.

    Args:
        query:      the user's research question
        top_k:      number of chunks to retrieve
        doc_type:   optional filter — 'speech' or 'minutes'
        year:       optional year filter e.g. '2024'
        speaker:    optional partial speaker name filter
        model:      LLM model to use
        collection: pass existing ChromaDB collection
        bm25_index: pass existing BM25 index
        all_chunks: pass existing chunk list

    Returns:
        dict with answer, sources, query, model, chunks_used
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(
        query=query,
        top_k=top_k,
        doc_type=doc_type,
        year=year,
        speaker=speaker,
        collection=collection,
        bm25_index=bm25_index,
        all_chunks=all_chunks,
    )

    # Step 2: Generate grounded answer
    result = generate_answer(query=query, chunks=chunks, model=model)

    return result


# ── Formatting ────────────────────────────────────────────────────────────────

def format_response(result: dict) -> str:
    """Format the full response for terminal display."""
    lines = []

    lines.append(f"\n{'─' * 60}")
    lines.append(f"QUERY: {result['query']}")
    lines.append(f"Model: {result['model']} | Chunks used: {result['chunks_used']}")
    lines.append(f"{'─' * 60}\n")

    lines.append(result["answer"])

    if result["sources"]:
        lines.append(f"\n{'─' * 60}")
        lines.append("RETRIEVED SOURCES")
        lines.append(f"{'─' * 60}")
        for src in result["sources"]:
            lines.append(
                f"  [Source {src['source_num']}] "
                f"{src['doc_type'].upper()} | "
                f"{src['date']} | "
                f"{src['speaker'][:50]}"
            )
            lines.append(f"  Score: {src['score']:.4f} | Method: {src['method']}")
            lines.append(f"  {src['source_url']}")
            lines.append("")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ask MacroThread a research question"
    )
    parser.add_argument("query", type=str, help="Your research question")
    parser.add_argument("--top-k",    type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--doc-type", type=str, choices=["speech", "minutes"])
    parser.add_argument("--year",     type=str, help="Filter by year e.g. 2024")
    parser.add_argument("--speaker",  type=str, help="Partial speaker name")
    parser.add_argument("--model",    type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    print("\nMacroThread — Economic Research Assistant")
    print("Loading knowledge base...\n")

    # Load everything once upfront
    collection = get_collection()
    all_chunks = load_all_chunks()
    bm25_index, all_chunks = build_bm25_index(all_chunks)

    print(f"  ChromaDB: {collection.count():,} vectors ready")
    print(f"  BM25: {len(all_chunks):,} chunks indexed")
    print(f"  Model: {args.model}\n")

    result = ask(
        query=args.query,
        top_k=args.top_k,
        doc_type=args.doc_type,
        year=args.year,
        speaker=args.speaker,
        model=args.model,
        collection=collection,
        bm25_index=bm25_index,
        all_chunks=all_chunks,
    )

    print(format_response(result))


if __name__ == "__main__":
    main()