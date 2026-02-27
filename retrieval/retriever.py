"""
retriever.py
------------
Hybrid retrieval combining semantic search (ChromaDB)
with BM25 keyword search, merged via Reciprocal Rank Fusion (RRF).

Why hybrid?
- Semantic search: understands meaning and concept
- BM25: catches exact financial terms like "quantitative tightening",
  "basis points", "federal funds rate" that embeddings sometimes miss
- RRF: smartly merges both ranked lists without score normalization

Usage:
    python retriever.py "How has the Fed described inflation since 2022?"
    python retriever.py "quantitative tightening balance sheet" --doc-type minutes
    python retriever.py "Powell employment" --year 2024 --show-context
    python retriever.py "rate decisions" --semantic-only
    python retriever.py "federal funds rate" --bm25-only
"""

import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

CHROMA_DIR      = Path(__file__).parent.parent / "data" / "chroma"
CHUNKS_PATH     = Path(__file__).parent.parent / "data" / "processed" / "chunks.jsonl"
COLLECTION_NAME = "macrothread"
EMBED_MODEL     = "text-embedding-3-small"

DEFAULT_TOP_K       = 5
SEMANTIC_CANDIDATES = 20   # how many to pull from each system before merging
BM25_CANDIDATES     = 20
RRF_K               = 60   # RRF constant — 60 is the standard recommended value


# ── Connect to ChromaDB ───────────────────────────────────────────────────────

def get_collection():
    """Connect to the existing ChromaDB collection."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DIR}\n"
            "Run: python embeddings/embed.py first"
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBED_MODEL
    )
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )


# ── BM25 Index ────────────────────────────────────────────────────────────────

def build_bm25_index(chunks: list[dict]) -> tuple:
    """
    Build a BM25 index from all chunks.

    BM25 (Best Match 25) is a classic keyword ranking algorithm.
    It tokenizes each chunk into words and builds an inverted index
    that scores documents based on term frequency and document length.

    We build this in memory at startup — for 5,245 chunks it takes
    under a second and requires no external service.

    Returns:
        bm25: the BM25 index
        chunk_list: ordered list of chunks matching the index
    """
    print("  Building BM25 index...", end=" ", flush=True)

    # Simple whitespace tokenization — good enough for economic text
    # lowercase everything so "Inflation" and "inflation" match
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    print(f"done ({len(chunks):,} chunks indexed)")
    return bm25, chunks


def load_all_chunks() -> list[dict]:
    """Load all chunks from the JSONL file for BM25 indexing."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_PATH}\n"
            "Run: python ingestion/chunker.py first"
        )

    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    return chunks


# ── Metadata Filter ───────────────────────────────────────────────────────────

def build_chroma_filter(
    doc_type:    str | None = None,
    year:        str | None = None,
    speaker:     str | None = None,
    institution: str | None = None,
    country:     str | None = None,
) -> dict | None:
    """Build a ChromaDB where filter from optional parameters."""
    conditions = []

    if doc_type:
        conditions.append({"doc_type": {"$eq": doc_type}})
    if year:
        conditions.append({"year": {"$eq": str(year)}})
    if speaker:
        conditions.append({"speaker": {"$contains": speaker}})
    if institution:
        conditions.append({"institution": {"$eq": institution}})
    if country:
        conditions.append({"country": {"$eq": country}})

    if not conditions:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def apply_bm25_filter(chunks: list[dict], **filters) -> list[dict]:
    """Filter the chunk list for BM25 to match metadata filters."""
    filtered = chunks

    if filters.get("doc_type"):
        filtered = [c for c in filtered if c.get("doc_type") == filters["doc_type"]]
    if filters.get("year"):
        filtered = [c for c in filtered if c.get("date", "")[:4] == str(filters["year"])]
    if filters.get("speaker"):
        filtered = [c for c in filtered
                    if filters["speaker"].lower() in c.get("speaker", "").lower()]
    if filters.get("institution"):
        filtered = [c for c in filtered if c.get("institution") == filters["institution"]]
    if filters.get("country"):
        filtered = [c for c in filtered if c.get("country") == filters["country"]]

    return filtered


# ── Semantic Search ───────────────────────────────────────────────────────────

def semantic_search(
    query: str,
    collection,
    top_k: int,
    where_filter: dict | None
) -> list[dict]:
    """
    Run semantic similarity search against ChromaDB.
    Returns chunks ranked by embedding cosine similarity.
    """
    query_params = {
        "query_texts": [query],
        "n_results":   top_k,
        "include":     ["documents", "metadatas", "distances"]
    }
    if where_filter:
        query_params["where"] = where_filter

    try:
        results = collection.query(**query_params)
    except Exception as e:
        print(f"  [error] Semantic search failed: {e}")
        return []

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text":        doc,
            "score":       round(1 - dist, 4),
            "date":        meta.get("date", ""),
            "speaker":     meta.get("speaker", ""),
            "doc_type":    meta.get("doc_type", ""),
            "title":       meta.get("title", ""),
            "source_url":  meta.get("source_url", ""),
            "institution": meta.get("institution", ""),
            "country":     meta.get("country", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "chunk_id":    meta.get("chunk_id", ""),
        })

    return chunks


# ── BM25 Search ───────────────────────────────────────────────────────────────

def bm25_search(
    query: str,
    bm25: BM25Okapi,
    all_chunks: list[dict],
    top_k: int,
    **filters
) -> list[dict]:
    """
    Run BM25 keyword search over the chunk corpus.
    Returns chunks ranked by BM25 score.

    BM25 tokenizes the query and scores each chunk based on
    how often query terms appear, weighted by rarity across
    the full corpus (IDF — inverse document frequency).
    """
    # Apply metadata filters to narrow the search space
    filtered_chunks = apply_bm25_filter(all_chunks, **filters)

    if not filtered_chunks:
        return []

    # Rebuild BM25 index on filtered subset if filters applied
    if len(filtered_chunks) < len(all_chunks):
        tokenized = [c["text"].lower().split() for c in filtered_chunks]
        search_bm25 = BM25Okapi(tokenized)
        search_chunks = filtered_chunks
    else:
        search_bm25 = bm25
        search_chunks = all_chunks

    # Score and rank
    tokenized_query = query.lower().split()
    scores = search_bm25.get_scores(tokenized_query)

    # Get top_k indices by score
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        chunk = search_chunks[idx].copy()
        chunk["bm25_score"] = float(scores[idx])
        chunk["score"] = float(scores[idx])  # will be replaced by RRF score
        results.append(chunk)

    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    semantic_results: list[dict],
    bm25_results:     list[dict],
    top_k:            int,
    k:                int = RRF_K
) -> list[dict]:
    """
    Merge semantic and BM25 results using Reciprocal Rank Fusion.

    RRF formula for each document:
        score = sum(1 / (k + rank)) for each ranked list

    Where rank is 1-based position in each list.
    k=60 is standard — it dampens the impact of very high ranks
    so the algorithm is robust to outliers.

    Why RRF over score normalization?
    Semantic scores (cosine similarity 0-1) and BM25 scores
    (unbounded, corpus-dependent) live on completely different scales.
    Normalizing them is error-prone. RRF sidesteps this entirely
    by only caring about *rank*, not *score value*.
    """
    # Build a dict of chunk_text → RRF score
    # We use text as the key since chunk IDs may differ between systems
    rrf_scores  = {}
    chunk_store = {}  # text → chunk dict

    # Score semantic results
    for rank, chunk in enumerate(semantic_results, 1):
        key = chunk["text"]
        rrf_scores[key]  = rrf_scores.get(key, 0) + 1 / (k + rank)
        chunk_store[key] = chunk

    # Score BM25 results — add to existing RRF score if already seen
    for rank, chunk in enumerate(bm25_results, 1):
        key = chunk["text"]
        rrf_scores[key]  = rrf_scores.get(key, 0) + 1 / (k + rank)
        if key not in chunk_store:
            chunk_store[key] = chunk

    # Sort by RRF score and return top_k
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        chunk = chunk_store[key].copy()
        chunk["score"] = round(rrf_scores[key], 6)
        chunk["retrieval_method"] = _get_retrieval_method(
            key, semantic_results, bm25_results
        )
        results.append(chunk)

    return results


def _get_retrieval_method(
    text: str,
    semantic_results: list[dict],
    bm25_results: list[dict]
) -> str:
    """Label how a chunk was retrieved — useful for debugging."""
    in_semantic = any(c["text"] == text for c in semantic_results)
    in_bm25     = any(c["text"] == text for c in bm25_results)

    if in_semantic and in_bm25:
        return "both"
    elif in_semantic:
        return "semantic"
    else:
        return "bm25"


# ── Main Retrieve Function ────────────────────────────────────────────────────

def retrieve(
    query:          str,
    top_k:          int         = DEFAULT_TOP_K,
    doc_type:       str | None  = None,
    year:           str | None  = None,
    speaker:        str | None  = None,
    institution:    str | None  = None,
    country:        str | None  = None,
    semantic_only:  bool        = False,
    bm25_only:      bool        = False,
    collection                  = None,
    bm25_index                  = None,
    all_chunks:     list | None = None,
) -> list[dict]:
    """
    Main retrieval function — call this from chain.py and the Streamlit UI.

    By default runs hybrid retrieval (semantic + BM25 merged via RRF).
    Use semantic_only or bm25_only flags to isolate each method for comparison.

    Args:
        query:         the user's research question
        top_k:         number of results to return
        doc_type:      filter to 'speech' or 'minutes'
        year:          filter to a specific year e.g. '2024'
        speaker:       partial speaker name filter e.g. 'Powell'
        institution:   filter to institution e.g. 'Federal Reserve'
        country:       filter to country e.g. 'US'
        semantic_only: use only embedding similarity
        bm25_only:     use only keyword search
        collection:    pass existing ChromaDB collection (avoids reconnecting)
        bm25_index:    pass existing BM25 index (avoids rebuilding)
        all_chunks:    pass existing chunk list (avoids reloading)

    Returns:
        list of chunk dicts with text, score, metadata, retrieval_method
    """
    filters = dict(
        doc_type=doc_type,
        year=year,
        speaker=speaker,
        institution=institution,
        country=country
    )

    # Load dependencies if not passed in
    if collection is None and not bm25_only:
        collection = get_collection()

    if (all_chunks is None or bm25_index is None) and not semantic_only:
        all_chunks = load_all_chunks()
        bm25_index, all_chunks = build_bm25_index(all_chunks)

    semantic_results = []
    bm25_results     = []

    # Semantic search
    if not bm25_only:
        where_filter = build_chroma_filter(**filters)
        semantic_results = semantic_search(
            query=query,
            collection=collection,
            top_k=SEMANTIC_CANDIDATES,
            where_filter=where_filter
        )

    # BM25 search
    if not semantic_only:
        bm25_results = bm25_search(
            query=query,
            bm25=bm25_index,
            all_chunks=all_chunks,
            top_k=BM25_CANDIDATES,
            **filters
        )

    # Return early if only one method
    if semantic_only:
        return semantic_results[:top_k]
    if bm25_only:
        return bm25_results[:top_k]

    # Hybrid: merge with RRF
    return reciprocal_rank_fusion(semantic_results, bm25_results, top_k)


# ── Formatting ────────────────────────────────────────────────────────────────

def format_results(chunks: list[dict], query: str) -> str:
    """Format retrieved chunks for terminal display."""
    if not chunks:
        return "No results found. Try a different query or remove filters."

    lines = []
    lines.append(f"\n── Results for: '{query}' ──────────────────────────")
    lines.append(f"   Found {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks, 1):
        method = chunk.get("retrieval_method", "")
        method_label = f" [{method}]" if method else ""
        lines.append(f"  [{i}] Score: {chunk['score']:.4f}{method_label}")
        lines.append(f"      Date:    {chunk['date']}")
        lines.append(f"      Type:    {chunk['doc_type']}")
        lines.append(f"      Speaker: {chunk['speaker'][:60]}")
        lines.append(f"      Source:  {chunk['source_url']}")
        lines.append(f"      ─────────────────────────────────────────────")
        lines.append(f"      {chunk['text'][:400]}...")
        lines.append("")

    return "\n".join(lines)


def format_context_for_llm(chunks: list[dict]) -> str:
    """
    Format chunks into a labelled context block for the LLM.
    Each chunk is clearly sourced so the LLM can cite it.
    This is what gets injected as {context} in your generation prompt.
    """
    if not chunks:
        return "No relevant context found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        label = (
            f"[Source {i}] "
            f"{chunk['doc_type'].upper()} | "
            f"{chunk['date']} | "
            f"{chunk['speaker'][:50]}"
        )
        parts.append(f"{label}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Query the MacroThread knowledge base (hybrid BM25 + semantic)"
    )
    parser.add_argument("query", type=str, help="Your research question")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--doc-type", type=str, choices=["speech", "minutes"])
    parser.add_argument("--year", type=str, help="Filter by year e.g. 2024")
    parser.add_argument("--speaker", type=str, help="Partial speaker name")
    parser.add_argument("--semantic-only", action="store_true")
    parser.add_argument("--bm25-only", action="store_true")
    parser.add_argument("--show-context", action="store_true",
                        help="Show the formatted LLM context block")
    args = parser.parse_args()

    print("\nMacroThread Hybrid Retriever")

    # Load everything upfront
    collection = None
    bm25_index = None
    all_chunks = None

    if not args.bm25_only:
        collection = get_collection()
        print(f"  ChromaDB: {collection.count():,} vectors")

    if not args.semantic_only:
        all_chunks = load_all_chunks()
        bm25_index, all_chunks = build_bm25_index(all_chunks)

    chunks = retrieve(
        query=args.query,
        top_k=args.top_k,
        doc_type=args.doc_type,
        year=args.year,
        speaker=args.speaker,
        semantic_only=args.semantic_only,
        bm25_only=args.bm25_only,
        collection=collection,
        bm25_index=bm25_index,
        all_chunks=all_chunks,
    )

    print(format_results(chunks, args.query))

    if args.show_context:
        print("\n── LLM Context Block ───────────────────────────────")
        print(format_context_for_llm(chunks))


if __name__ == "__main__":
    main()