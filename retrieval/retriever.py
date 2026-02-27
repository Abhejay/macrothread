import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

CHROMA_DIR      = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION_NAME = "macrothread"
EMBED_MODEL     = "text-embedding-3-small"
DEFAULT_TOP_K   = 5

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

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )

    return collection

def build_filter(
    doc_type:    str | None = None,
    year:        str | None = None,
    speaker:     str | None = None,
    institution: str | None = None,
    country:     str | None = None,
) -> dict | None:
    """
    Build a ChromaDB metadata filter from optional parameters.

    ChromaDB uses a MongoDB-style filter syntax.
    Multiple conditions are combined with $and.

    Examples:
        {"doc_type": {"$eq": "speech"}}
        {"$and": [{"doc_type": {"$eq": "minutes"}}, {"year": {"$eq": "2024"}}]}
    """
    conditions = []

    if doc_type:
        conditions.append({"doc_type": {"$eq": doc_type}})

    if year:
        conditions.append({"year": {"$eq": str(year)}})

    if speaker:
        # Use $contains for partial speaker name matching
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

def retrieve(
    query:       str,
    top_k:       int = DEFAULT_TOP_K,
    doc_type:    str | None = None,
    year:        str | None = None,
    speaker:     str | None = None,
    institution: str | None = None,
    country:     str | None = None,
    collection   = None,
) -> list[dict]:
    """
    Main retrieval function. Takes a query string and returns
    the top_k most relevant chunks with their metadata.

    This is the function your generation layer will call.

    Returns a list of dicts, each containing:
        - text: the chunk content
        - score: similarity score (0-1, higher is better)
        - metadata: date, speaker, doc_type, source_url etc
    """
    if collection is None:
        collection = get_collection()

    where_filter = build_filter(
        doc_type=doc_type,
        year=year,
        speaker=speaker,
        institution=institution,
        country=country
    )

    # Run the query
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
        print(f"[error] Retrieval failed: {e}")
        return []

    # Format results into clean dicts
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text":        doc,
            "score":       round(1 - dist, 4),  # cosine distance → similarity
            "date":        meta.get("date", ""),
            "speaker":     meta.get("speaker", ""),
            "doc_type":    meta.get("doc_type", ""),
            "title":       meta.get("title", ""),
            "source_url":  meta.get("source_url", ""),
            "institution": meta.get("institution", ""),
            "country":     meta.get("country", ""),
            "chunk_index": meta.get("chunk_index", 0),
        })

    # Sort by score descending
    chunks.sort(key=lambda x: x["score"], reverse=True)

    return chunks


# ── Formatting ────────────────────────────────────────────────────────────────

def format_results(chunks: list[dict], query: str) -> str:
    """
    Format retrieved chunks for display in the terminal.
    This is also useful as a reference for how to present
    results in your Streamlit UI later.
    """
    if not chunks:
        return "No results found. Try a different query or remove filters."

    lines = []
    lines.append(f"\n── Results for: '{query}' ──────────────────────────")
    lines.append(f"   Found {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks, 1):
        lines.append(f"  [{i}] Score: {chunk['score']:.3f}")
        lines.append(f"      Date:    {chunk['date']}")
        lines.append(f"      Type:    {chunk['doc_type']}")
        lines.append(f"      Speaker: {chunk['speaker'][:60]}")
        lines.append(f"      Source:  {chunk['source_url']}")
        lines.append(f"      ─────────────────────────────────────")
        lines.append(f"      {chunk['text'][:400]}...")
        lines.append("")

    return "\n".join(lines)


def format_context_for_llm(chunks: list[dict]) -> str:
    """
    Format chunks into a context block ready to pass to an LLM.
    Each chunk is clearly labelled with its source and date
    so the LLM can cite sources in its answer.

    This is what you'll pass as {context} in your generation prompt.
    """
    if not chunks:
        return "No relevant context found."

    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        source_label = (
            f"[Source {i}] "
            f"{chunk['doc_type'].upper()} | "
            f"{chunk['date']} | "
            f"{chunk['speaker'][:50]}"
        )
        context_parts.append(f"{source_label}\n{chunk['text']}")

    return "\n\n---\n\n".join(context_parts)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Query the MacroThread knowledge base")
    parser.add_argument("query", type=str, help="Your research question")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of results (default: {DEFAULT_TOP_K})")
    parser.add_argument("--doc-type", type=str, choices=["speech", "minutes"],
                        help="Filter by document type")
    parser.add_argument("--year", type=str,
                        help="Filter by year e.g. 2024")
    parser.add_argument("--speaker", type=str,
                        help="Filter by speaker name (partial match)")
    parser.add_argument("--show-context", action="store_true",
                        help="Show the formatted LLM context block")
    args = parser.parse_args()

    print("\nMacroThread Retriever")

    collection = get_collection()
    print(f"  Connected — {collection.count():,} vectors in collection\n")

    chunks = retrieve(
        query=args.query,
        top_k=args.top_k,
        doc_type=args.doc_type,
        year=args.year,
        speaker=args.speaker,
        collection=collection
    )

    # Display results
    print(format_results(chunks, args.query))

    # Optionally show the LLM context block
    if args.show_context:
        print("\n── LLM Context Block ───────────────────────────────")
        print(format_context_for_llm(chunks))


if __name__ == "__main__":
    main()
