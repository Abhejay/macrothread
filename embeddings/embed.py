import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os

import chromedb
from chromedb.utils import embedding_function
from tqdm import tqdm

load_dotenv()

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "macrothread"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 50
RETRY_DELAY = 5


def load_chunks(path: Path) -> List[dict]:
    if not path.exists():
        print(f"[error] No chunks file found at {path}")
        print("  Run: python ingestion/chunker.py first")
        return []
    
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks):,} chunks from {path.name}")
    return chunks

def get_collection(reset: bool = False):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Make sure your .env file exists and contains OPENAI_API_KEY=..."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBED_MODEL
    )

    if reset:
        print("Resetting collection...")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )

    return collection

def already_embedded(collection, chunk_ids: list[str]) -> set[str]:
    try:
        existing = collection.get(ids=chunk_ids)
        return set(existing["ids"])
    except Exception:
        return set()

def prepare_batch(chunks: list[dict]) -> tuple[list, list, list]:
    ids       = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])

        metadata = {
            "chunk_index":  chunk.get("chunk_index", 0),
            "total_chunks": chunk.get("total_chunks", 0),
            "title":        chunk.get("title", "")[:500],
            "speaker":      chunk.get("speaker", "")[:100],
            "date":         chunk.get("date", ""),
            "source_url":   chunk.get("source_url", "")[:500],
            "institution":  chunk.get("institution", ""),
            "country":      chunk.get("country", ""),
            "region":       chunk.get("region", ""),
            "doc_type":     chunk.get("doc_type", ""),
            "year":         chunk.get("date", "")[:4],
        }

        metadata = {k: (v if v is not None else "") for k, v in metadata.items()}
        metadatas.append(metadata)

    return ids, documents, metadatas

def embed_chunks(collection, chunks: list[dict], batch_size: int):
    total    = len(chunks)
    saved    = 0
    skipped  = 0

    print(f"\n  Embedding {total:,} chunks in batches of {batch_size}...")
    print(f"  Model: {EMBED_MODEL}")
    print(f"  Estimated cost: ~${total * 0.00000002:.4f} (text-embedding-3-small)\n")

    for i in tqdm(range(0, total, batch_size), desc="Embedding"):
        batch = chunks[i:i + batch_size]

        ids, documents, metadatas = prepare_batch(batch)
        existing = already_embedded(collection, ids)
        new_ids       = [id_ for id_ in ids if id_ not in existing]
        new_documents = [doc for id_, doc in zip(ids, documents) if id_ not in existing]
        new_metadatas = [meta for id_, meta in zip(ids, metadatas) if id_ not in existing]

        if not new_ids:
            skipped += len(batch)
            continue

        for attempt in range(3):
            try:
                collection.upsert(
                    ids=new_ids,
                    documents=new_documents,
                    metadatas=new_metadatas
                )
                saved += len(new_ids)
                skipped += len(batch) - len(new_ids)
                break

            except Exception as e:
                if "rate" in str(e).lower() and attempt < 2:
                    print(f"\n  Rate limited — waiting {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"\n  [error] Failed to embed batch {i}-{i+batch_size}: {e}")
                    break

    return saved, skipped

def verify_collection(collection):
    print("\n── Verification Query ──────────────────────────────")
    print("  Query: 'Federal Reserve inflation interest rates'\n")

    try:
        results = collection.query(
            query_texts=["Federal Reserve inflation interest rates"],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ), 1):
            print(f"  Result {i}:")
            print(f"    Score:    {1 - dist:.3f}")
            print(f"    Date:     {meta.get('date', 'unknown')}")
            print(f"    Type:     {meta.get('doc_type', 'unknown')}")
            print(f"    Speaker:  {meta.get('speaker', 'unknown')[:50]}")
            print(f"    Preview:  {doc[:150]}...")
            print()

    except Exception as e:
        print(f"  [error] Verification failed: {e}")

def print_summary(collection, saved: int, skipped: int):
    count = collection.count()
    print(f"── Summary ─────────────────────────────────────────")
    print(f"  Newly embedded: {saved:,}")
    print(f"  Skipped (already existed): {skipped:,}")
    print(f"  Total in collection: {count:,}")
    print(f"  ChromaDB location: {CHROMA_DIR}")
    print(f"────────────────────────────────────────────────────\n")
    print("✓ Embedding complete. Next step: retrieval/retriever.py")

def main():
    parser = argparse.ArgumentParser(description="Embed chunks into ChromaDB")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Chunks per API batch (default: {BATCH_SIZE})")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe existing collection and re-embed everything")
    args = parser.parse_args()

    print("\nMacroThread Embedder")

    print("\n── Loading Chunks ───────────────────────────────────")
    chunks = load_chunks(CHUNKS_PATH)
    if not chunks:
        return

    print("\n── Connecting to ChromaDB ───────────────────────────")
    collection = get_collection(reset=args.reset)
    print(f"  Collection '{COLLECTION_NAME}' ready")
    print(f"  Currently contains: {collection.count():,} vectors")

    print("\n── Embedding ────────────────────────────────────────")
    saved, skipped = embed_chunks(collection, chunks, args.batch_size)

    verify_collection(collection)

    print_summary(collection, saved, skipped)

if __name__ == "__main__":
    main()
