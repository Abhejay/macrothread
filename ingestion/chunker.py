"""
chunker.py
----------
Splits raw Fed documents into chunks ready for embedding.
Preserves metadata on every chunk so we can filter by date,
institution, doc_type etc during retrieval.

Usage:
    python chunker.py
    python chunker.py --chunk-size 600 --overlap 100
    python chunker.py --show-preview
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR    = Path(__file__).parent.parent / "data" / "raw" / "fed"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chunk tuning — these are the two most important numbers in your RAG system
# - CHUNK_SIZE: how many characters per chunk (~500 chars ≈ ~120 tokens)
# - OVERLAP: how many chars to repeat between chunks so context isn't lost at boundaries
DEFAULT_CHUNK_SIZE = 500   # characters
DEFAULT_OVERLAP    = 80    # characters


# ── Core Chunking Logic ───────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks.

    We split on paragraph boundaries first (double newlines) to avoid
    cutting sentences mid-thought. If a paragraph is longer than chunk_size
    we fall back to splitting at the nearest sentence boundary.

    Why overlap? Imagine a key fact sits at the end of chunk 3 and the
    explanation is at the start of chunk 4. Without overlap, a query
    might retrieve only chunk 3 or only chunk 4 and miss the full picture.
    Overlap ensures boundary content appears in both adjacent chunks.
    """
    if not text or not text.strip():
        return []

    # Step 1: split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        # If adding this paragraph keeps us under chunk_size, append it
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + paragraph).strip()

        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk)

            # If the paragraph itself is longer than chunk_size,
            # split it further at sentence boundaries
            if len(paragraph) > chunk_size:
                sentence_chunks = split_long_paragraph(paragraph, chunk_size)
                chunks.extend(sentence_chunks[:-1])
                current_chunk = sentence_chunks[-1] if sentence_chunks else ""
            else:
                current_chunk = paragraph

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Step 2: apply overlap — each chunk gets a tail from the previous chunk
    if overlap > 0 and len(chunks) > 1:
        chunks = apply_overlap(chunks, overlap)

    return [c for c in chunks if len(c.strip()) > 50]  # filter out tiny fragments


def split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    """
    Split a paragraph that exceeds chunk_size at sentence boundaries.
    Falls back to hard split if no sentence boundary found.
    """
    sentences = paragraph.replace(". ", ".|").replace("? ", "?|").replace("! ", "!|").split("|")
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If a single sentence is still too long, hard split it
            if len(sentence) > chunk_size:
                for i in range(0, len(sentence), chunk_size):
                    chunks.append(sentence[i:i + chunk_size])
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """
    Prepend the tail of the previous chunk to each chunk.
    This ensures continuity across chunk boundaries.
    """
    overlapped = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        overlapped.append(prev_tail + " [...] " + chunks[i])

    return overlapped


# ── Document Processing ───────────────────────────────────────────────────────

def process_document(doc_dir: Path, chunk_size: int, overlap: int) -> list[dict] | None:
    """
    Load a raw document folder, chunk its content, and return
    a list of chunk dicts — each with text + full metadata.
    """
    content_path  = doc_dir / "content.txt"
    metadata_path = doc_dir / "metadata.json"

    if not content_path.exists() or not metadata_path.exists():
        print(f"  [warning] Missing files in {doc_dir.name}, skipping")
        return None

    text     = content_path.read_text(encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    chunks = chunk_text(text, chunk_size, overlap)

    if not chunks:
        print(f"  [warning] No chunks produced from {doc_dir.name}")
        return None

    # Attach metadata to every chunk
    chunk_dicts = []
    for i, chunk in enumerate(chunks):
        chunk_dicts.append({
            # The actual text that gets embedded
            "text": chunk,

            # Chunk identity
            "chunk_id": f"{doc_dir.name}_chunk_{i:03d}",
            "chunk_index": i,
            "total_chunks": len(chunks),

            # Document metadata — carried forward onto every chunk
            # This is what enables filtering during retrieval
            "title":        metadata.get("title", ""),
            "speaker":      metadata.get("speaker", ""),
            "date":         metadata.get("date", ""),
            "source_url":   metadata.get("source_url", ""),
            "institution":  metadata.get("institution", ""),
            "country":      metadata.get("country", ""),
            "region":       metadata.get("region", ""),
            "doc_type":     metadata.get("doc_type", ""),
            "scraped_at":   metadata.get("scraped_at", ""),
            "chunked_at":   datetime.utcnow().isoformat(),
        })

    return chunk_dicts


def process_all_documents(chunk_size: int, overlap: int, show_preview: bool = False) -> list[dict]:
    """
    Walk all document folders in RAW_DIR and chunk them all.
    Returns a flat list of all chunks across all documents.
    """
    doc_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]

    if not doc_dirs:
        print(f"[error] No documents found in {RAW_DIR}")
        print("  Run fed_scraper.py first to download documents.")
        return []

    print(f"\n── Chunking {len(doc_dirs)} documents ──────────────────────────")
    print(f"   chunk_size: {chunk_size} chars | overlap: {overlap} chars\n")

    all_chunks = []

    for i, doc_dir in enumerate(sorted(doc_dirs), 1):
        chunks = process_document(doc_dir, chunk_size, overlap)

        if chunks is None:
            continue

        print(f"  [{i}/{len(doc_dirs)}] {doc_dir.name}")
        print(f"    → {len(chunks)} chunks | avg {sum(len(c['text']) for c in chunks) // len(chunks)} chars each")

        # Show a preview of the first chunk if requested
        if show_preview and i == 1:
            print(f"\n  ── Preview: first chunk from first document ──")
            print(f"  {chunks[0]['text'][:300]}...")
            print(f"  metadata: date={chunks[0]['date']} | type={chunks[0]['doc_type']} | speaker={chunks[0]['speaker'][:30]}")
            print()

        all_chunks.extend(chunks)

    return all_chunks


# ── Save Output ───────────────────────────────────────────────────────────────

def save_chunks(chunks: list[dict], output_path: Path):
    """Save all chunks to a single JSONL file (one chunk per line)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    print(f"\n  Saved {len(chunks)} chunks → {output_path}")


def print_summary(chunks: list[dict]):
    """Print a summary of what was chunked."""
    if not chunks:
        return

    doc_types  = {}
    date_range = []

    for chunk in chunks:
        dtype = chunk.get("doc_type", "unknown")
        doc_types[dtype] = doc_types.get(dtype, 0) + 1
        if chunk.get("date"):
            date_range.append(chunk["date"])

    date_range.sort()

    print(f"\n── Summary ─────────────────────────────────────────")
    print(f"  Total chunks:  {len(chunks)}")
    print(f"  By doc_type:")
    for dtype, count in doc_types.items():
        print(f"    {dtype}: {count} chunks")
    if date_range:
        print(f"  Date range: {date_range[0]} → {date_range[-1]}")
    print(f"────────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chunk Fed documents for embedding")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Characters per chunk (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                        help=f"Overlap characters between chunks (default: {DEFAULT_OVERLAP})")
    parser.add_argument("--show-preview", action="store_true",
                        help="Print a preview of the first chunk")
    parser.add_argument("--output", type=str, default="chunks.jsonl",
                        help="Output filename in data/processed/ (default: chunks.jsonl)")
    args = parser.parse_args()

    print("\nMacroThread Chunker")

    chunks = process_all_documents(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        show_preview=args.show_preview
    )

    if not chunks:
        return

    output_path = OUTPUT_DIR / args.output
    save_chunks(chunks, output_path)
    print_summary(chunks)


if __name__ == "__main__":
    main()
