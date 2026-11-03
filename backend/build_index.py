#!/usr/bin/env python3
"""
Build the semantic search index for the SIGGRAPH 2025 chunks.

Embeds every chunk in chunks.json with OpenAI and saves the vectors to
embeddings_openai.npz (same order as chunks.json), plus a compact
chunks_compact.json.gz. Both files are small (~11 MB + ~5 MB) so they are
committed to the repo and loaded into memory by retrieval_pipeline.py at
startup - no external vector DB needed.

Usage:
    python build_index.py
"""

import gzip
import json
import os
import time
import unicodedata

import numpy as np
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()

CHUNKS_PATH = "./chunks.json"
COMPACT_CHUNKS_PATH = "./chunks_compact.json.gz"
INDEX_PATH = os.getenv("INDEX_PATH") or "./embeddings_openai.npz"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", 512))
MAX_BATCH_CHARS = 200_000  # ~50k tokens per request, well under API limits
MAX_INPUT_CHARS = 8_000  # at worst ~1 char/token, stays under the 8192-token limit


def batches(texts: list[str]):
    """Yield (start, end) index ranges whose total text fits in one request."""
    start, size = 0, 0
    for i, t in enumerate(texts):
        if size + len(t) > MAX_BATCH_CHARS and i > start:
            yield start, i
            start, size = i, 0
        size += len(t)
    yield start, len(texts)


def main():
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)["chunks"]
    # A few chunks are huge extracted tables that exceed the 8192-token input
    # limit; their opening text is enough to place them semantically.
    texts = [c["text"][:MAX_INPUT_CHARS] for c in chunks]
    print(f"Embedding {len(texts):,} chunks with {EMBEDDING_MODEL} ({EMBEDDING_DIMS} dims)...")

    client = OpenAI()
    vectors = np.zeros((len(texts), EMBEDDING_DIMS), dtype=np.float32)
    t0 = time.time()
    for start, end in batches(texts):
        for attempt in range(5):
            try:
                resp = client.embeddings.create(
                    model=EMBEDDING_MODEL, input=texts[start:end], dimensions=EMBEDDING_DIMS
                )
                break
            except BadRequestError:
                raise  # invalid input - retrying won't help
            except Exception as e:
                wait = 2 ** attempt
                print(f"  retry in {wait}s: {e}")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to embed chunks {start}-{end}")
        for item in resp.data:
            vectors[start + item.index] = item.embedding
        print(f"  {end:,}/{len(texts):,}  ({time.time() - t0:.0f}s)")

    # Normalize so a dot product equals cosine similarity
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.savez_compressed(
        INDEX_PATH,
        embeddings=vectors.astype(np.float16),
        chunk_ids=np.array([c["chunk_id"] for c in chunks]),
        model=np.array(EMBEDDING_MODEL),
    )
    print(f"Saved {INDEX_PATH} ({os.path.getsize(INDEX_PATH) / 1e6:.1f} MB)")

    # Compact copy of the chunks for the server. The PDF text is full of math
    # alphanumerics (e.g. U+1D465 '𝑥') that force Python to store each string at
    # 4 bytes/char; NFKC folds them to plain letters, cutting load memory from
    # ~380 MB to ~150 MB (important on 512 MB hosts) and helping BM25 match.
    for c in chunks:
        c["text"] = unicodedata.normalize("NFKC", c["text"])
    with gzip.open(COMPACT_CHUNKS_PATH, "wt") as f:
        json.dump({"chunks": chunks}, f)
    print(f"Saved {COMPACT_CHUNKS_PATH} ({os.path.getsize(COMPACT_CHUNKS_PATH) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
