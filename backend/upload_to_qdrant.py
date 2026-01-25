#!/usr/bin/env python3
"""
One-time script to embed all chunks and upload to Qdrant Cloud.
Run this ONCE before using the RAG pipeline.

Usage:
    python upload_to_qdrant.py
"""

import json
import os
import requests
import logging
import uuid
import time
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models

from dotenv import load_dotenv
load_dotenv()

# Configuration
COLLECTION_NAME = "siggraph2025_papers"
CHUNKS_PATH = "./chunks.json"
BATCH_SIZE = 50  # How many chunks to embed per API call (OpenRouter limit)
EMBEDDING_MODEL = "openai/text-embedding-3-large"  # Via OpenRouter
VECTOR_SIZE = 3072  # Dimension for text-embedding-3-large

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Keep tqdm progress bar on a single line by silencing noisy per-request logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def load_chunks(path: str) -> list[dict]:
    """
    Load chunks from the JSON file.
    
    TODO:
    1. Open the file at `path` using open() with encoding='utf-8'
    2. Parse JSON using json.load()
    3. Return the list of chunks (hint: data["chunks"])
    
    Returns:
        List of chunk dictionaries
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def get_embeddings_batch(
    texts: list[str],
    api_key: str,
    *,
    max_retries: int = 5,
    initial_backoff_s: float = 2.0,
) -> list[list[float]]:
    """
    Get embeddings for multiple texts in one API call.
    
    TODO:
    1. Build headers dict with:
       - "Authorization": f"Bearer {api_key}"
       - "Content-Type": "application/json"
    
    2. Build payload dict with:
       - "model": EMBEDDING_MODEL
       - "input": texts  (this is the list of texts)
    
    3. Make POST request to "https://openrouter.ai/api/v1/embeddings"
       using requests.post(url, headers=headers, json=payload)
    
    4. Check response.status_code == 200, raise error if not
    
    5. Parse response: data = response.json()
    
    6. Extract embeddings: [item["embedding"] for item in data["data"]]
    
    7. Return the list of embeddings
    
    Args:
        texts: List of text strings to embed
        api_key: OpenRouter API key
        
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }
    last_error: Exception | None = None
    backoff_s = initial_backoff_s

    for attempt in range(1, max_retries + 1):
        response = None
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=60,
            )

            # Try to parse JSON either way; OpenRouter returns useful error bodies.
            try:
                data = response.json()
            except Exception:
                data = None

            if response.status_code != 200:
                msg = response.text
                if isinstance(data, dict) and "error" in data:
                    msg = str(data["error"])
                raise RuntimeError(f"OpenRouter embeddings failed: HTTP {response.status_code} {msg}")

            if not isinstance(data, dict) or "data" not in data:
                # logging.error(payload)
                # logging.error(response.text)
                # logging.error(data)
                # e.g. {"error": {"message": "No successful provider responses.", "code": 404}}
                raise RuntimeError(f"OpenRouter embeddings response missing 'data': {data}")

            embeddings = [item["embedding"] for item in data["data"]]
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"OpenRouter embeddings count mismatch: got {len(embeddings)} for {len(texts)} inputs"
                )
            return embeddings

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    f"Embedding request failed (attempt {attempt}/{max_retries}), retrying in {backoff_s:.1f}s: {e}"
                )
                time.sleep(backoff_s)
                backoff_s *= 2
                continue
            break

    assert last_error is not None
    logger.error(f"Embedding request failed after {max_retries} attempts: {last_error}")
    raise last_error


def _get_existing_point_ids(
    client: QdrantClient, collection_name: str, point_ids: list[str]
) -> set[str]:
    """
    Return the subset of `point_ids` that already exist in Qdrant.
    Used to make uploads resumable/idempotent.
    """
    if not point_ids:
        return set()
    try:
        records = client.retrieve(
            collection_name=collection_name,
            ids=point_ids,
            with_payload=False,
            with_vectors=False,
        )
        # records may contain None entries depending on client version/settings
        return {str(r.id) for r in records if r is not None}
    except Exception as e:
        logger.warning(f"Could not check existing points (will not skip): {e}")
        return set()


def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Create a Qdrant collection if it doesn't exist.
    
    TODO:
    1. Check if collection exists:
    
    2. Create the collection:
    
    3. Print success message
    
    Args:
        client: QdrantClient instance
        collection_name: Name for the collection
        vector_size: Dimension of vectors (3072 for text-embedding-3-large)
    """
    if client.collection_exists(collection_name):
        logger.info(f"Collection {collection_name} already exists")
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
    )
    logger.info(f"Collection {collection_name} created")


def upload_chunks_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict],
    api_key: str,
    batch_size: int = 50,
    skip_existing: bool = True,
    failed_batches_log_path: str = "failed_batches.jsonl",
):
    """
    Embed all chunks and upload to Qdrant in batches.
    
    TODO:
    1. Calculate total batches: total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    2. Loop through chunks in batches using range(0, len(chunks), batch_size)
       Use tqdm for progress bar: for i in tqdm(range(...), desc="Uploading"):
    
    3. For each batch:
       a. Get the batch slice: batch = chunks[i:i + batch_size]
       
       b. Extract texts for each chunk in the batch
       
       c. Get embeddings: embeddings = get_embeddings_batch(texts, api_key)
       
       d. Create Qdrant points - for each chunk and embedding pair:
          points = [
              models.PointStruct(
                  id=chunk["chunk_id"],  # Use chunk_id as the unique point ID
                  vector=embedding,
                  payload={
                      "chunk_id": chunk["chunk_id"],
                      "paper_id": chunk["paper_id"],
                      "title": chunk["title"],
                      "authors": chunk["authors"],
                      "text": chunk["text"],
                      "chunk_type": chunk["chunk_type"],
                      "chunk_section": chunk.get("chunk_section", ""),
                      "pdf_url": chunk.get("pdf_url"),
                      "github_link": chunk.get("github_link"),
                      "video_link": chunk.get("video_link"),
                      "acm_url": chunk.get("acm_url"),
                      "abstract_url": chunk.get("abstract_url"),
                  }
              )
              for chunk, embedding in zip(batch, embeddings)
          ]
       
       e. Upload to Qdrant: client.upsert(collection_name=collection_name, points=points)
    
    4. Print completion message with total chunks uploaded
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        chunks: List of chunk dictionaries
        api_key: OpenRouter API key
        batch_size: Number of chunks per batch
    """
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading"):
        batch = chunks[i:i + batch_size]
        # Deterministic UUIDv5 makes reruns idempotent/resumable.
        original_chunk_ids = [chunk["chunk_id"] for chunk in batch]
        point_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
            for chunk_id in original_chunk_ids
        ]

        existing_ids: set[str] = set()
        if skip_existing:
            existing_ids = _get_existing_point_ids(client, collection_name, point_ids)

        missing_idxs = [idx for idx, pid in enumerate(point_ids) if pid not in existing_ids]
        if not missing_idxs:
            continue

        missing_chunks = [batch[idx] for idx in missing_idxs]
        missing_texts = [chunk["text"] for chunk in missing_chunks]
        try:
            embeddings = get_embeddings_batch(missing_texts, api_key)
            if len(embeddings) != len(missing_chunks):
                raise RuntimeError(
                    f"Embedding count mismatch: got {len(embeddings)} embeddings for {len(missing_chunks)} chunks"
                )
        except Exception as e:
            # If OpenRouter fails even after retries, skip this batch and continue.
            logger.error(f"Skipping batch starting at index {i} due to embedding failure: {e}")
            try:
                with open(failed_batches_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "batch_start_idx": i,
                                "batch_size": len(batch),
                                "missing_count": len(missing_chunks),
                                "missing_point_ids": [point_ids[idx] for idx in missing_idxs],
                                "missing_chunk_ids": [batch[idx].get("chunk_id") for idx in missing_idxs],
                                "paper_ids": list({batch[idx].get("paper_id") for idx in missing_idxs}),
                                "error": str(e),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception as log_err:
                logger.warning(f"Failed to write failed-batch log: {log_err}")
            continue

        missing_point_ids = [point_ids[idx] for idx in missing_idxs]
        points = [
            models.PointStruct(id=pid, vector=embedding, payload=chunk)
            for pid, embedding, chunk in zip(missing_point_ids, embeddings, missing_chunks)
        ]
        try:
            client.upsert(collection_name=collection_name, points=points)
        except Exception as e:
            logger.error(f"Skipping batch starting at index {i} due to Qdrant upsert failure: {e}")
            try:
                with open(failed_batches_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "batch_start_idx": i,
                                "batch_size": len(batch),
                                "missing_count": len(missing_chunks),
                                "missing_point_ids": missing_point_ids,
                                "missing_chunk_ids": [c.get("chunk_id") for c in missing_chunks],
                                "paper_ids": list({c.get("paper_id") for c in missing_chunks}),
                                "error": str(e),
                                "stage": "upsert",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception as log_err:
                logger.warning(f"Failed to write failed-batch log: {log_err}")
            continue
    logger.info(f"Uploaded {len(chunks)} chunks to {collection_name}")


def main():
    """Main function to run the upload process."""
    
    # Step 1: Load environment variables
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    # TODO: Step 2 - Validate environment variables
    # Check that all three variables are set (not None or empty)
    # If any are missing, print an error message and exit
    # Example: if not openrouter_api_key: raise ValueError("OPENROUTER_API_KEY not set")
    if not openrouter_api_key:
        logger.error("OPENROUTER_API_KEY not set")
        raise ValueError("OPENROUTER_API_KEY not set")
    if not qdrant_url:
        logger.error("QDRANT_URL not set")
        raise ValueError("QDRANT_URL not set")
    if not qdrant_api_key:
        logger.error("QDRANT_API_KEY not set")
        raise ValueError("QDRANT_API_KEY not set")
    
    # TODO: Step 3 - Initialize Qdrant client
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    # TODO: Step 4 - Load chunks from JSON file
    chunks = load_chunks(CHUNKS_PATH)
    # print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
    
    # TODO: Step 5 - Create collection
    create_qdrant_collection(client, COLLECTION_NAME, VECTOR_SIZE)
    
    # TODO: Step 6 - Upload all chunks with embeddings
    upload_chunks_to_qdrant(client, COLLECTION_NAME, chunks, openrouter_api_key, BATCH_SIZE)
    
    logger.info("Done! Your vectors are now in Qdrant Cloud.")
    logger.info(f"Collection: {COLLECTION_NAME}")
    logger.info("You can verify in your Qdrant Cloud dashboard.")


if __name__ == "__main__":
    main()
