#!/usr/bin/env python3
"""
Retrieval Pipeline for SIGGRAPH 2025 Papers.

Implements hybrid search:
1. Semantic search (OpenAI embeddings, in-memory cosine similarity)
2. Keyword search (BM25 - runs locally)
3. Reranking (Cohere API - optional, used when COHERE_API_KEY is set)

The chunk embeddings are precomputed by build_index.py and loaded from
embeddings_openai.npz, so only the query is embedded at request time.

Usage:
    from retrieval_pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline()
    results = pipeline.retrieve("3D Gaussian Splatting", top_k=5)
"""

import gc
import gzip
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class RetrievalResult:
    """
    Represents a single search result.
    The api_server.py expects these exact fields - do not change!
    """
    chunk_id: str
    paper_id: str
    title: str
    authors: str
    text: str
    score: float
    chunk_type: str = ""
    chunk_section: str = ""
    pdf_url: Optional[str] = None
    github_link: Optional[str] = None
    video_link: Optional[str] = None
    acm_url: Optional[str] = None
    abstract_url: Optional[str] = None


@dataclass
class RetrievalPipelineConfig:
    """Configuration for the retrieval pipeline."""
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 512
    chunks_path: str = "./chunks_compact.json.gz"  # written by build_index.py
    index_path: str = "./embeddings_openai.npz"
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    candidates_per_method: int = 50
    max_chunks_per_paper: int = 3
    use_reranker: bool = True
    cohere_api_key: Optional[str] = None


class OpenAIEmbedder:
    """
    Generate embeddings using the OpenAI API.
    Used to embed user queries for semantic search - must use the same
    model and dimensions as build_index.py used for the chunks.
    """

    def __init__(self, api_key: str, model: str, dims: int):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dims = dims

    def embed_query(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=text, dimensions=self.dims)
        vector = np.array(response.data[0].embedding, dtype=np.float32)
        return vector / np.linalg.norm(vector)


STOPWORDS = set("""
a an and are as at be but by can do does for from has have how i in into is it its
me my of on or our paper papers so that the their them then there these they this
to was we were what when where which who why will with would you your about any
some use used using also than more most such via between do does did
""".split())


class BM25Index:
    """
    BM25 index for keyword search.
    This runs entirely locally - no API calls needed!
    BM25 is good at finding exact keyword matches (method names, acronyms,
    author names) that semantic search might miss.

    Uses an inverted index (term -> postings) so scoring only touches documents
    that contain a query term, which keeps it fast and memory-light.
    """

    def __init__(self, chunks: list, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.num_docs = len(chunks)

        postings = defaultdict(lambda: ([], []))
        doc_lengths = np.zeros(self.num_docs, dtype=np.float32)
        for idx, chunk in enumerate(chunks):
            tokens = self._tokenize(chunk["text"])
            doc_lengths[idx] = len(tokens)
            for term, tf in Counter(tokens).items():
                postings[term][0].append(idx)
                postings[term][1].append(tf)

        self.avg_doc_length = float(doc_lengths.mean())
        # Precompute the length normalization part of the BM25 denominator
        self.length_norm = k1 * (1 - b + b * doc_lengths / self.avg_doc_length)
        self.postings = {
            term: (np.array(docs, dtype=np.int32), np.array(tfs, dtype=np.float32))
            for term, (docs, tfs) in postings.items()
        }
        self.idf = {
            term: math.log(1 + (self.num_docs - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, (docs, _) in self.postings.items()
        }

    @staticmethod
    def _tokenize(text: str) -> list:
        """Lowercase, split into alphanumeric words, drop stopwords."""
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]

    def search(self, query: str, top_k: int = 50) -> list:
        """Return up to top_k (chunk_index, score) tuples, best first."""
        scores = np.zeros(self.num_docs, dtype=np.float32)
        for term in set(self._tokenize(query)):
            if term not in self.postings:
                continue
            docs, tfs = self.postings[term]
            scores[docs] += self.idf[term] * tfs * (self.k1 + 1) / (tfs + self.length_norm[docs])

        top = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


class RetrievalPipeline:
    """
    Main retrieval pipeline combining semantic search + BM25 + reranking.
    This is what api_server.py uses to find relevant chunks.
    """

    def __init__(self, config: Optional[RetrievalPipelineConfig] = None):
        if config is None:
            config = RetrievalPipelineConfig(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                embedding_dims=int(os.getenv("EMBEDDING_DIMS", 512)),
                chunks_path=os.getenv("CHUNKS_PATH") or "./chunks_compact.json.gz",
                index_path=os.getenv("INDEX_PATH") or "./embeddings_openai.npz",
                cohere_api_key=os.getenv("COHERE_API_KEY") or None,
            )
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.config = config

        opener = gzip.open if config.chunks_path.endswith(".gz") else open
        with opener(config.chunks_path, "rt") as f:
            self.chunks = [self._clean_chunk(c) for c in json.load(f)["chunks"]]
        gc.collect()
        print(f"Loaded {len(self.chunks):,} chunks")

        index = np.load(config.index_path)
        if list(index["chunk_ids"]) != [c["chunk_id"] for c in self.chunks]:
            raise ValueError(f"{config.index_path} is out of date with {config.chunks_path} - run build_index.py")
        self.embeddings = index["embeddings"].astype(np.float32)
        print(f"Loaded embeddings {self.embeddings.shape}")

        self.embedder = OpenAIEmbedder(config.openai_api_key, config.embedding_model, config.embedding_dims)
        self.bm25_index = BM25Index(self.chunks)
        print("BM25 index built")

    @staticmethod
    def _clean_chunk(chunk: dict) -> dict:
        """Turn the dataset's 'None' / empty link strings into real None."""
        for key in ("pdf_url", "github_link", "video_link", "acm_url", "abstract_url"):
            if chunk.get(key) in ("None", "", "nan"):
                chunk[key] = None
        return chunk

    def semantic_search(self, query: str, top_k: int = 50) -> list:
        """Cosine similarity between the query and every chunk embedding."""
        query_embedding = self.embedder.embed_query(query)
        scores = self.embeddings @ query_embedding
        top = np.argpartition(-scores, top_k)[:top_k]
        top = top[np.argsort(-scores[top])]
        return [
            {"chunk_id": self.chunks[i]["chunk_id"], "score": float(scores[i]), "payload": self.chunks[i]}
            for i in top
        ]

    def bm25_search(self, query: str, top_k: int = 50) -> list:
        return [
            {"chunk_id": self.chunks[idx]["chunk_id"], "score": score, "payload": self.chunks[idx]}
            for idx, score in self.bm25_index.search(query, top_k)
        ]

    @staticmethod
    def _normalize(results: list) -> None:
        """Min-max normalize scores to [0, 1] so the two methods are comparable."""
        if not results:
            return
        scores = [r["score"] for r in results]
        lo, hi = min(scores), max(scores)
        for r in results:
            r["normalized_score"] = (r["score"] - lo) / (hi - lo) if hi > lo else 1.0

    def hybrid_search(self, query: str, use_hybrid: bool = True) -> list:
        """
        Combine semantic and BM25 results using weighted scoring:
            combined = semantic_weight * semantic + bm25_weight * bm25
        A chunk found by only one method gets 0 for the other.
        """
        k = self.config.candidates_per_method
        semantic_results = self.semantic_search(query, k)
        bm25_results = self.bm25_search(query, k) if use_hybrid else []
        self._normalize(semantic_results)
        self._normalize(bm25_results)

        combined = {}
        for r in semantic_results:
            combined[r["chunk_id"]] = {"payload": r["payload"], "semantic_score": r["normalized_score"], "bm25_score": 0.0}
        for r in bm25_results:
            entry = combined.setdefault(r["chunk_id"], {"payload": r["payload"], "semantic_score": 0.0, "bm25_score": 0.0})
            entry["bm25_score"] = r["normalized_score"]

        semantic_weight = self.config.semantic_weight if use_hybrid else 1.0
        for entry in combined.values():
            entry["combined_score"] = (
                semantic_weight * entry["semantic_score"] + self.config.bm25_weight * entry["bm25_score"]
                if use_hybrid else entry["semantic_score"]
            )
        return sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)

    def rerank(self, query: str, results: list, top_k: int = 20) -> list:
        """Rerank with Cohere if a key is configured; otherwise keep hybrid order."""
        if not self.config.cohere_api_key or not results:
            return results[:top_k]
        try:
            response = requests.post(
                "https://api.cohere.ai/v1/rerank",
                headers={"Authorization": f"Bearer {self.config.cohere_api_key}"},
                json={
                    "model": "rerank-english-v3.0",
                    "query": query,
                    "documents": [r["payload"]["text"] for r in results],
                    "top_n": top_k,
                },
                timeout=15,
            )
            response.raise_for_status()
            reranked = []
            for item in response.json()["results"]:
                r = results[item["index"]]
                r["rerank_score"] = item["relevance_score"]
                reranked.append(r)
            return reranked
        except Exception as e:
            print(f"Rerank failed, using hybrid order: {e}")
            return results[:top_k]

    def _diversify(self, results: list, top_k: int) -> list:
        """Keep at most max_chunks_per_paper chunks per paper so answers cite more papers."""
        per_paper = Counter()
        final = []
        for r in results:
            paper_id = r["payload"]["paper_id"]
            if per_paper[paper_id] < self.config.max_chunks_per_paper:
                per_paper[paper_id] += 1
                final.append(r)
                if len(final) == top_k:
                    break
        return final

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        use_hybrid: bool = True,
        use_reranker: Optional[bool] = None,
    ) -> list:
        """
        Full retrieval pipeline - THIS IS WHAT api_server.py CALLS!
        hybrid search -> optional rerank -> per-paper diversity -> top_k.
        """
        candidates = self.hybrid_search(query, use_hybrid=use_hybrid)
        if self.config.use_reranker if use_reranker is None else use_reranker:
            candidates = self.rerank(query, candidates, top_k=min(top_k * 3, len(candidates)))
        final = self._diversify(candidates, top_k)

        return [
            RetrievalResult(
                chunk_id=r["payload"]["chunk_id"],
                paper_id=r["payload"]["paper_id"],
                title=r["payload"]["title"],
                authors=r["payload"]["authors"],
                text=r["payload"]["text"],
                score=r.get("rerank_score", r["combined_score"]),
                chunk_type=r["payload"].get("chunk_type", ""),
                chunk_section=r["payload"].get("chunk_section", ""),
                pdf_url=r["payload"].get("pdf_url"),
                github_link=r["payload"].get("github_link"),
                video_link=r["payload"].get("video_link"),
                acm_url=r["payload"].get("acm_url"),
                abstract_url=r["payload"].get("abstract_url"),
            )
            for r in final
        ]


# For testing this file directly
if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "3D Gaussian Splatting"

    print(f"Testing retrieval pipeline with query: '{query}'")
    print("=" * 60)

    pipeline = RetrievalPipeline()
    results = pipeline.retrieve(query, top_k=5)

    print(f"\nFound {len(results)} results:\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.4f}] {r.title[:60]}...")
        print(f"   Paper ID: {r.paper_id}")
        print(f"   Text preview: {r.text[:100]}...")
        print()
