#!/usr/bin/env python3
"""
RAG Generation Pipeline for SIGGRAPH 2025 Papers.

Uses the retrieval pipeline to find relevant chunks,
then generates an answer using an OpenAI chat model.

Usage:
    from rag_generate import RAGGenerator, GenerationConfig, SYSTEM_PROMPT

    generator = RAGGenerator()
    result = generator.generate("What is 3D Gaussian Splatting?")
    print(result["answer"])
"""

import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()

from retrieval_pipeline import RetrievalPipeline, RetrievalResult


# =============================================================================
# SYSTEM PROMPT - This tells the LLM how to behave
# =============================================================================
SYSTEM_PROMPT = """You are an expert research assistant specializing in computer graphics, specifically SIGGRAPH 2025 papers.

Your task is to answer questions using ONLY the provided research paper excerpts.

Rules:
1. Cite sources inline right after each claim using the paper's exact title in square brackets, e.g. [Deformable Beta Splatting]. Copy the title exactly as given in the excerpt header - the interface links citations by exact title.
2. Never cite by source number (no [1] or [Source 2]) and never put two titles in one bracket - use [Title A] [Title B].
3. Be comprehensive and technically accurate. Use markdown: short paragraphs, bullet lists, and bold for key terms.
4. If the excerpts don't contain the answer, say so plainly instead of guessing.
5. Use LaTeX for math: $inline$ or $$block$$
6. Do NOT make up information not in the excerpts.
7. Do NOT include a References section at the end.
"""


# =============================================================================
# QUERY REFINEMENT PROMPT
# =============================================================================
QUERY_REFINEMENT_PROMPT = """You are an expert at refining search queries for academic paper retrieval.

Given a user's question, rewrite it as a clear, focused search query that will retrieve the most relevant SIGGRAPH research papers.

Keep it concise (under 20 words). Focus on key technical terms and keep any names, acronyms, or method names exactly as written. Output only the query.

User question: {query}

Refined search query:"""


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class GenerationConfig:
    """Configuration for the RAG generator."""
    llm_provider: str = "openai"
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = 0.1  # Low temperature for factual answers
    max_tokens: int = 1500  # Max length of generated answer
    openai_api_key: Optional[str] = None  # Will load from env if not set
    refine_query: bool = True  # Whether to refine queries before retrieval
    refinement_model: str = field(default_factory=lambda: os.getenv("REFINEMENT_MODEL", "gpt-4o-mini"))
    retrieval_top_k: int = 8  # Number of chunks to retrieve
    use_reranker: bool = True


# =============================================================================
# RAG GENERATOR CLASS
# =============================================================================
class RAGGenerator:
    """
    Main RAG class - this is what api_server.py uses!

    Flow:
    1. Refine the user's query (optional)
    2. Retrieve relevant chunks using the retrieval pipeline
    3. Format chunks into context
    4. Generate answer using LLM (blocking or streamed)
    5. Return answer with source metadata
    """

    def __init__(self, config: Optional[GenerationConfig] = None, retrieval_pipeline=None):
        self.config = config or GenerationConfig()
        self.retrieval = retrieval_pipeline or RetrievalPipeline()

        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key, timeout=60)
        self.async_client = AsyncOpenAI(api_key=api_key, timeout=60)

    def refine_query(self, query: str) -> str:
        """
        Use a small LLM call to turn a conversational question into a focused
        search query. Falls back to the original query on any failure.
        """
        if not self.config.refine_query:
            return query
        try:
            response = self.client.chat.completions.create(
                model=self.config.refinement_model,
                messages=[{"role": "user", "content": QUERY_REFINEMENT_PROMPT.format(query=query)}],
                temperature=0.3,
                max_tokens=100,
            )
            refined = response.choices[0].message.content.strip().strip('"\'')
            return refined or query
        except Exception as e:
            print(f"Query refinement failed, using original query: {e}")
            return query

    def _format_context(self, results: list) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        return "\n".join(
            f"""
--- Source {i} ---
Title: {r.title}
Authors: {r.authors}
Section: {r.chunk_section}

Content:
{r.text}
"""
            for i, r in enumerate(results, 1)
        )

    def _build_sources_metadata(self, results: list) -> list:
        """
        Build list of unique source papers (in retrieval order) for citations.
        The frontend displays these as clickable source links.
        """
        seen = {}
        for r in results:
            if r.title not in seen:
                seen[r.title] = {
                    "title": r.title,
                    "authors": r.authors,
                    "pdf_url": r.pdf_url,
                    "github_link": r.github_link,
                    "video_link": r.video_link,
                    "acm_url": r.acm_url,
                    "abstract_url": r.abstract_url,
                }
        return list(seen.values())

    def _build_messages(self, query: str, results: list) -> list:
        titles = "\n".join(f"- {t}" for t in dict.fromkeys(r.title for r in results))
        user_message = f"""Based on the following research paper excerpts, answer this question.

Question: {query}

Research Paper Excerpts:
{self._format_context(results)}

Papers you may cite (use these exact titles in square brackets):
{titles}

Review ALL excerpts and cite every paper that is relevant. For "which paper" or "what papers" questions, list ALL relevant papers."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    def _call_llm(self, query: str, results: list) -> str:
        """Generate a complete answer in one blocking call."""
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=self._build_messages(query, results),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content

    async def stream_answer(self, query: str, results: list) -> AsyncIterator[str]:
        """Stream the answer token-by-token (used by the SSE / WebSocket endpoints)."""
        stream = await self.async_client.chat.completions.create(
            model=self.config.llm_model,
            messages=self._build_messages(query, results),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def generate(self, query: str, top_k: Optional[int] = None, return_sources: bool = True) -> dict:
        """
        Full RAG pipeline - retrieve relevant chunks and generate an answer.
        THIS IS THE MAIN METHOD THAT api_server.py CALLS for /api/query!
        """
        refined = self.refine_query(query)
        results = self.retrieval.retrieve(
            refined, top_k=top_k or self.config.retrieval_top_k, use_reranker=self.config.use_reranker
        )
        if not results:
            return {
                "query": query,
                "refined_query": refined,
                "answer": "I couldn't find any relevant papers to answer this question.",
                "sources": [],
            }
        return {
            "query": query,
            "refined_query": refined,
            "answer": self._call_llm(query, results),
            "sources": self._build_sources_metadata(results) if return_sources else [],
        }


# =============================================================================
# CLI FOR TESTING
# =============================================================================
if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What is 3D Gaussian Splatting?"

    print("Initializing RAG Generator...")
    generator = RAGGenerator()

    print(f"\nQuery: {query}")
    print("=" * 60)

    result = generator.generate(query)

    print(f"Refined Query: {result.get('refined_query', 'N/A')}")
    print("=" * 60)
    print("\nAnswer:")
    print(result['answer'])
    print("=" * 60)
    print(f"\nSources: {len(result.get('sources', []))} papers")
    for source in result.get('sources', []):
        print(f"  - {source['title']}")
