#!/usr/bin/env python3
"""
RAG Generation Pipeline for SIGGRAPH 2025 Papers.

Uses the retrieval pipeline to find relevant chunks,
then generates an answer using an LLM via OpenRouter API.

Usage:
    from rag_generate import RAGGenerator, GenerationConfig, SYSTEM_PROMPT
    
    generator = RAGGenerator()
    result = generator.generate("What is 3D Gaussian Splatting?")
    print(result["answer"])
"""

import os
import requests
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from retrieval_pipeline import RetrievalPipeline, RetrievalResult


# =============================================================================
# SYSTEM PROMPT - This tells the LLM how to behave
# =============================================================================
SYSTEM_PROMPT = """You are an expert research assistant specializing in computer graphics, specifically SIGGRAPH 2025 papers.

Your task is to answer questions using ONLY the provided research paper excerpts.

Rules:
1. Cite sources using [Paper Title] format
2. Be comprehensive and technically accurate
3. If the excerpts don't contain the answer, say so
4. Use LaTeX for math: $inline$ or $$block$$
5. Do NOT make up information not in the excerpts
6. Do NOT include a References section at the end
"""


# =============================================================================
# QUERY REFINEMENT PROMPT
# =============================================================================
QUERY_REFINEMENT_PROMPT = """You are an expert at refining search queries for academic paper retrieval.

Given a user's question, rewrite it as a clear, focused search query that will retrieve the most relevant research papers.

Keep it concise (under 20 words). Focus on key technical terms.

User question: {query}

Refined search query:"""


# =============================================================================
# CONFIGURATION 
# =============================================================================
@dataclass
class GenerationConfig:
    """Configuration for the RAG generator."""
    llm_model: str = "openai/gpt-4o"  # Model to use for answer generation
    temperature: float = 0.1  # Low temperature for factual answers
    max_tokens: int = 2000  # Max length of generated answer
    openrouter_api_key: Optional[str] = None  # Will load from env if not set
    refine_query: bool = True  # Whether to refine queries before retrieval
    refinement_model: str = "openai/gpt-3.5-turbo"  # Cheaper model for refinement
    retrieval_top_k: int = 8  # Number of chunks to retrieve


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
    4. Generate answer using LLM
    5. Return answer with source metadata
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None, retrieval_pipeline=None):
        """
        Initialize the RAG generator.
        
        TODO:
        1. Set config (use default if not provided):
           self.config = config or GenerationConfig()
        
        2. Initialize the retrieval pipeline:
           self.retrieval = retrieval_pipeline or RetrievalPipeline()
        
        3. Get OpenRouter API key (from config or environment):
           self.openrouter_api_key = self.config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        
        4. Validate API key exists:
           if not self.openrouter_api_key:
               raise ValueError("OPENROUTER_API_KEY not set")
        
        5. Store the base URL:
           self.openrouter_base_url = "https://openrouter.ai/api/v1"
        
        Args:
            config: Optional configuration object
            retrieval_pipeline: Optional pre-initialized retrieval pipeline
        """
        # 1. Set config
        self.config = config or GenerationConfig()

        # 2. Initialize retrieval pipeline
        self.retrieval = retrieval_pipeline or RetrievalPipeline()

        # 3. Load OpenRouter API key
        self.openrouter_api_key = self.config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        print(f"DEBUG: API key length: {len(self.openrouter_api_key) if self.openrouter_api_key else 'None'}")
        print(f"DEBUG: API key starts with: {self.openrouter_api_key[:10] if self.openrouter_api_key else 'None'}...")

        # 4. Validate API key
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        # 5. Store base URL
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
    
    def refine_query(self, query: str) -> str:
        """
        Use LLM to improve the search query (optional but helps retrieval).
        
        TODO:
        1. If self.config.refine_query is False, return query unchanged
        
        2. Build the prompt using QUERY_REFINEMENT_PROMPT.format(query=query)
        
        3. Build headers:
           {"Authorization": f"Bearer {self.openrouter_api_key}", "Content-Type": "application/json"}
        
        4. Build payload:
           {
               "model": self.config.refinement_model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.3,
               "max_tokens": 100
           }
        
        5. Make POST request to f"{self.openrouter_base_url}/chat/completions"
        
        6. If request fails, return original query (don't crash)
        
        7. Parse response and extract the refined query from the response
           refined = response_json["choices"][0]["message"]["content"].strip()
        
        8. Return refined query (strip any quotes)
        
        Args:
            query: Original user query
            
        Returns:
            Refined query (or original if refinement disabled/fails)
        """
        if not self.config.refine_query:
            return query

        prompt = QUERY_REFINEMENT_PROMPT.format(query=query)

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.refinement_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 100
        }

        try:
            response = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload
            )

            response.raise_for_status()

            response_json = response.json()

            refined = response_json["choices"][0]["message"]["content"].strip()

            return refined.strip('"')

        except Exception:
            return query
    
    def _format_context(self, results: list[RetrievalResult]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.
        
        TODO:
        1. Build a list of formatted source strings
        
        2. For each result (enumerate with index starting at 1):
           formatted = f'''
           --- Source {i} ---
           Title: {result.title}
           Authors: {result.authors}
           Section: {result.chunk_section}
           
           Content:
           {result.text}
           '''
        
        3. Join all formatted strings with newlines
        
        4. Return the combined context string
        
        Args:
            results: List of RetrievalResult objects
            
        Returns:
            Formatted context string
        """
        formatted_sources = []

        for i, result in enumerate(results, start=1):

            formatted = f"""
                --- Source {i} ---
                Title: {result.title}
                Authors: {result.authors}
                Section: {result.chunk_section}

                Content:
                {result.text}
                """

            formatted_sources.append(formatted)

        return "\n".join(formatted_sources)
    
    def _build_sources_metadata(self, results: list[RetrievalResult]) -> list[dict]:
        """
        Build list of unique source papers for citations.
        The frontend displays these as clickable source links.
        
        TODO:
        1. Create a dict to track seen titles (for deduplication):
           seen = {}
        
        2. For each result:
           - If title not in seen:
             - Add to seen with value:
               {
                   "title": result.title,
                   "authors": result.authors,
                   "pdf_url": result.pdf_url,
                   "github_link": result.github_link,
                   "video_link": result.video_link,
                   "acm_url": result.acm_url,
                   "abstract_url": result.abstract_url,
               }
        
        3. Return list(seen.values())
        
        Args:
            results: List of RetrievalResult objects
            
        Returns:
            List of unique source metadata dicts
        """
            # 1. Track seen titles
        seen = {}
        print(f"DEBUG: seen type: {type(seen)}")
        print(f"DEBUG: seen initial: {seen}")

        # 2. Iterate through results
        for result in results:
            if result.title not in seen:
                seen[result.title] = {
                    "title": result.title,
                    "authors": result.authors,
                    "pdf_url": result.pdf_url,
                    "github_link": result.github_link,
                    "video_link": result.video_link,
                    "acm_url": result.acm_url,
                    "abstract_url": result.abstract_url,
                }

        # 3. Return dictionary of unique sources (so .values() works)
        print(f"DEBUG: seen before values(): {type(seen)} - {seen}")
        try:
            return seen  # Return the dictionary, not list(seen.values())
        except Exception as e:
            print(f"DEBUG: Error in seen.values(): {type(seen)} - {str(e)}")
            raise
    
    def _call_llm(self, query: str, context: str) -> str:
        """
        Call OpenRouter API to generate an answer.
        
        TODO:
        1. Build the user message:
           user_message = f'''Based on the following research paper excerpts, answer this question.

           Question: {query}

           Research Paper Excerpts:
           {context}

           Remember to cite papers using [Paper Title] format.'''
        
        2. Build headers:
           {"Authorization": f"Bearer {self.openrouter_api_key}", "Content-Type": "application/json"}
        
        3. Build payload:
           {
               "model": self.config.llm_model,
               "messages": [
                   {"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": user_message}
               ],
               "temperature": self.config.temperature,
               "max_tokens": self.config.max_tokens
           }
        
        4. Make POST request to f"{self.openrouter_base_url}/chat/completions"
        
        5. Check response status, raise error if not 200
        
        6. Parse response and extract answer from the response
           answer = response_json["choices"][0]["message"]["content"]
        
        7. Return the answer
        
        Args:
            query: User's question
            context: Formatted context from retrieved chunks
            
        Returns:
            Generated answer string
        """
        user_message = f"""Based on the following research paper excerpts, answer this question.

            Question: {query}

            Research Paper Excerpts:
            {context}

            Remember to cite papers using [Paper Title] format.
            """

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

        response = requests.post(
            f"{self.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

        response_json = response.json()

        answer = response_json["choices"][0]["message"]["content"]

        return answer
    
    def generate(self, query: str, top_k: Optional[int] = None, return_sources: bool = True) -> dict:
        """
        Full RAG pipeline - retrieve relevant chunks and generate an answer.
        THIS IS THE MAIN METHOD THAT api_server.py CALLS!
        
        TODO:
        1. Refine the query:
        
        2. Retrieve relevant chunks:
           
        3. Handle empty results:
           if not results:
               return {
                   "query": query,
                   "refined_query": refined,
                   "answer": "I couldn't find any relevant papers to answer this question.",
                   "sources": []
               }
        
        4. Format context from results:
           
        5. Generate answer using LLM:
        
        6. Build and return response dict:
           {
               "query": query,
               "refined_query": refined,
               "answer": answer,
               "sources": self._build_sources_metadata(results) if return_sources else []
           }
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve (uses config default if None)
            return_sources: Whether to include source metadata
            
        Returns:
            Dict with query, refined_query, answer, and sources
        """
        if top_k is None:
            top_k = self.config.retrieval_top_k

        # 1. Refine query
        refined = self.refine_query(query)

        # 2. Retrieve chunks
        results = self.retrieval.retrieve(refined, top_k=top_k)

        # 3. Handle empty results
        if not results:
            return {
                "query": query,
                "refined_query": refined,
                "answer": "I couldn't find any relevant papers to answer this question.",
                "sources": []
            }

        # 4. Format context
        context = self._format_context(results)

        # 5. Generate answer
        answer = self._call_llm(refined, context)

        # 6. Return result
        return {
            "query": query,
            "refined_query": refined,
            "answer": answer,
            "sources": self._build_sources_metadata(results) if return_sources else []
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
