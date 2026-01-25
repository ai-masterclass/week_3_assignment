"""Mock RAG Generator for testing API integration."""
import os
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions about SIGGRAPH 2025 papers."""

@dataclass
class GenerationConfig:
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    retrieval_top_k: int = 8
    refine_query: bool = True
    use_reranker: bool = True
    temperature: float = 0.3
    max_tokens: int = 1000

class MockResult:
    def __init__(self, paper_id: str):
        self.paper_id = paper_id
        self.metadata = {
            "title": f"Sample Paper {paper_id}",
            "authors": "Author A, Author B",
            "pdf_url": "https://example.com/paper.pdf"
        }

class MockRetrieval:
    def retrieve(self, query: str, top_k: int = 8) -> List[MockResult]:
        """Mock retrieval - returns fake results."""
        return [MockResult(f"paper_{i}") for i in range(top_k)]

class MockLLMClient:
    class MockChoices:
        class MockDelta:
            def __init__(self, content: str):
                self.content = content
        
        def __init__(self, content: str):
            self.delta = self.MockDelta(content)
    
    class MockChunk:
        def __init__(self, content: str):
            self.choices = [MockLLMClient.MockChoices(content)]
    
    class Chat:
        class Completions:
            @staticmethod
            def create(*args, **kwargs):
                """Mock streaming response."""
                if kwargs.get('stream'):
                    answer = "This is a mock answer for testing. The RAG pipeline will be implemented later."
                    for word in answer.split():
                        yield MockLLMClient.MockChunk(word + " ")
                        time.sleep(0.05)
        
        completions = Completions()
    
    chat = Chat()

class RAGGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.retrieval = MockRetrieval()
        self.llm_client = MockLLMClient()
    
    def refine_query(self, query: str) -> str:
        """Mock query refinement."""
        return f"{query} (refined)"
    
    def _format_context(self, results: List[MockResult]) -> str:
        """Mock context formatting."""
        return "\n".join([f"[{r.metadata['title']}]: Sample content" for r in results])
    
    def _build_sources_metadata(self, results: List[MockResult]) -> Dict[str, Dict[str, Any]]:
        """Mock sources metadata."""
        sources = {}
        for r in results:
            sources[r.paper_id] = r.metadata
        return sources
    
    def generate(self, query: str, top_k: int = 8, return_sources: bool = True) -> Dict[str, Any]:
        """Mock generation."""
        results = self.retrieval.retrieve(query, top_k)
        
        return {
            "query": query,
            "refined_query": self.refine_query(query),
            "answer": "This is a mock answer for testing the API integration.",
            "sources": list(self._build_sources_metadata(results).values()) if return_sources else []
        }
