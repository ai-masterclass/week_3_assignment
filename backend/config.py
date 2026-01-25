from pydantic import BaseModel
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
load_dotenv()


@dataclass
class RetrievalPipelineConfig:
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY")
    qdrant_url: str = os.getenv("QDRANT_URL")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY")
    chunks_path: str = os.getenv("CHUNKS_PATH")
    reranker_type: str = os.getenv("RERANKER_TYPE", "none")
    cross_encoder_model: str = os.getenv("CROSS_ENCODER_MODEL", "")
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL")
    semantic_weight: float = os.getenv("SEMANTIC_WEIGHT", 0.80)
    bm25_weight: float = os.getenv("BM25_WEIGHT", 0.20)
    use_reranker: bool = os.getenv("USE_RERANKER", True)

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
class GenerationConfig:
    """Configuration for the RAG generator."""
    llm_provider: str = "openai"
    use_reranker: bool = True
    llm_model: str = "gpt-4o-mini"  # Model to use for answer generation
    temperature: float = 0.1  # Low temperature for factual answers
    max_tokens: int = 2000  # Max length of generated answer
    openrouter_api_key: Optional[str] = None  # Will load from env if not set
    refine_query: bool = True  # Whether to refine queries before retrieval
    refinement_model: str = "openai/gpt-3.5-turbo"  # Cheaper model for refinement
    retrieval_top_k: int = 8  # Number of chunks to retrieve