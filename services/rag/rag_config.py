# rag/rag_config.py
# Configuration for RAG system behavior and knowledge base organization

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class DocumentCategory(str, Enum):
    """Document categories for targeted retrieval."""
    INTERVIEW_QUESTIONS = "interview_questions"
    EVALUATION_RUBRICS = "evaluation_rubrics"
    SKILL_FRAMEWORKS = "skill_frameworks"
    COURSE_MATERIALS = "course_materials"
    BEST_PRACTICES = "best_practices"
    TECHNICAL_CONTENT = "technical_content"
    COMMUNICATION_CONTENT = "communication_content"


class RetrievalStrategy(str, Enum):
    """Strategies for context retrieval."""
    SEMANTIC = "semantic"          # Pure semantic similarity
    HYBRID = "hybrid"              # Semantic + metadata filtering
    CATEGORY_BASED = "category"    # Category-specific retrieval


@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    
    # Vector store settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    vector_store_type: str = "faiss"
    
    # Retrieval settings
    default_k: int = 5                          # Number of documents to retrieve
    score_threshold: float = 30.0               # Minimum similarity score
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    
    # Context settings
    max_context_length: int = 2000              # Max characters in retrieved context
    context_separator: str = "\n\n---\n\n"     # Context chunk separator
    
    # Per-category retrieval settings
    category_retrieval_k: Dict[DocumentCategory, int] = field(default_factory=lambda: {
        DocumentCategory.INTERVIEW_QUESTIONS: 3,
        DocumentCategory.EVALUATION_RUBRICS: 4,
        DocumentCategory.SKILL_FRAMEWORKS: 3,
        DocumentCategory.COURSE_MATERIALS: 5,
        DocumentCategory.BEST_PRACTICES: 3,
        DocumentCategory.TECHNICAL_CONTENT: 4,
        DocumentCategory.COMMUNICATION_CONTENT: 4,
    })
    
    # Role/level specific retrieval
    role_level_boost: Dict[str, float] = field(default_factory=lambda: {
        "junior": 1.0,
        "mid": 1.2,
        "senior": 1.5,
        "lead": 1.8,
    })
    
    # Enable/disable RAG features
    enable_interview_rag: bool = True
    enable_evaluation_rag: bool = True
    enable_course_rag: bool = True
    enable_metadata_filtering: bool = True
    enable_reranking: bool = True
    
    # Caching settings
    cache_retrieved_context: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    
    # Performance settings
    batch_processing: bool = True
    num_workers: int = 2
    embedding_batch_size: int = 32
    
    # Logging settings
    log_retrieval_stats: bool = True
    log_context_usage: bool = False
    depth_trace: bool = False
    
    # Persistence
    save_vector_store: bool = True
    vector_store_path: str = "data/vector_store.json"


@dataclass
class RetrievalContext:
    """Context for retrieval operation."""
    
    query: str
    category: Optional[DocumentCategory] = None
    role: Optional[str] = None
    level: Optional[str] = None
    k: Optional[int] = None  # Override default_k
    metadata_filters: Dict[str, any] = field(default_factory=dict)
    min_score: Optional[float] = None  # Override default threshold
    
    def get_filter_metadata(self) -> Dict:
        """Build metadata filter dict."""
        filters = self.metadata_filters.copy()
        
        if self.category:
            filters["category"] = self.category.value
        if self.role:
            filters["role"] = self.role
        if self.level:
            filters["level"] = self.level
        
        return filters


# Default RAG configuration
DEFAULT_RAG_CONFIG = RAGConfig()


def get_rag_config() -> RAGConfig:
    """Get or create RAG configuration."""
    return DEFAULT_RAG_CONFIG


def update_rag_config(**kwargs) -> RAGConfig:
    """Update RAG configuration with kwargs."""
    config = get_rag_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
