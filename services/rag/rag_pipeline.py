# rag/rag_pipeline.py
# Main RAG pipeline orchestration and coordination

import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from utils.logger import Logger
from .retriever import Retriever, RetrievalResult
from .vector_store import VectorStore, Document
from .rag_config import (
    RAGConfig, DocumentCategory, RetrievalStrategy,
    RetrievalContext, get_rag_config
)
from .document_ingester import DocumentIngester, DocumentValidator
from .seed_data import format_seed_data_for_ingestion

logger = Logger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline coordinating ingestion, retrieval, and context augmentation.
    """
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or get_rag_config()
        self.vector_store = VectorStore(
            embedding_dim=self.config.embedding_dim,
            index_type=self.config.vector_store_type
        )
        self.retriever = Retriever(self.vector_store)
        self.document_ingester = DocumentIngester()
        self.initialized = False
        
        # Caching for retrieved contexts
        self._context_cache: Dict[str, Tuple[str, datetime]] = {}
        
        # Statistics tracking
        self.stats = {
            "documents_indexed": 0,
            "retrievals_performed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_retrieval_time": 0.0,
            "total_retrieval_time": 0.0,
        }
        
        logger.info("RAGPipeline initialized")
    
    # =====================================================================
    # INITIALIZATION
    # =====================================================================
    
    async def initialize(self):
        """Initialize pipeline with seed data."""
        
        if self.initialized:
            logger.warning("RAGPipeline already initialized")
            return
        
        logger.info("Initializing RAGPipeline with seed data...")
        
        try:
            # Get seed data
            seed_data = format_seed_data_for_ingestion()
            logger.info(f"Loaded {len(seed_data)} seed documents")
            
            # Ingest documents
            await self._ingest_batch(seed_data)
            
            self.initialized = True
            logger.info("RAGPipeline initialization complete")
        
        except Exception as e:
            logger.error(f"RAGPipeline initialization failed: {str(e)}")
            raise
    
    async def _ingest_batch(self, documents: List[Dict]) -> int:
        """Ingest batch of documents asynchronously."""
        
        # Validate documents
        valid_docs = []
        for doc in documents:
            is_valid, error = DocumentValidator.validate_document(doc)
            if not is_valid:
                logger.warning(f"Skipping invalid document: {error}")
                continue
            valid_docs.append(doc)
        
        # Ingest in parallel
        ingested_docs = await asyncio.to_thread(
            self.document_ingester.ingest_documents_batch,
            valid_docs
        )
        
        # Add to vector store
        docs_to_add = []
        for doc in ingested_docs:
            doc_id = f"{doc['metadata'].get('category', 'unknown')}_{len(self.vector_store.documents)}"
            docs_to_add.append((
                doc_id,
                doc["content"],
                doc["metadata"]
            ))
        
        added_ids = self.vector_store.add_documents_batch(docs_to_add)
        
        self.stats["documents_indexed"] += len(added_ids)
        
        logger.info(f"Ingested {len(added_ids)} documents into vector store")
        
        return len(added_ids)
    
    # =====================================================================
    # RETRIEVAL & CONTEXT AUGMENTATION
    # =====================================================================
    
    async def retrieve_context(
        self,
        query: str,
        context: RetrievalContext = None,
        use_cache: bool = True
    ) -> str:
        """
        Retrieve and format context for a query.
        
        Args:
            query: Search query
            context: RetrievalContext with metadata/filters
            use_cache: Use cached results
        
        Returns:
            Formatted context string for LLM augmentation
        """
        
        import time
        start_time = time.time()
        
        # Create default context if not provided
        if context is None:
            context = RetrievalContext(query=query)
        
        # Check cache
        cache_key = self._make_cache_key(query, context)
        if use_cache and cache_key in self._context_cache:
            cached_context, cached_time = self._context_cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self.config.cache_ttl_seconds):
                self.stats["cache_hits"] += 1
                logger.debug(f"Cache hit for: {query[:50]}...")
                return cached_context
        
        self.stats["cache_misses"] += 1
        
        try:
            # Perform retrieval based on strategy
            results = await self._retrieve_by_strategy(query, context)
            
            # Format results
            formatted_context = self._format_context(results, context)
            
            # Cache result
            if self.config.cache_retrieved_context:
                self._context_cache[cache_key] = (formatted_context, datetime.now())
            
            # Record statistics
            elapsed = time.time() - start_time
            self.stats["retrievals_performed"] += 1
            self.stats["total_retrieval_time"] += elapsed
            self.stats["avg_retrieval_time"] = (
                self.stats["total_retrieval_time"] / self.stats["retrievals_performed"]
            )
            
            if self.config.log_retrieval_stats:
                logger.info(
                    f"Retrieved {len(results)} docs in {elapsed:.3f}s for: {query[:50]}..."
                )
            
            return formatted_context
        
        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            return ""
    
    async def _retrieve_by_strategy(
        self,
        query: str,
        context: RetrievalContext
    ) -> List[RetrievalResult]:
        """Retrieve results based on strategy."""
        
        if self.config.strategy == RetrievalStrategy.SEMANTIC:
            return await self._retrieve_semantic(query, context)
        
        elif self.config.strategy == RetrievalStrategy.HYBRID:
            return await self._retrieve_hybrid(query, context)
        
        elif self.config.strategy == RetrievalStrategy.CATEGORY_BASED:
            return await self._retrieve_category_based(query, context)
        
        else:
            return await self._retrieve_semantic(query, context)
    
    async def _retrieve_semantic(
        self,
        query: str,
        context: RetrievalContext
    ) -> List[RetrievalResult]:
        """Pure semantic retrieval."""
        
        k = context.k or self.config.default_k
        threshold = context.min_score or self.config.score_threshold
        
        results = await self.retriever.retrieve(
            query=query,
            k=k,
            score_threshold=threshold
        )
        
        return results
    
    async def _retrieve_hybrid(
        self,
        query: str,
        context: RetrievalContext
    ) -> List[RetrievalResult]:
        """Hybrid retrieval with semantic + metadata filtering."""
        
        # Semantic retrieval
        k = (context.k or self.config.default_k) * 2  # Get more for filtering
        results = await self._retrieve_semantic(query, context)
        
        # Apply metadata filters
        if self.config.enable_metadata_filtering:
            filters = context.get_filter_metadata()
            results = self._filter_by_metadata(results, filters)
        
        # Re-rank if enabled
        if self.config.enable_reranking:
            results = self._rerank_results(results, query)
        
        # Return top-k
        return results[:context.k or self.config.default_k]
    
    async def _retrieve_category_based(
        self,
        query: str,
        context: RetrievalContext
    ) -> List[RetrievalResult]:
        """Category-specific retrieval."""
        
        if not context.category:
            return await self._retrieve_semantic(query, context)
        
        # Get category-specific k
        k = self.config.category_retrieval_k.get(
            context.category,
            self.config.default_k
        )
        
        # Retrieve with category metadata
        context.metadata_filters["category"] = context.category.value
        
        return await self._retrieve_hybrid(query, context)
    
    def _filter_by_metadata(
        self,
        results: List[RetrievalResult],
        filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Filter results by metadata."""
        
        filtered = []
        
        for result in results:
            doc = self.vector_store.get_document(result.doc_id)
            if not doc:
                continue
            
            match = True
            for key, value in filters.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            
            if match:
                filtered.append(result)
        
        return filtered
    
    def _rerank_results(
        self,
        results: List[RetrievalResult],
        query: str
    ) -> List[RetrievalResult]:
        """Re-rank results using semantic similarity boost."""
        
        # Simple re-ranking: boost results with query keywords
        query_tokens = set(query.lower().split())
        
        for result in results:
            doc = self.vector_store.get_document(result.doc_id)
            if doc:
                content_tokens = set(doc.content.lower().split())
                overlap = len(query_tokens & content_tokens)
                result.score = result.score * (1 + overlap * 0.1)
        
        # Re-sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results
    
    def _format_context(
        self,
        results: List[RetrievalResult],
        context: RetrievalContext = None
    ) -> str:
        """Format retrieval results as context string."""
        
        if not results:
            return ""
        
        parts = []
        total_length = 0
        max_length = self.config.max_context_length
        
        parts.append("## Retrieved Context\n")
        
        for i, result in enumerate(results, 1):
            part = f"""### Document {i} (Relevance: {result.score:.0f}%)
{result.content}
"""
            
            # Check length limit
            part_length = len(part)
            if total_length + part_length > max_length:
                # Truncate
                remaining = max_length - total_length
                if remaining > 100:  # Only add if meaningful size
                    part = part[:remaining] + "..."
                    parts.append(part)
                break
            
            parts.append(part)
            total_length += part_length
        
        return self.config.context_separator.join(parts)
    
    # =====================================================================
    # CACHE MANAGEMENT
    # =====================================================================
    
    def _make_cache_key(
        self,
        query: str,
        context: RetrievalContext
    ) -> str:
        """Generate cache key for query + context."""
        
        key_parts = [
            query,
            context.category.value if context.category else "none",
            context.role or "none",
            context.level or "none",
            str(context.k or self.config.default_k),
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear retrieval cache."""
        self._context_cache.clear()
        logger.info("Cache cleared")
    
    # =====================================================================
    # STATISTICS & MONITORING
    # =====================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        
        return {
            "initialized": self.initialized,
            "vector_store": self.vector_store.stats(),
            "retrieval_stats": self.stats.copy(),
            "cache_size": len(self._context_cache),
            "ingestion_stats": self.document_ingester.get_stats(),
        }
    
    def print_stats(self):
        """Print formatted statistics."""
        
        stats = self.get_stats()
        
        logger.info("=== RAG PIPELINE STATISTICS ===")
        logger.info(f"Initialized: {stats['initialized']}")
        logger.info(f"Documents Indexed: {stats['vector_store']['total_documents']}")
        logger.info(f"Retrievals: {stats['retrieval_stats']['retrievals_performed']}")
        logger.info(f"Cache Hits: {stats['retrieval_stats']['cache_hits']}")
        logger.info(f"Cache Misses: {stats['retrieval_stats']['cache_misses']}")
        logger.info(f"Avg Retrieval Time: {stats['retrieval_stats']['avg_retrieval_time']:.3f}s")
        logger.info(f"Cache Size: {stats['cache_size']}")
    
    # =====================================================================
    # DOCUMENT MANAGEMENT
    # =====================================================================
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        category: DocumentCategory
    ) -> List[str]:
        """Add new documents to RAG system."""
        
        for doc in documents:
            doc["category"] = category.value
        
        return await self._ingest_batch(documents)
    
    async def add_document(
        self,
        content: str,
        category: DocumentCategory,
        metadata: Dict = None
    ) -> str:
        """Add single document."""
        
        doc = {
            "content": content,
            "category": category.value,
            "metadata": metadata or {}
        }
        
        added = await self._ingest_batch([doc])
        return (added[0] if added else None)
    
    # =====================================================================
    # PERSISTENCE
    # =====================================================================
    
    def save(self, path: str):
        """Save vector store to disk."""
        
        try:
            self.vector_store.save(path)
            logger.info(f"RAG pipeline saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save RAG pipeline: {str(e)}")
    
    def load(self, path: str):
        """Load vector store from disk."""
        
        try:
            self.vector_store.load(path)
            self.initialized = True
            logger.info(f"RAG pipeline loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load RAG pipeline: {str(e)}")


# Global pipeline instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_or_create_rag_pipeline(config: RAGConfig = None) -> RAGPipeline:
    """Get or create global RAG pipeline."""
    
    global _rag_pipeline
    
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(config)
    
    return _rag_pipeline


async def initialize_rag() -> RAGPipeline:
    """Initialize global RAG pipeline."""
    
    pipeline = get_or_create_rag_pipeline()
    await pipeline.initialize()
    return pipeline
