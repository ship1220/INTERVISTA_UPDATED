# rag/retriever.py
# Retriever for RAG system - fetches relevant documents

import asyncio
from typing import List, Dict, Any, Optional
from .vector_store import VectorStore, Document
from utils.logger import Logger

logger = Logger(__name__)


class RetrievalResult:
    """Single retrieval result."""
    
    def __init__(self, doc_id: str, content: str, score: float, metadata: Dict = None):
        self.doc_id = doc_id
        self.content = content
        self.score = score  # 0-100
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata
        }


class Retriever:
    """
    Document retriever for RAG pipeline.
    
    Searches vector store and returns ranked results.
    """
    
    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store or VectorStore()
        logger.info("Retriever initialized")
    
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 30.0
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum relevance score (0-100)
        
        Returns:
            List of RetrievalResult ranked by relevance
        """
        
        if self.vector_store.size() == 0:
            logger.warning("Vector store is empty")
            return []
        
        try:
            # Search vector store
            results = await asyncio.to_thread(
                self.vector_store.search,
                query,
                k
            )
            
            # Convert to RetrievalResult
            retrieval_results = []
            
            for doc_id, score in results:
                if score < score_threshold:
                    continue
                
                doc = self.vector_store.get_document(doc_id)
                if doc:
                    result = RetrievalResult(
                        doc_id=doc_id,
                        content=doc.content,
                        score=score,
                        metadata=doc.metadata
                    )
                    retrieval_results.append(result)
            
            logger.info(f"Retrieved {len(retrieval_results)} documents for: '{query[:50]}...'")
            return retrieval_results
        
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise
    
    async def retrieve_context(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 30.0,
        join_with: str = "\n\n"
    ) -> str:
        """
        Retrieve documents and join as context string.
        
        Args:
            query: Search query
            k: Number of results
            score_threshold: Minimum relevance score
            join_with: String to join documents with
        
        Returns:
            Context string for LLM augmentation
        """
        
        results = await self.retrieve(query, k, score_threshold)
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            part = f"""[Document {i} - Relevance: {result.score:.0f}%]
{result.content}

Source: {result.metadata.get('source', 'unknown')}"""
            context_parts.append(part)
        
        return join_with.join(context_parts)
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Index multiple documents.
        
        Args:
            documents: List of {"content": ..., "metadata": {...}, "doc_id": ...}
        
        Returns:
            List of added document IDs
        """
        
        docs_to_add = []
        
        for doc in documents:
            doc_id = doc.get("doc_id", f"doc_{len(self.vector_store.documents)}")
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            docs_to_add.append((doc_id, content, metadata))
        
        doc_ids = self.vector_store.add_documents_batch(docs_to_add)
        logger.info(f"Indexed {len(doc_ids)} documents")
        
        return doc_ids
    
    def add_document(
        self,
        content: str,
        doc_id: str = None,
        metadata: Dict = None
    ) -> str:
        """
        Index single document.
        
        Args:
            content: Document text
            doc_id: Document ID (auto-generated if None)
            metadata: Optional metadata
        
        Returns:
            Document ID
        """
        
        if doc_id is None:
            doc_id = f"doc_{self.vector_store.size()}"
        
        result_id = self.vector_store.add_document(doc_id, content, metadata)
        logger.debug(f"Document indexed: {result_id}")
        
        return result_id
    
    def add_from_file(self, file_path: str, metadata: Dict = None) -> str:
        """
        Index document from file.
        
        Args:
            file_path: Path to text file
            metadata: Optional metadata
        
        Returns:
            Document ID
        """
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc_id = self.add_document(
                content=content,
                metadata=metadata or {"source": file_path}
            )
            
            logger.info(f"Indexed file: {file_path}")
            return doc_id
        
        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {str(e)}")
            raise
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        return {
            "vector_store_stats": self.vector_store.stats(),
            "retriever_type": "faiss_based"
        }


# Singleton retriever instance
_retriever_instance = None


def get_retriever() -> Retriever:
    """Get or create singleton retriever."""
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    
    return _retriever_instance
