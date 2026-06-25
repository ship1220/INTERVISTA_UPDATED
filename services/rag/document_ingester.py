# rag/document_ingester.py
# Load, process, and prepare documents for RAG system

import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from utils.logger import Logger
from .rag_config import DocumentCategory

logger = Logger(__name__)


class DocumentProcessor(ABC):
    """Base class for document processors."""
    
    @abstractmethod
    def process(self, raw_text: str, metadata: Dict) -> List[Dict[str, Any]]:
        """
        Process raw text into document chunks.
        
        Returns:
            List of {"content": str, "metadata": dict} documents
        """
        pass


class ChunkProcessor(DocumentProcessor):
    """Chunk large documents by character limit or paragraphs."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def process(self, raw_text: str, metadata: Dict) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        
        paragraphs = raw_text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": metadata.copy()
                    })
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": metadata.copy()
            })
        
        return chunks


class JSONDocumentProcessor(DocumentProcessor):
    """Process JSON documents with predefined structure."""
    
    def process(self, raw_text: str, metadata: Dict) -> List[Dict[str, Any]]:
        """Parse JSON documents."""
        
        try:
            data = json.loads(raw_text)
            
            if isinstance(data, list):
                # Array of documents
                docs = []
                for i, item in enumerate(data):
                    doc_metadata = metadata.copy()
                    doc_metadata["index"] = i
                    docs.append({
                        "content": json.dumps(item),
                        "metadata": doc_metadata
                    })
                return docs
            
            elif isinstance(data, dict):
                # Single document
                return [{
                    "content": raw_text,
                    "metadata": metadata
                }]
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            return [{
                "content": raw_text,
                "metadata": metadata
            }]


class DocumentIngester:
    """Main ingester for loading and preparing documents."""
    
    def __init__(self, processor: DocumentProcessor = None):
        self.processor = processor or ChunkProcessor()
        self.ingestion_stats = {
            "total_loaded": 0,
            "total_chunks": 0,
            "categories": {}
        }
    
    def ingest_text(
        self,
        text: str,
        category: DocumentCategory,
        source_id: str = "unknown",
        additional_metadata: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        Ingest plain text document.
        
        Args:
            text: Document content
            category: Document category
            source_id: Source identifier
            additional_metadata: Extra metadata to attach
        
        Returns:
            Processed document chunks
        """
        
        metadata = {
            "category": category.value,
            "source_id": source_id,
            "source_type": "text"
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        docs = self.processor.process(text, metadata)
        
        self.ingestion_stats["total_loaded"] += 1
        self.ingestion_stats["total_chunks"] += len(docs)
        
        if category.value not in self.ingestion_stats["categories"]:
            self.ingestion_stats["categories"][category.value] = 0
        self.ingestion_stats["categories"][category.value] += len(docs)
        
        logger.debug(f"Ingested {len(docs)} chunks from {source_id}")
        
        return docs
    
    def ingest_json_file(
        self,
        file_path: str,
        category: DocumentCategory,
        metadata_key_mapping: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """
        Ingest documents from JSON file.
        
        Args:
            file_path: Path to JSON file
            category: Document category
            metadata_key_mapping: Map JSON fields to metadata fields
        
        Returns:
            Processed documents
        """
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = {
                "category": category.value,
                "source_id": Path(file_path).stem,
                "source_type": "json_file",
                "file_path": file_path
            }
            
            docs = self.processor.process(content, metadata)
            
            self.ingestion_stats["total_loaded"] += 1
            self.ingestion_stats["total_chunks"] += len(docs)
            
            if category.value not in self.ingestion_stats["categories"]:
                self.ingestion_stats["categories"][category.value] = 0
            self.ingestion_stats["categories"][category.value] += len(docs)
            
            logger.info(f"Ingested {len(docs)} chunks from {file_path}")
            
            return docs
        
        except Exception as e:
            logger.error(f"Failed to ingest JSON file {file_path}: {str(e)}")
            return []
    
    def ingest_documents_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Ingest batch of documents with structured format.
        
        Args:
            documents: List of {"content": str, "category": str, "metadata": dict}
        
        Returns:
            All processed chunks
        """
        
        all_docs = []
        
        for doc in documents:
            try:
                category = DocumentCategory(doc.get("category"))
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                source_id = doc.get("source_id", "batch_item")
                
                processed = self.ingest_text(
                    content,
                    category,
                    source_id,
                    metadata
                )
                
                all_docs.extend(processed)
            
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipped invalid document: {str(e)}")
                continue
        
        return all_docs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return self.ingestion_stats.copy()


class DocumentValidator:
    """Validate documents before ingestion."""
    
    @staticmethod
    def validate_document(doc: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate document structure.
        
        Returns:
            (is_valid, error_message)
        """
        
        if not isinstance(doc, dict):
            return False, "Document must be dict"
        
        if "content" not in doc:
            return False, "Document missing 'content' field"
        
        if not isinstance(doc["content"], str) or not doc["content"].strip():
            return False, "Document content must be non-empty string"
        
        if "category" in doc:
            try:
                DocumentCategory(doc["category"])
            except ValueError:
                return False, f"Invalid category: {doc.get('category')}"
        
        if "metadata" in doc and not isinstance(doc["metadata"], dict):
            return False, "Document metadata must be dict"
        
        return True, ""


# Convenience functions
async def ingest_documents_async(
    documents: List[Dict[str, Any]],
    num_workers: int = 2
) -> List[Dict[str, Any]]:
    """Process documents asynchronously."""
    
    ingester = DocumentIngester()
    
    # Split into batches
    batch_size = max(1, len(documents) // num_workers)
    batches = [
        documents[i:i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]
    
    # Process batches concurrently
    results = await asyncio.gather(*[
        asyncio.to_thread(ingester.ingest_documents_batch, batch)
        for batch in batches
    ])
    
    # Flatten results
    all_docs = []
    for batch_results in results:
        all_docs.extend(batch_results)
    
    return all_docs
