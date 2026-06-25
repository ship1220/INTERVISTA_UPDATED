#!/usr/bin/env python3
"""Simple RAG test script"""

import asyncio
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

async def main():
    try:
        print("Testing RAG system...")

        # Import RAG components
        from services.rag.rag_config import get_rag_config
        from services.rag.vector_store import VectorStore
        from services.rag.retriever import Retriever
        from services.rag.seed_data import format_seed_data_for_ingestion

        print("✓ Imports successful")

        # Test config
        config = get_rag_config()
        print(f"✓ Config loaded: embedding_dim={config.embedding_dim}")

        # Test vector store
        vector_store = VectorStore(
            embedding_dim=config.embedding_dim,
            index_type=config.vector_store_type
        )
        print("✓ Vector store initialized")

        # Test seed data
        seed_data = format_seed_data_for_ingestion()
        print(f"✓ Seed data loaded: {len(seed_data)} documents")

        # Test retriever
        retriever = Retriever(vector_store)
        print("✓ Retriever initialized")

        # Test document ingestion
        from services.rag.document_ingester import DocumentIngester
        ingester = DocumentIngester()

        # Ingest first document as test
        if seed_data:
            test_doc = seed_data[0]
            await ingester.ingest_document(test_doc, vector_store)
            print("✓ Document ingestion successful")

        # Test retrieval
        results = retriever.retrieve("Python programming", top_k=2)
        print(f"✓ Retrieval successful: {len(results)} results")

        print("\n🎉 RAG system test PASSED!")

    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())