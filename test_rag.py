import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

async def test_rag():
    try:
        # Import and initialize RAG
        from services.rag.rag_pipeline import get_or_create_rag_pipeline, initialize_rag

        print("Initializing RAG pipeline...")
        rag_pipeline = get_or_create_rag_pipeline()
        await initialize_rag()

        print("RAG initialized successfully!")

        # Test retrieval
        from services.rag.retriever import get_retriever
        retriever = get_retriever()

        print("\nTesting retrieval with query: 'Python interview questions'")
        results = retriever.retrieve("Python interview questions", top_k=3)
        print(f"Retrieved {len(results)} documents")

        for i, result in enumerate(results):
            print(f"\nDocument {i+1}:")
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"Category: {result.get('category', 'N/A')}")
            print(f"Content preview: {result.get('content', '')[:200]}...")

        # Test stats
        stats = rag_pipeline.get_stats()
        print("
RAG Stats:")
        print(f"Documents indexed: {stats.get('documents_indexed', 0)}")
        print(f"Cache hits: {stats.get('cache_hits', 0)}")
        print(f"Cache misses: {stats.get('cache_misses', 0)}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rag())