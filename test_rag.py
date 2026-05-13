from rag_engine import RAGEngine

def main():
    # Initialize
    print("="*60)
    print("RAG System Test")
    print("="*60)
    
    rag = RAGEngine()
    
    # Check if we have any documents
    stats = rag.get_stats()
    print(f"\nCurrent database stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Documents: {stats['documents']}")
    
    if stats['documents'] > 0:
        print(f"  Document names: {stats['document_names']}")
    
    # If you have a PDF to test, uncomment and modify:
    # print("\nIngesting document...")
    # rag.ingest_document("path/to/your.pdf", "test_document")
    
    # Test queries
    if stats['total_chunks'] > 0:
        print("\n" + "="*60)
        print("Testing Queries")
        print("="*60)
        
        test_questions = [
            "What is this document about?",
            "What are the main points discussed?",
            "Can you summarize the key findings?"
        ]
        
        for question in test_questions:
            result = rag.query(question, n_results=3, verbose=True)
            input("\nPress Enter to continue to next question...")
    else:
        print("\n⚠️  No documents in database yet!")
        print("Add a document using:")
        print('  rag.ingest_document("your_file.pdf", "doc_name")')

if __name__ == "__main__":
    main()