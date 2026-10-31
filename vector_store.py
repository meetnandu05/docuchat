import chromadb
from chromadb.config import Settings
from typing import List, Dict
import os

class VectorStore:
    """Manage vector database for document chunks"""
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB
        
        Args:
            collection_name: Name of the collection to store documents
            persist_directory: Where to save the database
        """
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # When deployed, use ChromaDB's built-in embedder (onnxruntime, ~100MB)
        # Locally, use sentence-transformers (PyTorch, better quality but ~2GB)
        if os.getenv("LLM_PROVIDER") == "groq":
            print("Using ChromaDB built-in embedder (lightweight)")
            self.embedding_model = None
        else:
            print("Loading embedding model...")
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model loaded!")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document chunks for RAG"}
        )
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Convert texts to embeddings
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def add_documents(self, chunks: List[Dict[str, any]], document_name: str):
        """
        Add document chunks to vector database
        
        Args:
            chunks: List of chunks from DocumentProcessor
            document_name: Name of the source document
        """
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Prepare data
        texts = [chunk['text'] for chunk in chunks]
        ids = [f"{document_name}_{chunk['id']}" for chunk in chunks]
        metadatas = [
            {
                'document': document_name,
                'chunk_index': chunk['chunk_index'],
                'start_char': chunk['start_char'],
                'end_char': chunk['end_char']
            }
            for chunk in chunks
        ]
        
        # Add to database (with or without explicit embeddings)
        if self.embedding_model:
            embeddings = self.create_embeddings(texts)
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
        else:
            # ChromaDB generates embeddings automatically
            self.collection.add(
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
        
        print(f"✓ Added {len(chunks)} chunks from {document_name}")
    
    def search(self, query: str, n_results: int = 3) -> Dict:
        """
        Search for relevant chunks
        
        Args:
            query: User's question
            n_results: Number of results to return
            
        Returns:
            Dictionary with results and metadata
        """
        if self.embedding_model:
            # Use explicit embeddings
            query_embedding = self.create_embeddings([query])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
        else:
            # Let ChromaDB handle embedding
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
        
        return results
    
    def delete_document(self, document_name: str):
        """Delete all chunks from a specific document"""
        # Get all IDs for this document
        results = self.collection.get(
            where={"document": document_name}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"✓ Deleted {len(results['ids'])} chunks from {document_name}")
        else:
            print(f"No chunks found for {document_name}")
    
    def list_documents(self) -> List[str]:
        """List all unique documents in the database"""
        all_data = self.collection.get()
        
        if not all_data['metadatas']:
            return []
        
        documents = set(meta['document'] for meta in all_data['metadatas'])
        return list(documents)
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        count = self.collection.count()
        documents = self.list_documents()
        
        return {
            'total_chunks': count,
            'documents': len(documents),
            'document_names': documents
        }


# Test the vector store
if __name__ == "__main__":
    store = VectorStore()
    
    # Example: Add some test data
    test_chunks = [
        {
            'id': 'chunk_0',
            'text': 'FastAPI is a modern web framework for Python.',
            'chunk_index': 0,
            'start_char': 0,
            'end_char': 50
        },
        {
            'id': 'chunk_1',
            'text': 'It is very fast and easy to use for building APIs.',
            'chunk_index': 1,
            'start_char': 50,
            'end_char': 100
        }
    ]
    
    store.add_documents(test_chunks, "test_doc")
    
    # Test search
    results = store.search("What is FastAPI?", n_results=2)
    print("\nSearch Results:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"\n{i+1}. {doc}")
    
    # Get stats
    stats = store.get_stats()
    print(f"\nDatabase Stats: {stats}")