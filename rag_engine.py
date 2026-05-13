from document_processor import DocumentProcessor
from vector_store import VectorStore
import ollama
from typing import List, Dict

class RAGEngine:
    """Complete RAG pipeline: process → store → retrieve → generate"""
    
    def __init__(self, model_name: str = "llama3.2"):
        """
        Initialize RAG engine
        
        Args:
            model_name: Ollama model to use for generation
        """
        self.processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        self.vector_store = VectorStore()
        self.model_name = model_name
        
        print(f"RAG Engine initialized with model: {model_name}")
    
    def ingest_document(self, pdf_path: str, document_name: str):
        """
        Process and store a document
        
        Args:
            pdf_path: Path to PDF file
            document_name: Unique name for this document
        """
        print(f"\n{'='*60}")
        print(f"Ingesting document: {document_name}")
        print(f"{'='*60}")
        
        # Process PDF into chunks
        chunks = self.processor.process_pdf(pdf_path)
        
        # Store in vector database
        self.vector_store.add_documents(chunks, document_name)
        
        print(f"✓ Document '{document_name}' successfully ingested!")
        print(f"{'='*60}\n")
    
    def retrieve_context(self, query: str, n_results: int = 3) -> tuple[List[str], List[Dict]]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            
        Returns:
            Tuple of (context_texts, metadata)
        """
        results = self.vector_store.search(query, n_results=n_results)
        
        contexts = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        # Add distance scores to metadata
        for i, meta in enumerate(metadatas):
            meta['relevance_score'] = round(1 - (distances[i] / 2), 3)  # Convert to 0-1 scale
        
        return contexts, metadatas
    
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """
        Generate answer using LLM with retrieved context
        
        Args:
            query: User's question
            contexts: Retrieved document chunks
            
        Returns:
            Generated answer
        """
        # Build prompt with context
        context_text = "\n\n".join([
            f"[Context {i+1}]\n{ctx}"
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""You are a helpful assistant answering questions about documents.

Use the following context to answer the question. If the answer cannot be found in the context, say so clearly.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER (be concise and cite which context sections you used):"""
        
        # Generate response
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that answers questions based on provided context. Always cite your sources.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.3  # Lower temperature for more factual responses
            }
        )
        
        return response['message']['content']
    
    def query(self, question: str, n_results: int = 3, verbose: bool = True) -> Dict:
        """
        Complete RAG query: retrieve + generate
        
        Args:
            question: User's question
            n_results: Number of contexts to retrieve
            verbose: Print detailed information
            
        Returns:
            Dictionary with answer, contexts, and metadata
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"{'='*60}\n")
        
        # Step 1: Retrieve relevant contexts
        if verbose:
            print("🔍 Searching for relevant information...")
        
        contexts, metadatas = self.retrieve_context(question, n_results)
        
        if not contexts:
            return {
                'answer': "I don't have any relevant information to answer this question.",
                'contexts': [],
                'metadatas': [],
                'sources': []
            }
        
        if verbose:
            print(f"✓ Found {len(contexts)} relevant chunks\n")
            for i, (ctx, meta) in enumerate(zip(contexts, metadatas)):
                print(f"Context {i+1} (relevance: {meta['relevance_score']}):")
                print(f"  Document: {meta['document']}")
                print(f"  Preview: {ctx[:100]}...")
                print()
        
        # Step 2: Generate answer
        if verbose:
            print("🤖 Generating answer...\n")
        
        answer = self.generate_answer(question, contexts)
        
        if verbose:
            print(f"{'='*60}")
            print(f"Answer:\n{answer}")
            print(f"{'='*60}\n")
        
        # Prepare sources for citation
        sources = [
            {
                'document': meta['document'],
                'chunk_index': meta['chunk_index'],
                'relevance': meta['relevance_score'],
                'text': ctx[:200] + "..." if len(ctx) > 200 else ctx
            }
            for ctx, meta in zip(contexts, metadatas)
        ]
        
        return {
            'answer': answer,
            'contexts': contexts,
            'metadatas': metadatas,
            'sources': sources
        }
    
    def list_documents(self) -> List[str]:
        """List all ingested documents"""
        return self.vector_store.list_documents()
    
    def delete_document(self, document_name: str):
        """Remove a document from the system"""
        self.vector_store.delete_document(document_name)
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return self.vector_store.get_stats()


# Interactive test
if __name__ == "__main__":
    print("Initializing RAG Engine...")
    rag = RAGEngine()
    
    # Example: Ingest a document
    # rag.ingest_document("path/to/your.pdf", "my_document")
    
    # Example: Query
    # result = rag.query("What are the main points?")
    
    # Show stats
    stats = rag.get_stats()
    print(f"\nSystem Stats: {stats}")