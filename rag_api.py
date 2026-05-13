from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from rag_engine import RAGEngine

app = FastAPI(title="DocuChat API", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag = RAGEngine()

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    n_results: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

class DocumentInfo(BaseModel):
    name: str
    chunks: int

@app.get("/")
def read_root():
    return {
        "message": "DocuChat API - RAG-powered document Q&A",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "query": "/query",
            "documents": "/documents",
            "stats": "/stats"
        }
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process and ingest
        document_name = file.filename.replace('.pdf', '')
        rag.ingest_document(file_path, document_name)
        
        # Get stats
        stats = rag.get_stats()
        
        return {
            "message": f"Document '{file.filename}' uploaded and processed successfully",
            "document_name": document_name,
            "total_documents": stats['documents'],
            "total_chunks": stats['total_chunks']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Query the document database
    """
    try:
        result = rag.query(request.question, n_results=request.n_results, verbose=False)
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/documents")
def list_documents():
    """
    List all uploaded documents
    """
    documents = rag.list_documents()
    stats = rag.get_stats()
    
    return {
        "documents": documents,
        "count": len(documents),
        "total_chunks": stats['total_chunks']
    }

@app.delete("/documents/{document_name}")
def delete_document(document_name: str):
    """
    Delete a document from the database
    """
    try:
        rag.delete_document(document_name)
        return {"message": f"Document '{document_name}' deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.get("/stats")
def get_stats():
    """
    Get system statistics
    """
    return rag.get_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)