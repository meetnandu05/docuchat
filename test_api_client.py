import requests

BASE_URL = "http://localhost:8000"

def test_upload():
    """Test document upload"""
    print("Testing document upload...")
    
    # Replace with your PDF path
    pdf_path = "path/to/your/document.pdf"
    
    with open(pdf_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()

def test_query(question: str):
    """Test querying"""
    print(f"\nTesting query: {question}")
    
    response = requests.post(
        f"{BASE_URL}/query",
        json={"question": question, "n_results": 3}
    )
    
    result = response.json()
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources:")
    for i, source in enumerate(result['sources']):
        print(f"{i+1}. {source['document']} (relevance: {source['relevance']})")
        print(f"   {source['text'][:100]}...")

def test_list_documents():
    """Test listing documents"""
    print("\nListing documents...")
    
    response = requests.get(f"{BASE_URL}/documents")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    # Test upload (uncomment and provide PDF path)
    # test_upload()
    
    # Test queries
    test_query("What is this document about?")
    test_query("What are the main points?")
    
    # List documents
    test_list_documents()