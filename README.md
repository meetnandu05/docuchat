# DocuChat - AI-Powered Document Q&A

Building a production-ready AI application from scratch using 100% free tools.

📖 **Blog Series:** [Building Production-Ready AI Applications](https://medium.com/@meetnandu996)

## What is DocuChat?

An AI-powered application that lets you upload PDF documents and ask questions about them using local LLMs.

## Current Progress

- ✅ **Part 1**: Local LLM setup with Ollama + FastAPI
- ✅ **Part 2**: Advanced prompt engineering with streaming
- ✅ **Part 3**: RAG system with ChromaDB and document processing
- 🔄 **Part 4**: Frontend interface (coming soon)

## Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed
- 8GB+ RAM recommended

### Installation

1. Clone the repository:
```bash
git clone https://github.com/meetnandu05/docuchat.git
cd docuchat
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama and pull model:
```bash
# Download from ollama.ai
ollama pull llama3.2
```

## Running the Application

### Part 1: Basic API

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run the API
python main.py
```

Visit: http://localhost:8000/docs

### Part 2: Advanced Prompting with Streaming

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run advanced API
python advanced_prompts.py

# Terminal 3: Test prompts
python test_prompts.py

# Or open test_stream.html in browser for interactive chat
```

### Part 3: RAG System (Document Q&A)

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run RAG API
python rag_api.py
```

**Upload a document:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/your/document.pdf"
```

**Query the document:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

Or visit http://localhost:8000/docs for interactive API documentation.

## Features by Part

### Part 1
- Local LLM integration with Ollama
- FastAPI backend with automatic documentation
- Basic chat endpoint

### Part 2
- Advanced prompting techniques (zero-shot, few-shot, chain-of-thought)
- Temperature and parameter control
- Streaming responses
- Interactive chat interface

### Part 3
- PDF document processing and chunking
- Vector embeddings with sentence-transformers
- ChromaDB for semantic search
- RAG pipeline (retrieval + generation)
- Document upload and management API
- Source citations

## Blog Posts

- [Part 1: Setting Up Your Free AI Development Environment](https://medium.com/@meetnandu996/building-production-ready-ai-applications-part-1-setting-up-your-free-ai-development-18b4b0681cd1)
- [Part 2: Prompt Engineering with Local Models](https://medium.com/@meetnandu996/building-production-ready-ai-applications-part-2-prompt-engineering-with-local-models-f4d10a5d0e5c)
- Part 3: Building RAG Systems (publishing soon)

## Tech Stack

- **LLM**: Ollama (Llama 3.2)
- **Backend**: FastAPI
- **Vector DB**: ChromaDB
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Document Processing**: pypdf
- **Frontend**: HTML/CSS/JS (Part 4)
- **Deployment**: Free tier hosting (Part 5)

## Project Structure
docuchat/
├── main.py                   # Part 1: Basic API
├── advanced_prompts.py       # Part 2: Advanced prompting
├── test_prompts.py          # Part 2: Testing script
├── test_stream.html         # Part 2: Streaming chat UI
├── document_processor.py    # Part 3: PDF processing
├── vector_store.py          # Part 3: ChromaDB management
├── rag_engine.py            # Part 3: RAG pipeline
├── rag_api.py               # Part 3: RAG API endpoints
├── test_rag.py              # Part 3: RAG testing
├── requirements.txt         # Dependencies
└── README.md                # This file

## Troubleshooting

### Common Issues

**1. "could not connect to a running Ollama instance"**
- Solution: Run `ollama serve` in a separate terminal

**2. "ModuleNotFoundError"**
- Solution: Make sure virtual environment is activated and dependencies are installed

**3. "Port already in use"**
- Solution: Kill the process using port 8000 or change the port in the Python file

**4. Slow performance**
- Solution: Use a smaller model like `llama3.2:1b` or reduce chunk count

## Contributing

This is an educational project. Feel free to fork and experiment!

## License

MIT License

## Author

**Meet Nandu** - Software Development Engineer @ Amazon

- Medium: [@meetnandu996](https://medium.com/@meetnandu996)
- GitHub: [@meetnandu05](https://github.com/meetnandu05)
- LinkedIn: [meetnandu](https://www.linkedin.com/in/meetnandu)

---

⭐ Star this repo if you find it helpful!
