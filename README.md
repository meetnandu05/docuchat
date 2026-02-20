# DocuChat - AI-Powered Document Q&A

Part of my Medium blog series: "Building Production-Ready AI Applications"

## What is DocuChat?

An AI-powered application that lets you upload documents and ask questions about them using local LLMs.

## Tech Stack

- **LLM**: Ollama (Llama 3.2)
- **Backend**: FastAPI
- **Vector DB**: ChromaDB (coming in Part 3)
- **Deployment**: Render/Railway (coming in Part 6)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/docuchat.git
cd docuchat
```

2. Create and activate virtual environment:
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

4. Install and run Ollama:
- Download from [ollama.ai](https://ollama.ai)
- Pull the model: `ollama pull llama3.2`
- Start Ollama: `ollama serve`

5. Run the application:
```bash
python main.py
```

6. Visit http://localhost:8000/docs to test the API

## Blog Series

- **Part 1**: Setting Up Your Free AI Development Environment (current)
- **Part 2**: Prompt Engineering with Local Models (coming soon)
- **Part 3**: Building RAG Systems (coming soon)

## Author

Meet Nandu - Software Development Engineer at Amazon

Follow me on [Medium](https://medium.com/@meetnandu996) for updates!

## License

MIT License