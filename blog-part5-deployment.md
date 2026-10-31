# Building Production-Ready AI Applications (Part 5): Deploying to the Internet for Free

*This is Part 5 (the finale!) of a five-part series on building production-ready AI applications. In [Part 1](https://medium.com/@meetnandu996/building-production-ready-ai-applications-part-1-setting-up-your-free-ai-development-18b4b0681cd1), we set up our local LLM. In [Part 2](https://medium.com/@meetnandu996/building-production-ready-ai-applications-part-2-prompt-engineering-with-local-models-f4d10a5d0e5c), we mastered prompt engineering. In [Part 3](#), we built a RAG backend. In [Part 4](https://medium.com/@meetnandu996/building-production-ready-ai-applications-part-4-building-the-frontend-connecting-everything-a347b94a83ad), we built the frontend and connected everything.*

*Today, we're taking DocuChat from your laptop to the internet.*

---

## What You'll Learn

- Why you can't just deploy Ollama to a free hosting provider (and what to use instead)
- Adding a cloud LLM provider (Groq) while keeping local Ollama for development
- Environment variables and configuration management
- Dockerizing the application
- Deploying to Render's free tier
- Handling the free-tier sleep/wake cycle
- Getting a live URL to share with anyone

## The Deployment Problem

Here's the thing nobody tells you in most "deploy your AI app" tutorials: **you can't deploy Ollama to Render's free tier**.

Ollama needs:
- **Several GB of RAM** to load even a small model (llama3.2 needs ~4GB)
- **CPU/GPU power** to run inference

Render's free tier gives you:
- **512MB RAM**
- **0.1 CPU**

That's not even close. And this isn't a Render problem — no free hosting platform gives you enough resources to run an LLM locally.

### The Solution: Swap the LLM at Deployment Time

We'll use a **cloud LLM API** for the deployed version. The best free option right now is [Groq](https://groq.com):

| Feature | Ollama (Local) | Groq (Cloud) |
|---------|---------------|--------------|
| Cost | Free | Free tier (generous) |
| Speed | Depends on your hardware | Extremely fast (~500 tokens/sec) |
| Privacy | 100% local | Data sent to Groq servers |
| Setup | Install Ollama + pull model | Get API key |
| RAM needed | 4GB+ | ~100MB (just your app) |

The architecture stays the same — we just swap what answers the questions:

```
LOCAL DEVELOPMENT:
  Browser → FastAPI → ChromaDB → Ollama (local)

DEPLOYED:
  Browser → FastAPI → ChromaDB → Groq API (cloud)
```

The key insight: **the retrieval part (ChromaDB, embeddings) still runs locally on the server**. Only the generation step changes. This means our RAG quality stays the same — we're just using a different brain to synthesize the answer.

## Step 1: Get a Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card)
3. Go to **API Keys** → **Create API Key**
4. Copy and save the key — you'll need it soon

Groq's free tier gives you 30 requests per minute with models like Llama 3, which is more than enough for a demo app.

## Step 2: Make the App Provider-Agnostic

Right now, `rag_engine.py` hardcodes `ollama.chat()`. We need it to use Ollama locally and Groq when deployed. The cleanest way: **an environment variable that picks the provider**.

Create a new file called `llm_provider.py`:

```python
"""
LLM Provider abstraction.
Uses Ollama locally, Groq API when deployed.
Controlled by LLM_PROVIDER environment variable.
"""

import os


def get_llm_response(messages: list, model: str = None, temperature: float = 0.3) -> str:
    """
    Get a response from the configured LLM provider.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name (defaults based on provider)
        temperature: Response temperature
    
    Returns:
        The assistant's response text
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider == "groq":
        return _groq_chat(messages, model or "llama-3.1-8b-instant", temperature)
    else:
        return _ollama_chat(messages, model or "llama3.2", temperature)


def _ollama_chat(messages: list, model: str, temperature: float) -> str:
    """Local Ollama inference"""
    import ollama
    
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature}
    )
    return response["message"]["content"]


def _groq_chat(messages: list, model: str, temperature: float) -> str:
    """Groq cloud inference"""
    import requests
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required when using Groq provider")
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024
        },
        timeout=30
    )
    
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
```

**Design decisions:**
- **Lazy imports** — `import ollama` only runs when you actually use Ollama. This means the deployed version never tries to import (or install) the `ollama` package.
- **Standard OpenAI-compatible API** — Groq uses the OpenAI chat completions format, so switching to OpenAI, Together AI, or any other provider later is trivial.
- **No new dependencies** — We already have `requests` in our requirements. No SDK needed.

## Step 3: Update the RAG Engine

Now update `rag_engine.py` to use our provider abstraction. Only one method changes — `generate_answer`:

```python
# At the top, replace:
import ollama

# With:
from llm_provider import get_llm_response
```

Then update the `generate_answer` method:

```python
def generate_answer(self, query: str, contexts: list) -> str:
    """Generate answer using LLM with retrieved context"""
    
    context_text = "\n\n".join([
        f"[Context {i+1}]\n{ctx}" 
        for i, ctx in enumerate(contexts)
    ])

    prompt = f"""You are a helpful assistant answering questions about documents.
Use the following context to answer the question. If the answer cannot be found 
in the context, say so clearly.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER (be concise and cite which context sections you used):"""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on provided context. Always cite your sources."
        },
        {
            "role": "user", 
            "content": prompt
        }
    ]

    return get_llm_response(messages, model=self.model_name, temperature=0.3)
```

Also update the `__init__` to pick a sensible default model:

```python
import os

class RAGEngine:
    def __init__(self, model_name: str = None):
        self.processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        self.vector_store = VectorStore()
        
        # Pick model based on provider
        if model_name:
            self.model_name = model_name
        elif os.getenv("LLM_PROVIDER", "ollama") == "groq":
            self.model_name = "llama-3.1-8b-instant"
        else:
            self.model_name = "llama3.2"
        
        print(f"RAG Engine initialized with model: {self.model_name}")
```

**That's it.** The rest of the RAG engine — document processing, vector search, source citations — stays exactly the same.

Test locally to make sure nothing broke:

```bash
# Should still work exactly as before
python rag_api.py
```

Then test with Groq (optional, to verify before deploying):

```bash
export LLM_PROVIDER=groq
export GROQ_API_KEY=your_key_here
python rag_api.py
# Upload a PDF and ask a question — should work identically, just faster
```

## Step 4: Production Hardening

Before deploying, let's handle some production concerns.

### Add Health Check and Error Recovery

Update `rag_api.py` — add a health check endpoint and improve error handling:

```python
@app.get("/health")
def health_check():
    """Health check for Render"""
    return {"status": "healthy", "provider": os.getenv("LLM_PROVIDER", "ollama")}
```

### Pin Dependencies

Replace `requirements.txt` with pinned versions. This prevents "it worked yesterday" surprises:

```
fastapi==0.115.0
uvicorn==0.32.0
requests==2.32.3
chromadb==0.5.23
pypdf==5.1.0
sentence-transformers==3.3.1
python-multipart==0.0.17

# Only needed for local development (Ollama)
# ollama==0.4.4
```

Notice `ollama` is commented out. On Render, we don't need it — and it pulls in heavy dependencies we'd rather skip.

Create a separate `requirements-deploy.txt` for the deployed version:

```
fastapi==0.115.0
uvicorn==0.32.0
requests==2.32.3
chromadb==0.5.23
pypdf==5.1.0
python-multipart==0.0.17
```

Notice we removed **both** `ollama` and `sentence-transformers` from the deploy file. Ollama isn't needed because we use Groq. And `sentence-transformers` pulls in PyTorch (~1.5GB installed, ~400MB RAM at startup) — that alone blows past Render's 512MB free-tier limit.

Instead, we'll use ChromaDB's **built-in embedder**, which runs the same `all-MiniLM-L6-v2` model via `onnxruntime` at ~100MB RAM. Same embedding quality, 4x less memory. More on this in a moment.

### Update the Vector Store for Lightweight Embeddings

This is the key change that makes free-tier deployment possible. Update `vector_store.py` to conditionally use ChromaDB's built-in embedder when deployed:

```python
import chromadb
from chromadb.config import Settings
from typing import List, Dict
import os

class VectorStore:
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
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
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document chunks for RAG"}
        )
```

The `add_documents` and `search` methods now branch on whether `self.embedding_model` exists:

```python
def add_documents(self, chunks, document_name):
    texts = [chunk['text'] for chunk in chunks]
    ids = [f"{document_name}_{chunk['id']}" for chunk in chunks]
    metadatas = [...]  # same as before
    
    if self.embedding_model:
        embeddings = self.create_embeddings(texts)
        self.collection.add(embeddings=embeddings, documents=texts, ids=ids, metadatas=metadatas)
    else:
        # ChromaDB generates embeddings automatically
        self.collection.add(documents=texts, ids=ids, metadatas=metadatas)

def search(self, query, n_results=3):
    if self.embedding_model:
        query_embedding = self.create_embeddings([query])[0]
        return self.collection.query(query_embeddings=[query_embedding], n_results=n_results, ...)
    else:
        # ChromaDB handles embedding the query too
        return self.collection.query(query_texts=[query], n_results=n_results, ...)
```

**Why this works:** ChromaDB ships with a default embedding function that uses `all-MiniLM-L6-v2` via `onnxruntime`. When you don't pass explicit embeddings, ChromaDB embeds the text automatically. Same model, same quality — just a lighter runtime (~100MB vs ~400MB for PyTorch).

**Why not always use the built-in embedder?** If you're running locally with plenty of RAM, `sentence-transformers` gives you more flexibility (easy to swap models, GPU acceleration, batch tuning). For a 512MB server, the built-in embedder is the right choice.

### Add Environment Configuration

Create a `.env.example` file so others know what to configure:

```bash
# LLM Provider: "ollama" (local) or "groq" (cloud)
LLM_PROVIDER=ollama

# Required when LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here

# Server configuration
PORT=8000
HOST=0.0.0.0
```

Update your `.gitignore` to keep secrets out of Git:

```
.env
venv/
uploads/
__pycache__/
chroma_db/
*.pyc
```

## Step 5: Dockerize (Optional but Recommended)

Docker ensures "it works on my machine" actually means "it works everywhere."

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why `python:3.11-slim`?** The `slim` variant is ~150MB vs ~1GB for the full image. We add `build-essential` for packages that need C compilation (like ChromaDB's native extensions).

**Important:** Create a `.dockerignore` file to prevent your `venv/` (which can be 1.5GB+) from being sent to Docker:

```
venv/
__pycache__/
chroma_db/
uploads/
.env
*.pyc
.git/
```

Without this, `docker build` will spend minutes just transferring your virtual environment — only to throw it away and install fresh inside the container.

Test locally:

```bash
docker build -t docuchat .
docker run -p 8000:8000 -e LLM_PROVIDER=groq -e GROQ_API_KEY=your_key docuchat
```

## Step 6: Deploy to Render

### Option A: Deploy from GitHub (Recommended)

1. Push your code to GitHub:

```bash
git add -A
git commit -m "feat: add deployment support with Groq provider"
git push origin main
```

2. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service**

3. Connect your GitHub repository (`meetnandu05/docuchat`)

4. Configure the service:

| Setting | Value |
|---------|-------|
| **Name** | `docuchat` |
| **Region** | Oregon (US West) or nearest to you |
| **Branch** | `main` |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements-deploy.txt` |
| **Start Command** | `uvicorn rag_api:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Free |

5. Add **Environment Variables**:

| Key | Value |
|-----|-------|
| `LLM_PROVIDER` | `groq` |
| `GROQ_API_KEY` | `your_actual_groq_key` |
| `PYTHON_VERSION` | `3.11.9` |

6. Click **Create Web Service**

Render will build and deploy your app. First deploy takes 3-5 minutes. Subsequent deploys are faster thanks to build caching.

### Option B: Deploy with Docker

If you prefer Docker (more reproducible):

1. In Render settings, change **Environment** to **Docker**
2. Remove the Build/Start commands (Docker handles it)
3. Add the same environment variables
4. Deploy

### Your Live URL

Once deployed, you'll get a URL like:

```
https://docuchat-xxxx.onrender.com
```

Open it in your browser — you should see the full DocuChat interface!

## Step 7: Handle the Free-Tier Sleep Cycle

Render's free tier **spins down your service after 15 minutes of no traffic**. When someone visits after a sleep period, the first request takes ~30-60 seconds while the service wakes up.

This is fine for a portfolio/demo app. But the user experience of staring at a blank page for 60 seconds isn't great. Here are two ways to handle it:

### Option 1: Add a Loading State (Recommended)

Update `frontend/app.js` — modify the `init()` function to show a friendly loading state:

```javascript
async function init() {
    try {
        // Show connecting state
        inputHint.textContent = '⏳ Waking up the server (free tier may take ~30s)...';
        
        const response = await fetch(`${API_URL}/health`, {
            signal: AbortSignal.timeout(90000) // 90 second timeout for cold starts
        });
        
        if (!response.ok) throw new Error();
        
        inputHint.textContent = '✅ Connected!';
    } catch (error) {
        showToast('Server is waking up. Please refresh in 30 seconds.', 'error');
        inputHint.textContent = '😴 Server is sleeping. Refresh to wake it up.';
        return;
    }

    await refreshDocuments();
    await refreshStats();
}
```

### Option 2: Keep-Alive Ping (Free, External)

Use a free cron service like [cron-job.org](https://cron-job.org) to ping your `/health` endpoint every 14 minutes:

1. Sign up at cron-job.org (free)
2. Create a new cron job:
   - URL: `https://docuchat-xxxx.onrender.com/health`
   - Schedule: Every 14 minutes
3. This keeps your service awake during the day

**Caveat:** Render gives 750 free hours/month. A 24/7 keep-alive uses ~744 hours, leaving almost no buffer. If you use keep-alive, set it only during your local daytime hours to save hours.

## Step 8: Final Project Structure

Here's the complete project after all five parts:

```
docuchat/
├── frontend/
│   ├── index.html          (Part 4)
│   ├── styles.css          (Part 4)
│   └── app.js              (Part 4, updated for cold-start handling)
├── llm_provider.py         (Part 5 -- NEW)
├── rag_engine.py           (Part 3, updated to use llm_provider)
├── rag_api.py              (Part 3-4, with health check)
├── document_processor.py   (Part 3)
├── vector_store.py         (Part 3, updated for lightweight embeddings)
├── main.py                 (Part 1)
├── advanced_prompts.py     (Part 2)
├── Dockerfile              (Part 5 -- NEW)
├── .dockerignore           (Part 5 -- NEW)
├── requirements.txt        (Parts 1-4, with ollama + sentence-transformers)
├── requirements-deploy.txt (Part 5 -- NEW, lightweight for free tier)
├── .env.example            (Part 5 -- NEW)
├── .gitignore
└── README.md
```

## Step 9: Test End-to-End

Verify the deployed app works:

1. **Open your Render URL** — you should see the DocuChat UI
2. **Upload a PDF** — drag-and-drop or click to browse
3. **Ask a question** — type or click a suggestion
4. **Check the answer** — it should have source citations, just like locally
5. **Share the URL** — send it to a friend and have them try it!

## What We Built: The Complete Series Recap

Over five parts, we built a production-ready AI application from scratch:

| Part | What We Built | Key Concept |
|------|--------------|-------------|
| **Part 1** | Local LLM + FastAPI | Running AI locally with Ollama |
| **Part 2** | Prompt engineering | Getting better answers from LLMs |
| **Part 3** | RAG backend | Teaching AI about your documents |
| **Part 4** | Frontend UI | Making it usable by real humans |
| **Part 5** | Deployment | Putting it on the internet for free |

The final tech stack:

- **LLM**: Ollama (local) / Groq (deployed) — Llama 3
- **Backend**: FastAPI + Python
- **Vector DB**: ChromaDB
- **Embeddings**: sentence-transformers / ChromaDB built-in (all-MiniLM-L6-v2)
- **Frontend**: Vanilla HTML/CSS/JS
- **Hosting**: Render (free tier)

And the total cost? **$0.** Everything — the LLM, the hosting, the vector database, the embeddings — is completely free.

## Where to Go From Here

This series gave you a solid foundation. Here's what you could build next:

### Improve DocuChat
- **Conversation memory** — Send previous messages as context for follow-up questions
- **Streaming responses** — Show tokens as they arrive (Part 2's streaming + frontend SSE)
- **Multiple file formats** — Add support for Word docs, text files, Markdown
- **Better chunking** — Use semantic chunking instead of fixed-size splits
- **Authentication** — Add user accounts so uploaded documents are private

### Apply the Pattern Elsewhere
The RAG architecture we built works for way more than PDF Q&A:
- **Code documentation chatbot** — Ingest your codebase, answer questions about it
- **Customer support bot** — Ingest your help docs, handle first-line support
- **Research assistant** — Ingest academic papers, synthesize across them
- **Personal knowledge base** — Ingest your notes, make them searchable

### Go Deeper
- **Fine-tuning** — Train a model on your specific domain (Unsloth + QLoRA)
- **Evaluation** — Measure RAG quality with RAGAS or custom metrics
- **Agents** — Give the LLM tools (web search, calculations, database queries)
- **Multi-modal** — Process images and tables alongside text

## Complete Code Repository

All code is available on GitHub:

- **Branch**: `part-5-deployment`
- **New files**: `llm_provider.py`, `Dockerfile`, `.dockerignore`, `requirements-deploy.txt`, `.env.example`
- **Modified**: `rag_engine.py`, `rag_api.py`, `vector_store.py`, `requirements.txt`, `.gitignore`

**GitHub Repository**: [github.com/meetnandu05/docuchat](https://github.com/meetnandu05/docuchat)

---

## Wrapping Up the Series

When I started this series, the goal was simple: show that you can build a real, production-ready AI application using nothing but free tools. No GPT-4 API bill. No AWS instance. No $200/month vector database.

We went from `ollama pull llama3.2` to a deployed application with document upload, semantic search, cited answers, and a polished UI — all for $0.

The AI landscape changes fast, but the fundamentals don't:
1. **Start local** — Prototype with free, local tools before committing to paid services
2. **Build in layers** — Each part of our stack is replaceable (swap Ollama for GPT-4, ChromaDB for Pinecone, Render for AWS — the architecture holds)
3. **Ship it** — A deployed demo is worth a thousand README files

Thanks for following along. If this series helped you build something, I'd love to hear about it — drop a comment or find me on [LinkedIn](https://www.linkedin.com/in/meet-nandu-05/).

Happy building! 🚀
