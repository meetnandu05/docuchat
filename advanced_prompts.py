from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import ollama

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str  # 'system', 'user', or 'assistant'
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "llama3.2"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

@app.get("/")
def read_root():
    return {
        "message": "Advanced Prompting API",
        "available_templates": ["summarize", "code_review", "explain", "creative"]
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Advanced chat endpoint with full control over conversation
    """
    try:
        # Convert our messages to Ollama format
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        # Prepare options
        options = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        
        # Call Ollama - use options only if we have any
        if options:
            response = ollama.chat(
                model=request.model,
                messages=ollama_messages,
                options=options
            )
        else:
            response = ollama.chat(
                model=request.model,
                messages=ollama_messages
            )
        
        return ChatResponse(
            response=response['message']['content'],
            model=request.model,
            prompt_tokens=response.get('prompt_eval_count'),
            completion_tokens=response.get('eval_count')
        )
    
    except Exception as e:
        # Print the error to console for debugging
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream responses token by token for better UX
    """
    async def generate():
        try:
            # Convert messages
            ollama_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            # Prepare options
            options = {}
            if request.temperature is not None:
                options["temperature"] = request.temperature
            
            # Stream from Ollama
            stream = ollama.chat(
                model=request.model,
                messages=ollama_messages,
                options=options if options else None,
                stream=True
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    # Send as Server-Sent Events format
                    yield f"data: {json.dumps({'content': content})}\n\n"
            
            # Send done signal
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)