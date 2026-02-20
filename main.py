from fastapi import FastAPI
from pydantic import BaseModel
import ollama

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    model: str = "llama3.2"

class PromptResponse(BaseModel):
    response: str
    model: str

@app.get("/")
def read_root():
    return {"message": "Welcome to DocuChat API!"}

@app.post("/chat", response_model=PromptResponse)
def chat(request: PromptRequest):
    """
    Send a prompt to the local LLM and get a response
    """
    try:
        # Call Ollama with the prompt
        response = ollama.chat(
            model=request.model,
            messages=[
                {
                    'role': 'user',
                    'content': request.prompt
                }
            ]
        )
        
        return PromptResponse(
            response=response['message']['content'],
            model=request.model
        )
    
    except Exception as e:
        return PromptResponse(
            response=f"Error: {str(e)}",
            model=request.model
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)