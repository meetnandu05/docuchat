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
        return _groq_chat(messages, model or "openai/gpt-oss-20b", temperature)
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
    
    if not response.ok:
        print(f"Groq API error {response.status_code}: {response.text}")
        response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"]