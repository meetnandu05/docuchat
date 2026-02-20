import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "prompt": "Write a haiku about coding",
        "model": "llama3.2"
    }
)

print(response.json()['response'])