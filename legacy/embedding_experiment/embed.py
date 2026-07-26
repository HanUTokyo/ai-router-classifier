import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

def embed(text: str):
    res = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": text
    })

    return res.json()["embedding"]