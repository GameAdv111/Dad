import requests
import json
from config import OLLAMA_URL, LLM_MODEL

def ask_llm(prompt):

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload)

    return r.json()["response"]