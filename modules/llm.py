import requests
from config import OLLAMA_URL, LLM_MODEL


def ask_llm(prompt: str) -> str:

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()

        data = r.json()

        # иногда Ollama возвращает без "response"
        if "response" not in data:
            print("LLM raw response:", data)
            return ""

        return data["response"].strip()

    except requests.exceptions.ConnectionError:
        return "Ошибка: Ollama не запущена."

    except requests.exceptions.Timeout:
        return "Ошибка: модель слишком долго отвечает."

    except Exception as e:
        return f"LLM error: {repr(e)}"