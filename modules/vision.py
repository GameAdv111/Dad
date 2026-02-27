import mss
import base64
import io
import requests
from PIL import Image
from config import OLLAMA_URL, VISION_MODEL


def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)

        img = Image.frombytes("RGB", shot.size, shot.rgb)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        return base64.b64encode(buffer.getvalue()).decode()


def ask_vision(prompt: str) -> str:
    try:
        image = capture_screen()

        payload = {
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [image],
            "stream": False
        }

        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()

        data = r.json()

        if "response" not in data:
            print("Vision raw response:", data)
            return ""

        return data["response"].strip()

    except requests.exceptions.ConnectionError:
        return "Ошибка: Ollama не запущена."

    except requests.exceptions.Timeout:
        return "Ошибка: Vision модель слишком долго отвечает."

    except Exception as e:
        return f"Vision error: {repr(e)}"