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

def ask_vision(prompt):

    image = capture_screen()

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [image],
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload)

    return r.json()["response"]