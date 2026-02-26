import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading

# ===== НАСТРОЙКИ =====

MODEL_SIZE = "medium"
# варианты:
# tiny  - очень быстро
# base  - лучший баланс
# small - точнее
# medium - очень точно
# large-v3 - максимум точности

SAMPLE_RATE = 16000
BLOCK_DURATION = 3  # секунд записи

# ===== ЗАГРУЗКА МОДЕЛИ =====

print("Загрузка Whisper...")

model = WhisperModel(
    MODEL_SIZE,
    device="cuda",      # "cuda" если есть NVIDIA
    compute_type="float16"
)

print("Whisper загружен")

# ===== ОЧЕРЕДЬ =====

audio_queue = queue.Queue()

# ===== CALLBACK =====

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

# ===== ЗАПИСЬ И РАСПОЗНАВАНИЕ =====

def record_chunk():

    audio_data = []

    blocks = int(SAMPLE_RATE / 1024 * BLOCK_DURATION)

    for _ in range(blocks):
        data = audio_queue.get()
        audio_data.append(data)

    audio_np = np.concatenate(audio_data, axis=0)

    audio_np = audio_np.flatten().astype(np.float32) / 32768.0

    return audio_np

# ===== ОСНОВНАЯ ФУНКЦИЯ =====

def listen():

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=1024,
        callback=audio_callback
    ):

        while True:

            audio = record_chunk()

            segments, info = model.transcribe(
                audio,
                language="ru",
                beam_size=1,
                vad_filter=True
            )

            text = ""

            for segment in segments:
                text += segment.text

            text = text.strip()

            if text:
                return text