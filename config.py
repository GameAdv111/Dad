OLLAMA_URL = "http://localhost:11434/api/generate"

# Проверь через: ollama list
LLM_MODEL = "mistral"
VISION_MODEL = "llava"

FFMPEG_PATH = r"C:\Users\sepp3\OneDrive\Desktop\Project\ffmpeg\bin\ffmpeg.exe"
PITCH_SEMITONES = 3

VOSK_MODEL_PATH = "models/vosk/vosk-model-small-ru-0.22"
STT_DEVICE_INDEX = 1   # <-- поставь индекс микрофона из mic_test.py (например 3)
STT_SAMPLE_RATE = None    # None = взять default_samplerate устройства (рекомендую)
STT_BLOCK_SEC = 0.2
STT_SILENCE_TIMEOUT = 1.2
STT_MIN_CHARS = 2
STT_DEBUG = True
STT_MIN_WORDS = 2
STT_MAX_UTTERANCE_SEC = 12