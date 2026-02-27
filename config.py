OLLAMA_URL = "http://localhost:11434/api/generate"

# Проверь через: ollama list
LLM_MODEL = "mistral"
VISION_MODEL = "llava"

# Whisper
WHISPER_DEVICE = "cuda"      # если нестабильно → "cpu"
WHISPER_COMPUTE = "float16"  # если cpu → "int8"