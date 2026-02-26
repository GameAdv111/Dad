import queue
import threading
import pyttsx3
import time

_engine = pyttsx3.init()
_engine.setProperty("rate", 170)

# очередь фраз на озвучку
_q: "queue.Queue[str]" = queue.Queue()

# флаг: сейчас говорит (можно использовать для подавления self-listen)
is_speaking = False

def _tts_worker():
    global is_speaking
    while True:
        text = _q.get()  # блокируется, пока не появится текст
        if text is None:
            break

        try:
            is_speaking = True
            print("Jarvis:", text)
            _engine.say(text)
            _engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)
        finally:
            is_speaking = False
            # маленькая пауза, чтобы микрофон не схватил хвост
            time.sleep(0.05)
            _q.task_done()

# запускаем один раз
_thread = threading.Thread(target=_tts_worker, daemon=True)
_thread.start()

def set_voice_by_index(index: int):
    """Опционально: выбрать голос по индексу."""
    voices = _engine.getProperty("voices")
    if 0 <= index < len(voices):
        _engine.setProperty("voice", voices[index].id)

def list_voices():
    """Опционально: вывести список голосов в консоль."""
    voices = _engine.getProperty("voices")
    for i, v in enumerate(voices):
        print(i, getattr(v, "name", "Unknown"), v.id)

def speak(text: str):
    """Неблокирующая озвучка: кладёт текст в очередь."""
    if not text:
        return
    _q.put(text)

def speak_blocking(text: str):
    """Блокирующая озвучка: дождаться, пока фраза будет произнесена."""
    if not text:
        return
    _q.put(text)
    _q.join()

def stop_tts():
    """Остановить поток TTS (обычно не нужно)."""
    _q.put(None)