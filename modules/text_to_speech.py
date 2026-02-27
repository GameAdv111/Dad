import subprocess
import tempfile
import os
import threading
import queue
import winsound

MODEL_PATH = os.path.join("models", "piper", "ru_RU-irina-medium.onnx")

_q = queue.Queue()
is_speaking = False


def _worker():
    global is_speaking
    print("TTS worker started (Piper + winsound)")

    while True:
        text = _q.get()
        if text is None:
            _q.task_done()
            break

        wav_path = None

        try:
            is_speaking = True
            text = str(text).strip()
            if not text:
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                wav_path = f.name

            # ✅ ВАЖНО: Piper читает stdin, поэтому даём bytes UTF-8 + \n
            result = subprocess.run(
                ["piper", "--model", MODEL_PATH, "--output_file", wav_path],
                input=(text + "\n").encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if result.returncode != 0:
                err = result.stderr.decode("utf-8", errors="replace")
                print("Piper error:", err)
                continue

            if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                print("TTS error: wav not created")
                continue

            print("Jarvis(TTS):", text)
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)

        except Exception as e:
            print("TTS exception:", repr(e))

        finally:
            is_speaking = False
            _q.task_done()

            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass


threading.Thread(target=_worker, daemon=False).start()


def speak(text: str):
    if text:
        _q.put(text)


def speak_blocking(text: str):
    if not text:
        return
    _q.put(text)
    _q.join()


def stop_tts():
    _q.put(None)