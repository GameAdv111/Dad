from config import FFMPEG_PATH, PITCH_SEMITONES
import os
import subprocess
import tempfile
import threading
import queue
import time
import simpleaudio as sa

# Путь к модели Piper (поправь под себя)
PIPER_MODEL_PATH = os.path.join("models", "piper", "ru_RU-irina-medium.onnx")

_q: "queue.Queue[str | None]" = queue.Queue()
is_speaking = False

def _pitch_shift(input_wav: str, output_wav: str, semitones: float = None):
    import subprocess, os
    if semitones is None:
        semitones = PITCH_SEMITONES

    if not os.path.exists(FFMPEG_PATH):
        raise FileNotFoundError(f"ffmpeg not found at: {FFMPEG_PATH}")

    factor = 2 ** (semitones / 12)
    subprocess.run([
        FFMPEG_PATH,
        "-y",
        "-i", input_wav,
        "-filter:a", f"asetrate=22050*{factor},aresample=22050",
        output_wav
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def _play_wav(path: str) -> None:
    wave = sa.WaveObject.from_wave_file(path)
    play = wave.play()
    play.wait_done()


def _synth_piper_to_wav(text: str, out_wav: str) -> None:
    """
    Генерация WAV через piper CLI.
    ВАЖНО: input должен быть bytes (utf-8), иначе ловишь Unicode/TypeError.
    """
    if not os.path.exists(PIPER_MODEL_PATH):
        raise FileNotFoundError(f"Piper model not found: {PIPER_MODEL_PATH}")

    r = subprocess.run(
        ["piper", "--model", PIPER_MODEL_PATH, "--output_file", out_wav],
        input=(text.strip() + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Piper failed: {err[:500]}")


def _worker() -> None:
    global is_speaking
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

            _synth_piper_to_wav(text, wav_path)

            pitched_path = wav_path.replace(".wav", "_pitch.wav")
            _pitch_shift(wav_path, pitched_path, semitones=2.5)

            _play_wav(pitched_path)

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
            time.sleep(0.02)


threading.Thread(target=_worker, daemon=False).start()


def speak(text: str) -> None:
    if text:
        _q.put(text)


def speak_blocking(text: str) -> None:
    if not text:
        return
    _q.put(text)
    _q.join()


def stop_tts() -> None:
    _q.put(None)