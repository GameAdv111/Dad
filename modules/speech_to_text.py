import json
import queue
import time

import sounddevice as sd
from vosk import Model, KaldiRecognizer

import modules.text_to_speech as tts
from config import (
    VOSK_MODEL_PATH,
    STT_DEVICE_INDEX,
    STT_SAMPLE_RATE,
    STT_BLOCK_SEC,
    STT_SILENCE_TIMEOUT,
    STT_MIN_WORDS,
    STT_MAX_UTTERANCE_SEC,
)

_model = None
_rec = None
_q: "queue.Queue[bytes]" = queue.Queue()
_stream = None

_last_voice_ts = 0.0
_utt_start_ts = 0.0
_best_text = ""


def _init():
    global _model, _rec, _stream, _last_voice_ts, _utt_start_ts, _best_text
    if _stream is not None:
        return

    if STT_DEVICE_INDEX is not None:
        sd.default.device = (STT_DEVICE_INDEX, None)

    dev = sd.default.device[0]
    dev_info = sd.query_devices(dev, "input")
    sr = int(dev_info["default_samplerate"]) if STT_SAMPLE_RATE is None else int(STT_SAMPLE_RATE)

    _model = Model(VOSK_MODEL_PATH)
    _rec = KaldiRecognizer(_model, sr)
    _rec.SetWords(False)

    _last_voice_ts = time.time()
    _utt_start_ts = time.time()
    _best_text = ""

    def callback(indata, frames, time_info, status):
        _q.put(bytes(indata))

    _stream = sd.RawInputStream(
        samplerate=sr,
        blocksize=int(sr * STT_BLOCK_SEC),
        dtype="int16",
        channels=1,
        callback=callback,
    )
    _stream.start()


def _flush_queue():
    try:
        while True:
            _q.get_nowait()
    except queue.Empty:
        pass


def _reset_utt():
    global _best_text, _utt_start_ts
    _best_text = ""
    _utt_start_ts = time.time()
    _rec.Reset()


def listen(timeout_sec: float = 0.15) -> str | None:
    """
    Возвращает фразу ТОЛЬКО когда наступила пауза STT_SILENCE_TIMEOUT.
    Это убирает ранние "обрывки".
    """
    global _last_voice_ts, _best_text

    _init()

    if tts.is_speaking:
        _flush_queue()
        _reset_utt()
        time.sleep(0.05)
        return None

    start = time.time()
    got_audio = False

    while (time.time() - start) < timeout_sec:
        try:
            data = _q.get_nowait()
            got_audio = True
        except queue.Empty:
            break

        # Даже если Vosk считает waveform "final", мы не возвращаем сразу.
        # Мы просто обновим _best_text.
        if _rec.AcceptWaveform(data):
            res = json.loads(_rec.Result() or "{}")
            txt = (res.get("text") or "").strip()
            if txt:
                _best_text = txt
                _last_voice_ts = time.time()
        else:
            pres = json.loads(_rec.PartialResult() or "{}")
            part = (pres.get("partial") or "").strip()
            if part:
                _best_text = part  # partial часто "свежее" и длиннее
                _last_voice_ts = time.time()

    now = time.time()

    # защита от "говорю слишком долго" — режем по максимуму
    if (now - _utt_start_ts) > STT_MAX_UTTERANCE_SEC and _best_text:
        text = _best_text.strip()
        _reset_utt()
        if len(text.split()) >= STT_MIN_WORDS:
            return text
        return None

    # финалим только по тишине
    if _best_text and (now - _last_voice_ts) >= STT_SILENCE_TIMEOUT:
        # добьём финалом
        fres = json.loads(_rec.FinalResult() or "{}")
        final_txt = (fres.get("text") or "").strip()
        text = (final_txt or _best_text).strip()

        _reset_utt()

        if len(text.split()) >= STT_MIN_WORDS:
            return text

    return None