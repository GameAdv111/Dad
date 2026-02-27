import collections
import queue
import time
import re

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

from config import WHISPER_DEVICE, WHISPER_COMPUTE
import modules.text_to_speech as tts  # важно: модуль, не переменная


# ===== Настройки =====
MODEL_SIZE = "small"
SAMPLE_RATE = 16000

FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

VAD_AGGRESSIVENESS = 2
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

START_TRIGGER_FRAMES = 6       # ~180ms речи
END_SILENCE_FRAMES = 25        # ~750ms тишины
MAX_UTTERANCE_SEC = 20

POST_TTS_GRACE_SEC = 0.55      # хвост динамиков после TTS

# анти-шум фильтры
MIN_UTTERANCE_SEC = 0.7        # слишком короткие куски чаще шум/эхо
MIN_CHARS = 4                  # слишком короткие тексты выкидываем
MIN_AVG_LOGPROB = -1.25        # чем ближе к 0, тем увереннее; -1.25 норм порог
MAX_NO_SPEECH_PROB = 0.55      # если модель думает "это не речь" — выкидываем

# типовые галлюцинации Whisper (рус/ютуб концовки)
HALLUCINATION_PHRASES = {
    "спасибо за внимание",
    "спасибо за просмотр",
    "подписывайтесь на канал",
    "ставьте лайк",
    "всем спасибо",
    "до новых встреч",
}

# иногда прилетает мусор вроде одиночных "а", "и", "ну"
SHORT_JUNK_RE = re.compile(r"^(а|и|ну|эм|ээ|мм)$", re.IGNORECASE)


print("Loading Whisper...")
model = WhisperModel(
    MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE
)
print("Whisper loaded")


_audio_q: "queue.Queue[np.ndarray]" = queue.Queue()


def _drain_queue(max_items: int = 100000):
    n = 0
    while n < max_items:
        try:
            _audio_q.get_nowait()
            n += 1
        except queue.Empty:
            break


def _callback(indata, frames, t, status):
    # half-duplex: пока ассистент говорит — вообще не слушаем
    if tts.is_speaking:
        return
    if status:
        print("Audio status:", status)
    _audio_q.put(indata.copy())


def _is_speech(frame_int16_mono: np.ndarray) -> bool:
    return vad.is_speech(frame_int16_mono.tobytes(), SAMPLE_RATE)


_stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
    blocksize=FRAME_SAMPLES,
    callback=_callback,
)
_stream.start()


def _should_drop(text: str, utter_sec: float, seg_avg_logprob: float | None, no_speech_prob: float | None) -> bool:
    t = text.strip().lower()

    if not t:
        return True

    # слишком короткий кусок аудио — часто шум/эхо
    if utter_sec < MIN_UTTERANCE_SEC:
        return True

    # слишком короткий текст — часто мусор
    if len(t) < MIN_CHARS:
        return True

    # одиночные мусорные междометия
    if SHORT_JUNK_RE.match(t):
        return True

    # типовые галлюцинации
    if t in HALLUCINATION_PHRASES:
        return True

    # если модель считает, что это "не речь"
    if no_speech_prob is not None and no_speech_prob > MAX_NO_SPEECH_PROB:
        return True

    # фильтр уверенности по logprob (если есть)
    if seg_avg_logprob is not None and seg_avg_logprob < MIN_AVG_LOGPROB:
        return True

    return False


def listen() -> str | None:
    # ждём окончания TTS и чистим хвост
    while tts.is_speaking:
        _drain_queue()
        time.sleep(0.05)

    time.sleep(POST_TTS_GRACE_SEC)
    _drain_queue()

    ring = collections.deque(maxlen=START_TRIGGER_FRAMES)
    voiced_frames: list[np.ndarray] = []

    in_speech = False
    silence_count = 0

    max_frames = int(MAX_UTTERANCE_SEC * 1000 / FRAME_MS)
    frames_collected = 0

    while True:
        if tts.is_speaking:
            # если внезапно стартанул TTS — сброс и ожидание
            ring.clear()
            voiced_frames.clear()
            in_speech = False
            silence_count = 0
            while tts.is_speaking:
                _drain_queue()
                time.sleep(0.05)
            time.sleep(POST_TTS_GRACE_SEC)
            _drain_queue()
            continue

        frame = _audio_q.get()
        mono = frame[:, 0].copy()
        frames_collected += 1

        speech = _is_speech(mono)

        if not in_speech:
            ring.append((mono, speech))
            num_voiced = sum(1 for _, s in ring if s)

            if num_voiced >= START_TRIGGER_FRAMES:
                in_speech = True
                voiced_frames.extend([f for f, _ in ring])  # pre-roll
                ring.clear()
        else:
            voiced_frames.append(mono)
            silence_count = 0 if speech else (silence_count + 1)

            if silence_count >= END_SILENCE_FRAMES:
                break
            if frames_collected >= max_frames:
                break

    if not voiced_frames:
        return None

    audio_int16 = np.concatenate(voiced_frames, axis=0)
    utter_sec = float(audio_int16.shape[0]) / float(SAMPLE_RATE)

    audio = audio_int16.astype(np.float32) / 32768.0

    # Анти-галлюцинационные настройки:
    # temperature=0 → меньше "додумываний"
    # condition_on_previous_text=False → меньше цепляния к прошлому мусору
    segments, info = model.transcribe(
        audio,
        language="ru",
        beam_size=1,
        vad_filter=False,
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
    )

    seg_texts = []
    avg_logprobs = []
    for s in segments:
        seg_texts.append(s.text)
        # в faster-whisper у сегмента обычно есть avg_logprob
        avg_logprobs.append(getattr(s, "avg_logprob", None))

    text = "".join(seg_texts).strip()

    # берём средний logprob по сегментам (если доступно)
    seg_avg_logprob = None
    vals = [v for v in avg_logprobs if isinstance(v, (int, float))]
    if vals:
        seg_avg_logprob = sum(vals) / len(vals)

    no_speech_prob = getattr(info, "no_speech_prob", None)

    if _should_drop(text, utter_sec, seg_avg_logprob, no_speech_prob):
        return None

    return text