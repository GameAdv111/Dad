import time
import random

from modules.speech_to_text import listen
from modules.llm import ask_llm
from modules.vision import ask_vision
from modules.chroma_memory import save_memory, search_memory
import modules.text_to_speech as tts


SYSTEM_PROMPT = """
Ты — живой разговорный компаньон и ассистент.
Твоя цель — не просто отвечать, а поддерживать естественный диалог.

Стиль:
- Говори по-человечески, тепло и уверенно, без канцелярита.
- Можно лёгкие эмоции/реакции, но без театральности.
- Чаще задавай 1 короткий вопрос в конце, чтобы разговор продолжался (если уместно).
- Если пользователь говорит коротко — помоги развить тему (уточни, предложи варианты).
- Не превращай каждый ответ в лекцию. Лучше 2–6 предложений, иногда больше по запросу.
- Помни, что это голосовой диалог: фразы должны быть удобны для слуха.

Поведение:
- Если пользователь явно просит ответ — ответь.
- Если пользователь просто делится мыслью — поддержи и задавай уточнение.
- Предлагай темы, связанные с тем, что пользователь обсуждал раньше.
- Если пользователь выглядит уставшим/раздражённым — снизь темп, будь мягче.

Запрещено:
- Не выдумывай фактов о пользователе.
- Если не уверен — уточни вопросом.
"""


# через сколько секунд тишины ассистент может “подать голос”
IDLE_MIN_SEC = 60
IDLE_MAX_SEC = 110

def build_prompt(user_text: str, context: str) -> str:
    return (
        SYSTEM_PROMPT.strip()
        + "\n\n"
        + "Релевантные воспоминания из прошлых разговоров:\n"
        + (context or "(нет)\n")
        + "\n"
        + f"Пользователь: {user_text}\n"
        + "Ответ:"
    )

def build_idle_prompt(context: str) -> str:
    return (
        SYSTEM_PROMPT.strip()
        + "\n\n"
        + "Ситуация: пользователь молчит уже некоторое время.\n"
        + "Твоя задача — мягко и ненавязчиво начать разговор.\n"
        + "Скажи одну короткую фразу (1–2 предложения) и задай один вопрос.\n"
        + "Не будь навязчивым.\n\n"
        + "Релевантные воспоминания:\n"
        + (context or "(нет)\n")
        + "\n"
        + "Фраза ассистента:"
    )

def main():
    tts.speak("Постоянный режим активирован")

    last_user_time = time.time()
    next_idle_at = last_user_time + random.uniform(IDLE_MIN_SEC, IDLE_MAX_SEC)

    while True:
        text = (listen() or "").strip()

        # если ничего не распознали — проверяем idle-таймер
        if not text:
            now = time.time()
            if (not tts.is_speaking) and (now >= next_idle_at):
                try:
                    context_list = search_memory("последняя тема разговора интересы предпочтения", limit=6)
                    context = "\n".join(context_list) if context_list else ""
                except Exception:
                    context = ""

                idle_answer = ask_llm(build_idle_prompt(context)).strip()
                if idle_answer:
                    print("Jarvis (idle):", idle_answer)
                    tts.speak(idle_answer)

                # планируем следующий “пинг” позже
                now = time.time()
                next_idle_at = now + random.uniform(IDLE_MIN_SEC, IDLE_MAX_SEC)
            continue

        # получили речь пользователя
        last_user_time = time.time()
        next_idle_at = last_user_time + random.uniform(IDLE_MIN_SEC, IDLE_MAX_SEC)

        print("You:", text)

        if text.lower() in ["выход", "стоп", "закройся"]:
            tts.speak("Выключаюсь")
            break

        # память
        try:
            relevant = search_memory(text, limit=6)
            context = "\n".join(relevant) if relevant else ""
        except Exception:
            context = ""

        # vision по ключевым словам
        if any(word in text.lower() for word in ["экран", "видишь", "покажи"]):
            answer = (ask_vision(text) or "").strip()
        else:
            answer = (ask_llm(build_prompt(text, context)) or "").strip()

        if not answer:
            answer = "Я не получил ответ. Проверь, запущена ли Ollama и правильна ли модель."

        # сохраняем в память
        try:
            save_memory("User", text)
            save_memory("Assistant", answer)
        except Exception:
            pass

        print("Jarvis:", answer)
        tts.speak(answer)


if __name__ == "__main__":
    main()