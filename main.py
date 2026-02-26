from modules.speech_to_text import listen
from modules.text_to_speech import is_speaking
from modules.text_to_speech import speak
from modules.llm import ask_llm
from modules.vision import ask_vision
from modules.chroma_memory import save_memory, search_memory

import time

# защита от реакции на собственный голос
last_response_time = 0
IGNORE_SECONDS = 3


def should_ignore():
    global last_response_time
    return time.time() - last_response_time < IGNORE_SECONDS


def main():

    global last_response_time

    speak("Постоянный режим активирован")

    while True:

        text = listen().strip()

        if not text:
            continue

        print("You:", text)

        # игнорируем если ассистент только что говорил
        if should_ignore():
            continue

        # команды выхода
        if text.lower() in ["выход", "стоп", "закройся"]:
            speak("Выключаюсь")
            break

        # загружаем память
        relevant = search_memory(text, limit=6)
        context = "\n".join(relevant)

        # если вопрос про экран
        if any(word in text.lower() for word in ["экран", "видишь", "покажи"]):

            answer = ask_vision(text)

        else:

            prompt = f"""
            Ты локальный голосовой ассистент.

            Релевантные воспоминания (это факты из прошлых разговоров, используй их):
            {context}

            Пользователь: {text}

            Ответь естественно и по делу.
            """
            answer = ask_llm(prompt)

        # сохраняем память
        save_memory("User", text)
        save_memory("Assistant", answer)

        # говорим
        speak(answer)

        # обновляем время ответа
        last_response_time = time.time()


if __name__ == "__main__":
    main()