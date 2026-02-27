from modules.speech_to_text import listen
from modules.llm import ask_llm
from modules.vision import ask_vision
from modules.chroma_memory import save_memory, search_memory
import modules.text_to_speech as tts


def main():
    tts.speak("Постоянный режим активирован")

    while True:
        text = (listen() or "").strip()
        if not text:
            continue

        print("You:", text)

        if text.lower() in ["выход", "стоп", "закройся"]:
            tts.speak("Выключаюсь")
            break

        try:
            relevant = search_memory(text, limit=6)
            context = "\n".join(relevant) if relevant else ""
        except Exception as e:
            print("Memory error:", repr(e))
            context = ""

        try:
            if any(word in text.lower() for word in ["экран", "видишь", "покажи"]):
                answer = ask_vision(text)
            else:
                prompt = (
                    "Ты локальный голосовой ассистент.\n\n"
                    "Релевантные воспоминания (это факты из прошлых разговоров, используй их):\n"
                    f"{context}\n\n"
                    f"Пользователь: {text}\n\n"
                    "Ответь естественно и по делу."
                )
                answer = ask_llm(prompt)
        except Exception as e:
            print("LLM/Vision error:", repr(e))
            answer = f"Ошибка при обработке запроса: {repr(e)}"

        answer = (answer or "").strip()
        if not answer:
            answer = "Я не получил ответ. Проверь, запущена ли Ollama и правильна ли модель."

        try:
            save_memory("User", text)
            save_memory("Assistant", answer)
        except Exception as e:
            print("Save memory error:", repr(e))

        print("Jarvis:", answer)
        tts.speak(answer)


if __name__ == "__main__":
    main()