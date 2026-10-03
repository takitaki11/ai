import os
import time
from dotenv import load_dotenv
from groq import Groq
from google import genai
from openai import OpenAI


# ==========================================
# ЗАГРУЖАЕМ API-КЛЮЧИ
# ==========================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")


# ==========================================
# СОЗДАЁМ КЛИЕНТОВ
# ==========================================

groq_client = Groq(
    api_key=groq_api_key
)

google_client = genai.Client(
    api_key=google_api_key
)

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key
)


# ==========================================
# 1. GROQ
# ==========================================

def ask_groq(prompt):
    start_time = time.time()

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    end_time = time.time()

    answer = response.choices[0].message.content
    response_time = end_time - start_time
    tokens = response.usage.total_tokens

    return answer, response_time, tokens


# ==========================================
# 2. GOOGLE GEMINI
# ==========================================

def ask_gemini(prompt):
    start_time = time.time()

    response = google_client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    end_time = time.time()

    answer = response.text
    response_time = end_time - start_time
    tokens = response.usage_metadata.total_token_count

    return answer, response_time, tokens


# ==========================================
# 3. OPENROUTER
# ==========================================

def ask_openrouter(prompt):
    start_time = time.time()

    response = openrouter_client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    end_time = time.time()

    answer = response.choices[0].message.content
    response_time = end_time - start_time

    if response.usage is not None:
        tokens = response.usage.total_tokens
    else:
        tokens = 0

    return answer, response_time, tokens


# ==========================================
# ОДИНАКОВЫЙ ПРОМПТ ДЛЯ ВСЕХ МОДЕЛЕЙ
# ==========================================

prompt = "Объясни разницу между машинным обучением и LLM"


# Здесь будем хранить результаты
results = []


# ==========================================
# ЗАПУСК GROQ
# ==========================================

print("\n==============================")
print("GROQ")
print("==============================\n")

try:
    groq_answer, groq_time, groq_tokens = ask_groq(prompt)

    print(groq_answer)

    print("\n--- Результаты Groq ---")
    print(f"Время ответа: {groq_time:.2f} сек.")
    print(f"Количество токенов: {groq_tokens}")

    results.append(
        ("Groq GPT-OSS 120B", groq_time, groq_tokens)
    )

except Exception as error:
    print("Ошибка Groq:")
    print(error)

    results.append(
        ("Groq GPT-OSS 120B", None, None)
    )


# ==========================================
# ЗАПУСК GOOGLE GEMINI
# ==========================================

print("\n==============================")
print("GOOGLE GEMINI")
print("==============================\n")

try:
    gemini_answer, gemini_time, gemini_tokens = ask_gemini(prompt)

    print(gemini_answer)

    print("\n--- Результаты Gemini ---")
    print(f"Время ответа: {gemini_time:.2f} сек.")
    print(f"Количество токенов: {gemini_tokens}")

    results.append(
        ("Gemini 3.8 Flash", gemini_time, gemini_tokens)
    )

except Exception as error:
    print("Gemini временно недоступен.")
    print("Ошибка:")
    print(error)

    results.append(
        ("Gemini 3.8 Flash", None, None)
    )


# ==========================================
# ЗАПУСК OPENROUTER
# ==========================================

print("\n==============================")
print("OPENROUTER")
print("==============================\n")

try:
    openrouter_answer, openrouter_time, openrouter_tokens = (
        ask_openrouter(prompt)
    )

    print(openrouter_answer)

    print("\n--- Результаты OpenRouter ---")
    print(f"Время ответа: {openrouter_time:.2f} сек.")
    print(f"Количество токенов: {openrouter_tokens}")

    results.append(
        ("OpenRouter Free", openrouter_time, openrouter_tokens)
    )

except Exception as error:
    print("Ошибка OpenRouter:")
    print(error)

    results.append(
        ("OpenRouter Free", None, None)
    )


# ==========================================
# ИТОГОВАЯ ТАБЛИЦА
# ==========================================

print("\n======================================================")
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("======================================================")

print(
    f"{'Модель':<25}"
    f"{'Время (сек)':<15}"
    f"{'Токены':<15}"
)

print("-" * 55)


for model, response_time, tokens in results:

    if response_time is None:
        time_text = "Ошибка"
    else:
        time_text = f"{response_time:.2f}"

    if tokens is None:
        tokens_text = "Ошибка"
    else:
        tokens_text = str(tokens)

    print(
        f"{model:<25}"
        f"{time_text:<15}"
        f"{tokens_text:<15}"
    )