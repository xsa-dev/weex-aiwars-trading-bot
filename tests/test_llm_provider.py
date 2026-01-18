import os

import openai
from dotenv import load_dotenv

load_dotenv()

VSE_LLM_API_KEY = os.getenv("VSE_LLM_API_KEY")

assert VSE_LLM_API_KEY

print("api not bad.")

if __name__ == "__main__":
    client = openai.OpenAI(api_key=VSE_LLM_API_KEY, base_url="https://api.vsellm.ru/v1")

    # Получение списка моделей
    models = client.models.list()

    # Вывод всех доступных моделей
    for model in models.data:
        print(f"ID: {model.id}, Owner: {model.owned_by}")
