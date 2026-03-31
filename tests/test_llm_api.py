"""Test LLM API connection."""

import os

from dotenv import load_dotenv

load_dotenv()

VSE_LLM_API_KEY = os.getenv("VSE_LLM_API_KEY")

assert VSE_LLM_API_KEY, "VSE_LLM_API_KEY not found in environment"

print("API key loaded successfully.")

if __name__ == "__main__":
    import openai

    client = openai.OpenAI(api_key=VSE_LLM_API_KEY, base_url="https://api.vsellm.ru/v1")

    # Get list of models
    models = client.models.list()

    # Print all available models
    print("\nAvailable models:")
    for model in models.data:
        print(f"  ID: {model.id}, Owner: {model.owned_by}")

    # Test request to a model
    print("\nTesting model deepseek/deepseek-v3.2-speciale...")
    response = client.chat.completions.create(
        model="deepseek/deepseek-v3.2-speciale",
        messages=[
            {"role": "system", "content": "You are a trading assistant."},
            {"role": "user", "content": "Analyze BTC price trend. Keep it brief."},
        ],
        max_tokens=100,
    )

    print(f"\nResponse from deepseek/deepseek-v3.2-speciale:")
    print(response.choices[0].message.content)
