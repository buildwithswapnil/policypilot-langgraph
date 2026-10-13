import os

from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found")

if not model_name:
    raise ValueError("OPENAI_MODEL was not found")

print("API key loaded:", bool(api_key))
print("Selected model:", model_name)