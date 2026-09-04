import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")

if not api_key:
    raise ValueError("OPENAI_API_KEY nije pronađen u .env fajlu.")

client = OpenAI(api_key=api_key)

response = client.responses.create(
    model=model,
    input=(
        "Answer briefly and factually. "
        "What is the role of insulin in the human body?"
    ),
)

print("Model:", model)
print("Odgovor:")
print(response.output_text)