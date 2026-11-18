import os

from dotenv import load_dotenv
from groq import Groq


# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found in the .env file."
    )


# Create Groq client
client = Groq(api_key=api_key)


# Simple test request
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Explain what a glioma is in one short sentence."
        }
    ],
    temperature=0.2,
    max_tokens=100
)


print()
print("====================================")
print("GROQ CONNECTION TEST")
print("====================================")

print(response.choices[0].message.content)

print()
print("Groq connection successful.")
print("====================================")