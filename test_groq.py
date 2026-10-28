
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise SystemExit(
        "GROQ_API_KEY not found. Check that your .env file is in the project root."
    )

try:
    client = Groq(api_key=api_key)
    models = client.models.list()

    print("SUCCESS: Groq API connection is working!")
    print("\nAvailable models (first 10):")

    for model in models.data[:10]:
        print(f"- {model.id}")

except Exception as error:
    print("Groq connection failed.")
    print(f"Error: {error}")
