import os
from openai import OpenAI
from app.config import BASE_DIR

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)