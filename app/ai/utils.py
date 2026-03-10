import os, json
from tenacity import retry, stop_after_attempt, wait_exponential
from app.ai.client import client
from app.config import BASE_DIR

def load_prompt(filename: str):
    path = os.path.join(BASE_DIR, "ai", "prompts", filename)
    with open(path, "r") as f:
        return f.read()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def call_llm(system_prompt, user_content, temperature=0.5, json_mode=False):
    extra_args = {}
    if json_mode:
        extra_args["response_format"] = {"type": "json_object"}

    return client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=500,
        **extra_args
    )