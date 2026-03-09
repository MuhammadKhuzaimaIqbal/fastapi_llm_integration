import os
import logging
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env from that folder
load_dotenv(os.path.join(BASE_DIR, ".env"))

print("GROQ_API_KEY =", os.getenv("GROQ_API_KEY"))

# 1. Initialize OpenAI Client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

app = FastAPI(title="AI Wrapper Service")

# 2. Setup Logging (to a file as requested)
logging.basicConfig(
    filename="api_usage.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 3. Request Model (What we expect from the user)
class SummarizeRequest(BaseModel):
    text: str

class ExtractRequest(BaseModel):
    text: str

class ExtractedData(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    company: str | None = None

class ClassifyRequest(BaseModel):
    feedback: str

class ClassificationResponse(BaseModel):
    label: str
    confidence: float

class TranslateRequest(BaseModel):
    text: str
    target_language: str

# 4. Utility function to load prompts
def load_prompt(filename: str):
    path = os.path.join(BASE_DIR, "prompts", filename)
    with open(path, "r") as f:
        return f.read()

# 5. The Core AI Logic with Retry Logic
# This will retry up to 3 times if the API fails, waiting longer each time
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

# 6. The Endpoint
@app.post("/ai/summarize")
async def summarize(request: SummarizeRequest):
    try:
        system_msg = load_prompt("summarize.txt")

        # Call the AI
        completion = call_llm(system_msg, request.text)

        summary = completion.choices[0].message.content

        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens

        # Log the usage
        logging.info(
            f"Endpoint: /summarize | Input Tokens: {input_tokens} | Output Tokens: {output_tokens}"
        )

        return {
            "summary": summary,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }

    except RetryError as re:
        cause = re.last_attempt.exception()
        logging.error(f"Error in /summarize (unwrapped): {repr(cause)}")
        raise HTTPException(status_code=500, detail=str(cause))

    except Exception as e:
        logging.error(f"Error in /summarize: {repr(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {"status": "online"}

@app.post("/ai/extract")
async def extract_data(request: ExtractRequest):
    try:
        system_msg = load_prompt("extract.txt")
        
        # Using the centralized call_llm for Retry Logic & JSON mode
        response = call_llm(
            system_prompt=system_msg, 
            user_content=request.text, 
            temperature=0.1, 
            json_mode=True
        )

        data = json.loads(response.choices[0].message.content)
        
        # Log requirements: Detailed token tracking
        logging.info(
            f"REQ: /extract | IN: {response.usage.prompt_tokens} | OUT: {response.usage.completion_tokens}"
        )

        return {
            "data": data,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
        }
    except Exception as e:
        logging.error(f"Error in /extract: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/ai/classify")
async def classify_feedback(request: ClassifyRequest):
    try:
        system_msg = load_prompt("classify.txt")
        
        response = call_llm(
            system_prompt=system_msg, 
            user_content=request.feedback, 
            temperature=0, 
            json_mode=True
        )

        data = json.loads(response.choices[0].message.content)
        
        logging.info(
            f"REQ: /classify | IN: {response.usage.prompt_tokens} | OUT: {response.usage.completion_tokens}"
        )

        return {
            "classification": data,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
        }
    except Exception as e:
        logging.error(f"Error in /classify: {str(e)}")
        raise HTTPException(status_code=500, detail="Classification failed.")


@app.post("/ai/translate")
async def translate_text(request: TranslateRequest):
    try:
        system_msg = load_prompt("translate.txt")
        user_msg = f"Target Language: {request.target_language}\nText: {request.text}"
        
        response = call_llm(
            system_prompt=system_msg, 
            user_content=user_msg, 
            temperature=0.3, 
            json_mode=True
        )

        data = json.loads(response.choices[0].message.content)

        logging.info(
            f"REQ: /translate | IN: {response.usage.prompt_tokens} | OUT: {response.usage.completion_tokens}"
        )

        return {
            "translation": data,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
        }
    except Exception as e:
        logging.error(f"Error in /translate: {str(e)}")
        raise HTTPException(status_code=500, detail="Translation failed.")