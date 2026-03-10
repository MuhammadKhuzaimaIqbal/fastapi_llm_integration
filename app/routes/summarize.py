from fastapi import APIRouter, HTTPException
import logging
from app.models.requests import SummarizeRequest
from app.ai.utils import call_llm, load_prompt
from tenacity import RetryError

router = APIRouter()

@router.post("/ai/summarize")
async def summarize(request: SummarizeRequest):
    try:
        system_msg = load_prompt("summarize.txt")
        completion = call_llm(system_msg, request.text)
        summary = completion.choices[0].message.content

        logging.info(
            f"Endpoint: /summarize | Input Tokens: {completion.usage.prompt_tokens} | Output Tokens: {completion.usage.completion_tokens}"
        )

        return {
            "summary": summary,
            "usage": {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
            },
        }
    except RetryError as re:
        raise HTTPException(status_code=500, detail=str(re.last_attempt.exception()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))