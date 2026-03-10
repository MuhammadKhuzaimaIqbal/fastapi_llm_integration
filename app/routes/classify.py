from fastapi import APIRouter, HTTPException
import logging, json
from app.models.requests import ClassifyRequest
from app.ai.utils import call_llm, load_prompt

router = APIRouter()

@router.post("/ai/classify")
async def classify_feedback(request: ClassifyRequest):
    try:
        system_msg = load_prompt("classify.txt")
        response = call_llm(system_msg, request.feedback, temperature=0, json_mode=True)
        data = json.loads(response.choices[0].message.content)

        logging.info(
            f"REQ: /classify | IN: {response.usage.prompt_tokens} | OUT: {response.usage.completion_tokens}"
        )

        return {"classification": data, "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens
        }}
    except Exception:
        raise HTTPException(status_code=500, detail="Classification failed.")