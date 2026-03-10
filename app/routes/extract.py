from fastapi import APIRouter, HTTPException
import logging, json
from app.models.requests import ExtractRequest
from app.ai.utils import call_llm, load_prompt

router = APIRouter()

@router.post("/ai/extract")
async def extract_data(request: ExtractRequest):
    try:
        system_msg = load_prompt("extract.txt")
        response = call_llm(system_msg, request.text, temperature=0.1, json_mode=True)
        data = json.loads(response.choices[0].message.content)

        logging.info(
            f"REQ: /extract | IN: {response.usage.prompt_tokens} | OUT: {response.usage.completion_tokens}"
        )

        return {"data": data, "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens
        }}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")