from fastapi import APIRouter, HTTPException
import logging, json
from app.models.requests import TranslateRequest
from app.ai.utils import call_llm, load_prompt

router = APIRouter()

@router.post("/ai/translate")
async def translate_text(request: TranslateRequest):
    try:
        system_msg = load_prompt("translate.txt")
        user_msg = f"Target Language: {request.target_language}\nText: {request.text}"
        response = call_llm(system_msg, user_msg, temperature=0.3, json_mode=True)
        data = json.loads(response.choices[0].message.content)

        logging.info(
            f"REQ: /translate | IN: {response.usage.prompt_tokens} | OUT: {response.usage.completion_tokens}"
        )

        return {"translation": data, "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens
        }}
    except Exception:
        raise HTTPException(status_code=500, detail="Translation failed.")