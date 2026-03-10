from pydantic import BaseModel

class SummarizeRequest(BaseModel):
    text: str

class ExtractRequest(BaseModel):
    text: str

class ClassifyRequest(BaseModel):
    feedback: str

class TranslateRequest(BaseModel):
    text: str
    target_language: str