from pydantic import BaseModel

class ExtractedData(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    company: str | None = None

class ClassificationResponse(BaseModel):
    label: str
    confidence: float