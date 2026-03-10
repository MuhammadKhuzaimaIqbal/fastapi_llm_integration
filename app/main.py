from fastapi import FastAPI
from app.routes import summarize, extract, classify, translate

app = FastAPI(title="AI Wrapper Service")

app.include_router(summarize.router)
app.include_router(extract.router)
app.include_router(classify.router)
app.include_router(translate.router)

@app.get("/")
def home():
    return {"status": "online"}