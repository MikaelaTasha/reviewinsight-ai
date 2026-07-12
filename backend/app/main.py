from fastapi import FastAPI
from app.api.upload import router as upload_router

app = FastAPI(
    title="ReviewInsight AI API",
    description="Backend API for evidence-based product review analysis.",
    version="0.1.0"
)

app.include_router(upload_router)

@app.get("/")
def read_root():
    return {
        "message": "ReviewInsight AI backend is running",
        "status": "success"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}