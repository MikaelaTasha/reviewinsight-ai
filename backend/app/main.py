import logging

from fastapi import FastAPI

from app.api.upload import router as upload_router
from app.api.analyze import router as analyze_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="ReviewInsight AI API",
    description="Backend API for evidence-based product review analysis.",
    version="0.1.0",
)

app.include_router(upload_router)
app.include_router(analyze_router)

@app.get("/")
def read_root() -> dict[str, str]:
    logger.info("Root endpoint accessed")

    return {
        "message": "ReviewInsight AI backend is running",
        "status": "success",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    logger.info("Health check requested")

    return {"status": "healthy"}