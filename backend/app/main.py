from fastapi import FastAPI

app = FastAPI(
    title="ReviewInsight AI API",
    description="Backend API for evidence-based product review analysis.",
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a simple status message for the API."""
    return {
        "message": "ReviewInsight AI backend is running",
        "status": "success",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the current health status of the backend."""
    return {"status": "healthy"}