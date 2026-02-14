from fastapi import APIRouter
from config import get_settings
import os

router = APIRouter()


@router.get("/health")
async def health_check():
    """System health check endpoint."""
    settings = get_settings()
    model_loaded = os.path.exists(settings.model_path)

    return {
        "status": "healthy",
        "environment": settings.app_env,
        "model_loaded": model_loaded,
        "version": "0.1.0",
    }
