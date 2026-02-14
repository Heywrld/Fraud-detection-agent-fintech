from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from config import get_settings
from api.middleware.security import APIKeyMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_guardian")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    settings = get_settings()
    logger.info(f"Fraud Guardian AI starting in {settings.app_env} mode")

    # Load ML model on startup
    try:
        from ml.predict import load_model
        load_model()
        logger.info("ML model loaded successfully")
    except Exception as e:
        logger.warning(f"ML model not loaded: {e}. Train the model first with: python -m ml.train")

    yield

    logger.info("Fraud Guardian AI shutting down")


app = FastAPI(
    title="Fraud Guardian AI",
    description="Real-time fraud detection agent for Nigerian fintechs",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if get_settings().is_development else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key authentication & rate limiting middleware
app.add_middleware(APIKeyMiddleware)

# Register routes
from api.routes.health import router as health_router
from api.routes.webhooks import router as webhooks_router
from api.routes.transactions import router as transactions_router
from api.routes.alerts import router as alerts_router
from api.routes.dashboard import router as dashboard_router

app.include_router(health_router, tags=["Health"])
app.include_router(webhooks_router, prefix="/webhooks", tags=["Webhooks"])
app.include_router(transactions_router, prefix="/transactions", tags=["Transactions"])
app.include_router(alerts_router, prefix="/alerts", tags=["Alerts"])
app.include_router(dashboard_router, prefix="/dashboard", tags=["Dashboard"])
