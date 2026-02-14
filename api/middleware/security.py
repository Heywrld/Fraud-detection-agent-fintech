"""
API key authentication and rate limiting middleware.

Provides a Starlette/FastAPI middleware that:
- Validates the ``X-API-Key`` header on every request.
- Exempts public paths (health, docs, webhooks).
- Applies simple in-memory per-key rate limiting (100 req/min default).
"""

import logging
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from config import get_settings

logger = logging.getLogger("fraud_guardian.middleware.security")

# ---------------------------------------------------------------------------
# Path exemptions — these routes skip API key authentication
# ---------------------------------------------------------------------------
EXEMPT_PATHS: set[str] = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}

EXEMPT_PREFIXES: tuple[str, ...] = (
    "/webhooks/paystack",
)

# ---------------------------------------------------------------------------
# Rate limiting state (in-memory — use Redis in production)
# ---------------------------------------------------------------------------
RATE_LIMIT_MAX_REQUESTS: int = 100
RATE_LIMIT_WINDOW_SECONDS: int = 60

# { api_key: [timestamp, ...] }
_request_log: dict[str, list[float]] = defaultdict(list)


def _is_exempt(path: str) -> bool:
    """Return True if the request path is exempt from API key auth."""
    if path in EXEMPT_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in EXEMPT_PREFIXES)


def _check_rate_limit(api_key: str) -> JSONResponse | None:
    """
    Track requests per API key within a sliding time window.

    Returns a 429 JSONResponse if the limit is exceeded, otherwise None.
    """
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS

    # Prune expired timestamps
    _request_log[api_key] = [ts for ts in _request_log[api_key] if ts > cutoff]

    if len(_request_log[api_key]) >= RATE_LIMIT_MAX_REQUESTS:
        logger.warning("Rate limit exceeded for API key: %s...", api_key[:8])
        return JSONResponse(
            status_code=429,
            content={
                "detail": (
                    f"Rate limit exceeded. Maximum {RATE_LIMIT_MAX_REQUESTS} "
                    f"requests per {RATE_LIMIT_WINDOW_SECONDS} seconds."
                )
            },
        )

    _request_log[api_key].append(now)
    return None


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that enforces API key authentication and rate limiting.

    - Reads the ``X-API-Key`` header and compares it to ``settings.api_key``.
    - Skips authentication for exempt paths (health, docs, webhooks).
    - Returns ``401`` with ``{"detail": "Invalid or missing API key"}`` on failure.
    - Returns ``429`` when the per-key rate limit is exceeded.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Let exempt paths through without authentication
        if _is_exempt(path):
            return await call_next(request)

        # --- API key validation ---
        settings = get_settings()
        api_key = request.headers.get("X-API-Key")

        if not api_key or api_key != settings.api_key:
            logger.warning(
                "Unauthorized request to %s from %s",
                path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        # --- Rate limiting (per API key) ---
        rate_limit_response = _check_rate_limit(api_key)
        if rate_limit_response is not None:
            return rate_limit_response

        return await call_next(request)
