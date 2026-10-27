"""Health check endpoint."""

from fastapi import APIRouter

from ml_app import __version__
from ml_app.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service status and version information."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        service="ML Classification System API",
    )
