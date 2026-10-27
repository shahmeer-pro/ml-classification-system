"""FastAPI application factory and lifecycle configuration."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ml_app import __version__
from ml_app.api.routes import datasets, health, models_routes, predictions, training
from ml_app.exceptions import MLAppException
from ml_app.logging_config import logger


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI classification service."""
    app = FastAPI(
        title="ML Classification System API",
        description=(
            "Production-ready REST API for interactive machine learning classification, "
            "comparisons, and inference based on scikit-learn pipelines."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register domain routers
    app.include_router(health.router)
    app.include_router(datasets.router)
    app.include_router(models_routes.router)
    app.include_router(training.router)
    app.include_router(predictions.router)

    # Domain exception handler
    @app.exception_handler(MLAppException)
    async def ml_exception_handler(request: Request, exc: MLAppException) -> JSONResponse:
        logger.warning(
            "MLAppException on %s %s: status=%d, message=%s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "status_code": exc.status_code,
            },
        )

    # Generic catch-all handler to shield internal stack traces from clients
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled server error on %s %s: %s",
            request.method,
            request.url.path,
            str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred while processing your request.",
                "status_code": 500,
            },
        )

    logger.info("Initialized ML Classification System API v%s", __version__)
    return app


app = create_app()
