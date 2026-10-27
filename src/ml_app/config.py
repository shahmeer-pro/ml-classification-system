"""Application configuration and constants."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Application settings and guardrail limits."""

    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

    # Resource and protection guardrails for uploads
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    MAX_ROWS: int = int(os.getenv("MAX_ROWS", "50000"))
    MIN_ROWS: int = 10
    MAX_COLS: int = int(os.getenv("MAX_COLS", "100"))

    # Modeling defaults
    DEFAULT_TEST_SIZE: float = 0.20
    DEFAULT_RANDOM_SEED: int = 42
    DEFAULT_CV_SPLITS: int = 5

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
