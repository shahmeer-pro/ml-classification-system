"""Custom application exceptions."""


class MLAppException(Exception):
    """Base exception for all ML application domain errors."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DataValidationError(MLAppException):
    """Raised when uploaded or provided data fails validation guardrails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class PreprocessingError(MLAppException):
    """Raised when data transformation, splitting, or encoding fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=422)


class ModelTrainingError(MLAppException):
    """Raised when model instantiation or fitting fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=422)


class PredictionError(MLAppException):
    """Raised when inference or prediction record formatting fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=422)


class ModelNotFoundError(MLAppException):
    """Raised when an in-memory model ID is not found or has expired."""

    def __init__(self, message: str = "Model ID not found. In-memory models reset upon restart."):
        super().__init__(message, status_code=404)
