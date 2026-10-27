"""Dataset exploration and validation endpoints."""

from fastapi import APIRouter, File, Form, UploadFile

from ml_app.api.schemas import BuiltinDatasetResponse, ValidationResponse
from ml_app.data.loader import load_builtin_dataset, load_csv_from_bytes
from ml_app.data.validation import validate_dataset

router = APIRouter(prefix="/api/v1/datasets", tags=["Datasets"])


@router.get("/builtin", response_model=BuiltinDatasetResponse)
async def get_builtin_dataset_metadata() -> BuiltinDatasetResponse:
    """Return metadata and documentation for the built-in Breast Cancer Wisconsin dataset."""
    _, meta = load_builtin_dataset()
    return BuiltinDatasetResponse(**meta)


@router.post("/validate", response_model=ValidationResponse)
async def validate_uploaded_dataset(
    file: UploadFile = File(...),
    target_column: str | None = Form(None),
) -> ValidationResponse:
    """Accept a CSV upload and return a structured validation summary."""
    file_bytes = await file.read()
    filename = file.filename or "upload.csv"
    df = load_csv_from_bytes(file_bytes, filename=filename)

    summary = validate_dataset(
        df=df,
        target_column=target_column if target_column else None,
    )

    return ValidationResponse(
        is_valid=summary.is_valid,
        num_rows=summary.num_rows,
        num_cols=summary.num_cols,
        column_names=summary.column_names,
        dtypes=summary.dtypes,
        missing_counts=summary.missing_counts,
        missing_percentages=summary.missing_percentages,
        numeric_columns=summary.numeric_columns,
        categorical_columns=summary.categorical_columns,
        constant_columns=summary.constant_columns,
        target_column=summary.target_column,
        target_classes=summary.target_classes,
        class_distribution=summary.class_distribution,
        warnings=summary.warnings,
        errors=summary.errors,
    )
