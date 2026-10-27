"""Modular Streamlit dashboard pages."""

from ml_app.ui.pages.compare_models import page_compare_models
from ml_app.ui.pages.dataset_explorer import page_dataset_explorer
from ml_app.ui.pages.educational_guide import page_educational_guide
from ml_app.ui.pages.overview import page_overview
from ml_app.ui.pages.predictions import page_predictions
from ml_app.ui.pages.train_evaluate import page_train_evaluate

__all__ = [
    "page_overview",
    "page_dataset_explorer",
    "page_train_evaluate",
    "page_compare_models",
    "page_predictions",
    "page_educational_guide",
]
