"""ML Classification Workbench - Streamlit Application Entrypoint."""

import sys
from pathlib import Path

# Ensure src/ is in sys.path so ml_app is always importable without manual environment flags
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st

from ml_app.ui.api_client import ClassificationApiClient
from ml_app.ui.cache import get_cached_builtin_dataset
from ml_app.ui.pages import (
    page_compare_models,
    page_dataset_explorer,
    page_educational_guide,
    page_overview,
    page_predictions,
    page_train_evaluate,
)

# Page configuration
st.set_page_config(
    page_title="ML Classification Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize API client
api_client = ClassificationApiClient()

# Initialize Session State
if "dataset_choice" not in st.session_state:
    st.session_state.dataset_choice = "builtin"
if "df" not in st.session_state:
    df_init, meta_init = get_cached_builtin_dataset()
    st.session_state.df = df_init
    st.session_state.meta = meta_init
    st.session_state.target_column = meta_init["target_name"]
    st.session_state.feature_columns = meta_init["feature_names"]
if "trained_artifact_id" not in st.session_state:
    st.session_state.trained_artifact_id = None
if "training_result" not in st.session_state:
    st.session_state.training_result = None
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

# Sidebar status indicator (cached health check)
backend_status = api_client.check_health()
if backend_status.get("status") == "healthy":
    st.sidebar.success(f"🟢 API Connected: v{backend_status.get('version')}")
else:
    st.sidebar.info("🔵 Standalone Direct Mode (Embedded Services)")

# Session Reset Option
if st.sidebar.button(
    "🔄 Reset Current Session", help="Clear in-memory models, caches, and reset state"
):
    st.session_state.clear()
    st.rerun()

# Modern Streamlit Multi-Page Navigation (only executes active page)
pg = st.navigation(
    [
        st.Page(page_overview, title="1. Overview & Architecture", icon="🏠", default=True),
        st.Page(page_dataset_explorer, title="2. Dataset Explorer", icon="📊"),
        st.Page(page_train_evaluate, title="3. Train & Evaluate", icon="🚀"),
        st.Page(page_compare_models, title="4. Compare Models", icon="⚡"),
        st.Page(page_predictions, title="5. Predictions & Inference", icon="🔮"),
        st.Page(page_educational_guide, title="6. Educational Guide & Limitations", icon="📚"),
    ]
)

pg.run()
