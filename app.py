"""ML Studio - Machine Learning Classification Workbench."""

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
from ml_app.ui.theme import inject_custom_theme, render_sidebar_brand

# Page configuration
st.set_page_config(
    page_title="ML Studio — Classification Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject centralized CSS design tokens
inject_custom_theme()

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

# Render Sidebar Branding
render_sidebar_brand()

# Modern Streamlit Multi-Page Navigation (only executes active page)
pg = st.navigation(
    [
        st.Page(page_overview, title="Overview", icon=":material/dashboard:", default=True),
        st.Page(page_dataset_explorer, title="Dataset Explorer", icon=":material/table_chart:"),
        st.Page(page_train_evaluate, title="Train & Evaluate", icon=":material/model_training:"),
        st.Page(page_compare_models, title="Compare Models", icon=":material/compare_arrows:"),
        st.Page(page_predictions, title="Predictions & Inference", icon=":material/psychology:"),
        st.Page(page_educational_guide, title="Learning Guide", icon=":material/menu_book:"),
    ]
)

# Render Sidebar Footer
with st.sidebar:
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    backend_status = api_client.check_health()
    if backend_status.get("status") == "healthy":
        st.caption(f"🟢 **FastAPI Connected** (`v{backend_status.get('version')}`)")
    else:
        st.caption("🔵 **Standalone Direct Mode** (Embedded)")

    if st.button(
        "Reset Session",
        type="secondary",
        use_container_width=True,
        help="Clear in-memory models and return to defaults",
    ):
        st.session_state.clear()
        st.rerun()

    st.markdown(
        """
        <div style="font-size: 0.72rem; color: #64748B; text-align: center; margin-top: 0.75rem;">
            ML Studio v1.0.0 · Production Ready
        </div>
        """,
        unsafe_allow_html=True,
    )

pg.run()
