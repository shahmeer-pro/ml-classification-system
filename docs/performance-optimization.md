# Performance Optimization & Architectural Evaluation Report

## 1. Executive Summary

This report documents the performance audit, bottleneck measurement, and optimizations applied to the **Machine Learning Classification Workbench**.

Prior to optimization, sidebar navigation and tab switching took **~3.3 seconds per click**, creating a sluggish user experience. After profiling and pinpointing the root causes, we implemented targeted optimizations in caching, network requests, and page modularity.

### Primary Results:
- **Warm Navigation Latency**: Decreased from **~3,300 ms** to **~45 ms** (**>98% latency reduction**).
- **API Health Check Overhead**: Decreased from **2,903.18 ms** to **0.85 ms** (**99.97% reduction**).
- **Dataset Loading Overhead**: Decreased from **47.80 ms** to **1.66 ms** (**96.5% reduction**).
- **Dataset Validation Overhead**: Decreased from **51.97 ms** to **2.01 ms** (**96.1% reduction**).
- **Test Suite**: 35 automated tests passing with zero failures.

---

## 2. Phase 1 — Measured Baseline & Root Cause Analysis

We created a deterministic diagnostic harness (`scratch/benchmark_baseline.py`) to measure the performance of every layer before altering code.

### Baseline Measurements (Measured on Python 3.13):

| Operation | Baseline Latency | Root Cause |
| :--- | :---: | :--- |
| **`api_check_health_standalone`** | **2,903.18 ms** | **Primary Bottleneck**: `ClassificationApiClient` was instantiated at the top level of `app.py` on every rerun. On line 57, `check_health()` attempted an HTTP connection to `http://127.0.0.1:8000/health`. In standalone mode (when FastAPI is not running), the OS socket connection attempt blocked for **2.9 seconds** before timing out on *every single tab switch*. |
| **`load_builtin_dataset`** | **47.80 ms** | Uncached: Scikit-learn dataset loader was called from scratch whenever Dataset Explorer was active or initialized. |
| **`validate_dataset`** | **51.97 ms** | Uncached: Column scans, type checks, missing value counters, and class balance loops re-executed on every widget change. |
| **`plot_confusion_matrix`** | **274.78 ms** | Uncached Plotly figure generation: Heatmap figures were re-generated from scratch on page transitions. |
| **`plot_model_comparison`** | **223.50 ms** | Uncached Plotly figure generation: Multi-bar comparison was recomputed on page switches. |
| **`plot_class_distribution`** | **84.58 ms** | Uncached Plotly figure generation. |
| **Monolithic Script Execution** | **Full 600 lines** | `app.py` was a single script where every page's logic resided in one global `if/elif` block. Streamlit parsed the entire script on every rerun. |

---

## 3. Phase 2 & 3 — Architectural Optimizations Implemented

### A. Non-Blocking Cached Health Check
- Created `src/ml_app/ui/cache.py` with `@st.cache_data(ttl=60)`.
- Reduced socket timeout from 2.0s to 0.3s.
- `check_api_health_cached` memoizes the status for 60 seconds.
- Warm health check dropped from **2,903 ms to 0.85 ms**.

### B. Safe Streamlit Caching for Data Operations
- Implemented `get_cached_builtin_dataset()`: loads static Breast Cancer Wisconsin data once and stores in memory cache (**1.66 ms** retrieval).
- Implemented `get_cached_validation_summary()`: memoizes CSV validation summaries keyed by content bytes (`max_entries=10`). Avoids redundant column scanning when switching between tabs.
- Preserved user privacy: mutable user-fitted models are **never stored in shared global caches**, but kept in per-session state (`st.session_state`) and in-memory training registry.

### C. Modern Modular Multi-Page Navigation (`st.navigation`)
- Decomposed the 600-line monolithic script into independent page modules under `src/ml_app/ui/pages/`:
  - `page_overview()`
  - `page_dataset_explorer()`
  - `page_train_evaluate()`
  - `page_compare_models()`
  - `page_predictions()`
  - `page_educational_guide()`
- Streamlit's `st.navigation` executes **only the active page function**.
- Code, widgets, and charts for hidden pages are never parsed or executed during navigation.

### D. Session-Preserved Plotly Figures
- When "Train Model" or "Compare Models" is clicked, generated figures (`fig_cm`, `fig_roc`, `fig_comp`) are stored directly in `st.session_state`.
- Navigating away to "Predictions" and returning to "Train & Evaluate" renders figures instantaneously without re-running `plot_confusion_matrix()` or `plot_roc_curve()`.

### E. Explicit Session Reset Guard
- Added a `🔄 Reset Current Session` button in the sidebar that clears `st.session_state` and model references cleanly on demand.

---

## 4. Phase 4 — Post-Optimization Performance Comparison

Measured using `scratch/benchmark_optimized.py`:

| Metric / Operation | Baseline (Before) | Optimized (After) | Improvement |
| :--- | :---: | :---: | :---: |
| **Health Check (Warm Navigation)** | 2,903.18 ms | **0.85 ms** | **99.97% faster** |
| **Health Check (Cold / Initial)** | 2,903.18 ms | 1,330.47 ms | 54.2% faster |
| **Dataset Loading (Warm)** | 47.80 ms | **1.66 ms** | **96.5% faster** |
| **Dataset Validation (Warm)** | 51.97 ms | **2.01 ms** | **96.1% faster** |
| **Single Record Prediction** | 65.41 ms | **23.16 ms** | **64.6% faster** |
| **Page Navigation Responsiveness** | **~3,300 ms** | **~45 ms** | **>98% faster** |

---

## 5. Phase 5 — Technology Stack Evaluation: Streamlit vs. Next.js

We evaluated whether migrating the frontend to Next.js is justified.

### Option A: Optimized Streamlit + FastAPI + Python ML Services
- **Navigation Latency**: ~45 ms (imperceptible to users).
- **Architecture**: Single unified Python codebase or two-service setup.
- **Hosting**: Deploys to **Streamlit Community Cloud completely free** in a single container with zero cloud infrastructure costs.
- **Cold Starts**: **Zero cold starts** in standalone mode (Python service layer runs in-process).
- **Data Serialization**: In-memory Python object passing; zero network bandwidth or JSON serialization overhead for 50k-row datasets.

### Option B: Next.js + FastAPI + Python ML Services
- **Navigation Latency**: ~10–20 ms (client-side React routing).
- **Architecture**: 2 separate applications: TypeScript/Next.js frontend + Python/FastAPI backend.
- **Hosting**: Next.js on Vercel Hobby (free) + FastAPI on Render Free Tier.
- **The Critical Free-Tier Bottleneck (Render Cold Starts)**:
  - Render free-tier web services spin down after 15 minutes of inactivity.
  - When a user on Next.js clicks "Train Model" or uploads a dataset, the request hits a sleeping Render container, causing a **50-second cold-start delay**!
  - In Streamlit standalone mode, the entire ML engine is loaded in RAM immediately, completely avoiding cold starts.
- **Development & Maintenance Overhead**:
  - Requires maintaining TypeScript schemas, React state (TanStack Query / Zustand), Recharts or Plotly.js bindings, and dual CI/CD pipelines.

### Recommendation:
> **Keep Streamlit (Option A)**.
> With the measured ~45 ms warm navigation latency, the sluggishness is completely resolved. Migrating to Next.js would introduce Render free-tier cold starts (50s delay), cross-origin network latency, and high maintenance overhead without offering user-facing speed benefits.

---

## 6. Phase 6 — Free Deployment & Database Architecture

### A. Recommended Free Hosting Platform
- **Streamlit Community Cloud (`share.streamlit.io`)**:
  - Connects directly to GitHub.
  - Single-container deployment.
  - Standalone mode runs Python ML services in memory with zero cold starts.

### B. Free Database Evaluation (PostgreSQL)
- **Candidates**: Neon Free Tier (0.5 GB storage, serverless) or Supabase Free Tier (500 MB).
- **Do you need a database?**
  - **For interactive classification & prediction**: **NO**. In-memory processing guarantees data privacy (datasets discarded after inference) and eliminates database connection limits.
  - **For ML Experiment Tracking**: **YES (Optional)**. If you wish to save training run benchmarks (Model Name, Accuracy, F1, Hyperparameters JSON, Timestamp) across sessions, a lightweight schema can be added without storing sensitive dataset rows.
