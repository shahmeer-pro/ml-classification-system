# Free-Tier Cloud Deployment Guide

This application requires **no external database** and can be deployed entirely on free hosting platforms.

---

## 1. Streamlit Community Cloud (Recommended: Quickest Free Deployment)

Streamlit Community Cloud allows free hosting of Streamlit applications directly linked to a public GitHub repository.

### Steps:
1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) using your GitHub account.
3. Click **New app**.
4. Configure the settings:
   - **Repository**: `your-username/your-repo`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy!**

> **Note on Standalone Mode**:
> In Streamlit Community Cloud, the application automatically runs in **Standalone Direct Mode**.
> `ClassificationApiClient` detects that the external FastAPI server is not present and invokes the Python service layer (`TrainingService`, `PredictionService`) directly in-process. All features (dataset upload, training, evaluation, comparison, prediction, downloads) operate without needing a second deployed server!

---

## 2. Hugging Face Spaces (Alternative Free Dashboard)

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** as the Space SDK.
3. Clone the Space repository and copy the project files (`app.py`, `src/`, `requirements.txt`, etc.).
4. Push to the Space Git remote:
   ```bash
   git add .
   git commit -m "Deploy ML Classification System"
   git push origin main
   ```
5. Hugging Face will automatically install dependencies from `requirements.txt` and launch `app.py`.

---

## 3. Render (Optional: Standalone FastAPI Deployment)

To deploy the FastAPI backend as a dedicated REST API service:

1. Create a free account at [render.com](https://render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Fill in the settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn ml_app.api.main:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**.
6. Note the public URL generated (e.g., `https://your-api.onrender.com`).

### Connecting Streamlit to your Render API:
In your Streamlit deployment settings, add an environment variable or Secret:
```toml
API_BASE_URL = "https://your-api.onrender.com"
```

---

## 4. Free-Tier Considerations & Known Behaviors

- **Cold Starts**: Render free-tier services spin down after 15 minutes of inactivity. The first incoming request may experience a 30-50 second delay while the container spins up.
- **In-Memory Volatility**: Because models are stored in RAM (`TrainingService._artifacts`), any platform restart or container recycling will clear existing trained model IDs. Users can train a model again in seconds.
- **File Limits**: Uploaded CSV files are processed transiently in memory with an enforced 10MB guardrail.
