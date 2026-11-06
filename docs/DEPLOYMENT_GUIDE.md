# Production Deployment Guide

**Project**: Intelligent Fine-Grained Vehicle Recognition and Analysis System  
**Version**: 1.0.0  
**Target Architecture**: Streamlit UI Dashboard + FastAPI REST Service  

---

## 1. Executive Summary & Recommended Hosting Platforms

The Intelligent Vehicle Recognition System is designed with a **decoupled, CPU-efficient architecture** optimized for deployment on free or standard cloud compute tiers without requiring high-cost GPU infrastructure.

| Deployment Target | Hosting Platform | Compute Specification | Suitability | Estimated Cold-Start |
| :--- | :--- | :--- | :---: | :---: |
| **Option 1 (Primary)** | **Streamlit Community Cloud** | 1–2 shared vCPUs, 1.5 GB RAM | **Recommended** (Direct GitHub repo deployment) | ~15–25 seconds |
| **Option 2 (Container)** | **Hugging Face Spaces (Docker)** | 2 vCPUs, 16 GB RAM (Free) | **Excellent** (Native container sandbox) | ~20 seconds |
| **Option 3 (PaaS API)** | **Render / Railway (Web Service)** | 0.5–1 vCPU, 512MB–1GB RAM | **Good** for decoupled FastAPI microservice | ~30 seconds |
| **Option 4 (On-Premises)** | **Docker / Docker Compose** | Multi-core x86_64, 4+ GB RAM | **Production Grade** (Self-hosted) | < 5 seconds |

---

## 2. Model Storage & Artifact Handling Strategy

In strict adherence to repository hygiene and cloud deployment limits:
- **Zero Raw Datasets in Production**: The 16,185-image Stanford Cars dataset is **never** uploaded to hosting providers. Only the lightweight taxonomy metadata (`data/processed/class_mapping.json`, 139 KB) is packaged with the application.
- **No Heavy Binaries in Git**: Model weight files (`.pt`, `.joblib`) are excluded via `.gitignore` and `.dockerignore`.
- **Dynamic Checkpoint Acquisition**:
  1. **YOLOv8n Vehicle Detector (6.2 MB)**: Ultralytics automatically downloads the official nano checkpoint (`yolov8n.pt`) on first invocation if custom fine-tuned weights (`models/detection/best.pt`) are not mounted.
  2. **ResNet50 Deep Classifier (91.2 MB)**: Pretrained torchvision/timm weights are loaded dynamically or fetched from external release storage via `scripts/download_models.py`.
  3. **Automated Fetch Script**: Run `python scripts/download_models.py` during deployment build phases.

---

## 3. Required Environment Variables

Configure these variables in your platform's dashboard or local `.env` file (see `.env.example`):

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `production` | Deployment mode (`production`, `development`) |
| `PORT` | `8501` (or `8000`) | Dynamic port assigned by hosting provider |
| `HOST` | `0.0.0.0` | IP binding address |
| `DEVICE` | `cpu` | Hardware acceleration target (`cpu`, `cuda`, `auto`) |
| `API_URL` | `http://127.0.0.1:8000` | Backend API URL for decoupled Streamlit mode |
| `BACKEND_URL` | `http://127.0.0.1:8000` | Fallback backend URL |
| `MAX_UPLOAD_SIZE_MB`| `25` | Ceiling on uploaded image sizes |
| `MODEL_STORAGE_URL`| *(Optional)* | Remote URL / S3 bucket for downloading checkpoints |
| `YOLO_MODEL_PATH` | *(Optional)* | Path override for YOLO detection weights |
| `CLASSIFIER_CHECKPOINT_PATH` | *(Optional)* | Path override for deep classification weights |

---

## 4. Step-by-Step Deployment Procedures

### Procedure A: Deploy to Streamlit Community Cloud (Zero-Cost, 1-Click)

1. **Prerequisites**:
   - Push all commits to your GitHub branch (`skanda`).
2. **Setup on Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
   - Click **"New app"**.
   - Select:
     - **Repository**: `intelligent-vehicle-recognition`
     - **Branch**: `skanda`
     - **Main file path**: `app/streamlit_app.py`
   - Under **Advanced settings**:
     - Python version: `3.11`
     - Add environment variable: `DEVICE=cpu`
3. **Deploy**:
   - Click **"Deploy!"**.
   - Streamlit Cloud will parse `requirements.txt`, install dependencies, and launch the web dashboard at your generated public URL (`https://<app-name>.streamlit.app`).

---

### Procedure B: Deploy via Docker (Self-Hosted / Hugging Face Spaces)

The included `Dockerfile` utilizes CPU-only PyTorch to reduce image size by over 80% (from 9 GB down to ~1.4 GB).

1. **Build the Container**:
   ```bash
   docker build -t vehicle-recognition-system:latest .
   ```
2. **Run the Unified Streamlit App**:
   ```bash
   docker run -d -p 8501:8501 \
     -e PORT=8501 \
     -e DEVICE=cpu \
     --name vehicle_app \
     vehicle-recognition-system:latest
   ```
   Open `http://localhost:8501` in your browser.

3. **Run Multi-Service Architecture via Docker Compose**:
   ```bash
   docker compose up -d
   ```
   - Streamlit UI: `http://localhost:8501`
   - FastAPI REST API: `http://localhost:8000` (Swagger docs at `/docs`)

---

### Procedure C: Deploy to Render / Railway / PaaS

1. Connect your GitHub repository to [Render.com](https://render.com/).
2. Create a **New Web Service**:
   - **Environment**: `Python`
   - **Build Command**:
     ```bash
     pip install --upgrade pip
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
     grep -v -E "^(torch==|torchvision==|--extra-index-url)" requirements.txt > requirements-prod.txt
     pip install -r requirements-prod.txt
     ```
   - **Start Command**:
     ```bash
     streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
     ```
   - **Health Check Path**: `/_stcore/health`
3. Deploy service.

---

## 5. Startup Commands Quick Reference

| Operational Mode | Startup Command |
| :--- | :--- |
| **Standalone Streamlit UI** | `streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0` |
| **Headless FastAPI Server** | `uvicorn app.api:app --host 0.0.0.0 --port=8000 --workers=2` |
| **Production Docker Entry** | `docker run -p 8501:8501 vehicle-recognition-system:latest` |
| **Check Model Status** | `python scripts/download_models.py --check-only` |

---

## 6. Known Production Limitations & Workarounds

1. **Free Cloud Compute RAM Headroom**:
   - *Limitation*: Free-tier instances (e.g. Render free tier with 512 MB RAM) may trigger Out-Of-Memory (OOM) errors if full CUDA libraries are loaded.
   - *Mitigation*: Our `Dockerfile` and `render.yaml` strictly pre-install CPU-only PyTorch wheels (`https://download.pytorch.org/whl/cpu`), capping idle memory consumption at under 450 MB.
2. **Cold-Start Delays on Inactive Free Tiers**:
   - *Limitation*: Free hosting platforms suspend inactive containers after 15 minutes of inactivity.
   - *Mitigation*: The application features a lightweight healthcheck endpoint (`/_stcore/health` and `/health`) that wake the service within 15–20 seconds.
3. **Absence of Dedicated Phase 7 Weights**:
   - *Limitation*: Phase 7 early-fusion weights are scheduled for subsequent training iterations.
   - *Mitigation*: The production system gracefully utilizes late decision-level score fusion, combining classical SVM probability estimates with deep neural feature rankings.

---

## 7. Verification & Health Monitoring

To verify deployment health programmatically:
```bash
# Verify FastAPI REST service
curl -f http://<your-host>:8000/health

# Verify Streamlit server
curl -f http://<your-host>:8501/_stcore/health
```
Both endpoints return HTTP 200 OK when operational.

