# =============================================================================
# Production Dockerfile: Intelligent Fine-Grained Vehicle Recognition System
# Optimized for lightweight cloud CPU instances (Streamlit Cloud / HF / Render)
# =============================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8501 \
    DEVICE=cpu

# Install system dependencies required by OpenCV and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user for security isolation
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/models /app/data/processed && \
    chown -R appuser:appuser /app

WORKDIR /app

# Step 1: Pre-install CPU-only PyTorch to avoid heavy CUDA bloat (reduces image from 9GB to ~1.4GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining application dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN grep -v -E "^(torch==|torchvision==|--extra-index-url)" requirements.txt > requirements-prod.txt && \
    pip install --no-cache-dir -r requirements-prod.txt

# Step 3: Copy application source code and configurations
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser data/processed/ ./data/processed/
COPY --chown=appuser:appuser data/metadata/ ./data/metadata/
COPY --chown=appuser:appuser data/splits/split_summary.json ./data/splits/split_summary.json
COPY --chown=appuser:appuser scripts/download_models.py ./scripts/download_models.py
COPY --chown=appuser:appuser .streamlit/ ./.streamlit/

# Switch to non-root user
USER appuser

# Health check to ensure web server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

EXPOSE 8501 8000

# Default entrypoint runs the interactive Streamlit dashboard binding to dynamic platform $PORT
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]

