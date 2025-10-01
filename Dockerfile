# -------- Stage 1: Build & Install Dependencies --------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into a local folder
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download SentenceTransformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# -------- Stage 2: Final Lightweight Runtime --------
FROM python:3.11-slim

WORKDIR /app

# Set Hugging Face cache location
ENV HF_HOME=/app/.cache/huggingface

# Copy only installed dependencies and preloaded model from builder
COPY --from=builder /install /usr/local
COPY --from=builder /root/.cache/huggingface /app/.cache/huggingface

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
