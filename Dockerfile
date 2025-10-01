# -------- Stage 1: Build & Install Dependencies --------
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload SentenceTransformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# -------- Stage 2: Final Lightweight Runtime --------
FROM python:3.11-slim

WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface

# Copy only site-packages and binaries from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /root/.cache/huggingface /app/.cache/huggingface

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
