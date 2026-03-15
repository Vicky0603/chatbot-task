FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETZ_NO_CACHE=1

WORKDIR /app

# System deps commonly needed by sentence-transformers / audio / choma backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage layer cache
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Ensure frontend static served by FastAPI is readable
RUN chmod -R a+r /app/frontend || true

EXPOSE 8000

# Use non-reload server in container (remove --reload)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

