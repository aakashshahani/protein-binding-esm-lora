# Serving image for the FastAPI inference endpoint (CPU).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PBSITE_MODEL_DIR=/models \
    PBSITE_EMB_DIR=/emb_cache

WORKDIR /app

# CPU torch first (smaller than the default CUDA wheels), then the rest.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install -e . --no-deps

EXPOSE 8000
# Mount a trained model at /models (model.pt + meta.json) to enable /predict.
CMD ["uvicorn", "pbsite.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
