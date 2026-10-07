# API + loader image. CPU-only torch (the voice encoder runs on CPU for the demo).
# Large layers first so rebuilds after code edits stay fast; the 1.2 GB XLS-R weights are NOT baked
# in -- they download once into the hf-cache volume on first start.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/hf-cache
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-serving.txt

COPY engines ./engines
COPY serving ./serving
COPY scripts/calibrate_voice_thresholds.py ./scripts/
# Model weights (artifacts/) and datasets (data/) are bind-mounted by docker-compose.yml, not baked in:
# dropping Model D's weights into artifacts/voice/ takes effect on restart, with no rebuild.

EXPOSE 8000
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
