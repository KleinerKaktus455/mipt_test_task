FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    patchelf \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    python - <<'PY'
import glob
import os
import site
import subprocess

paths = []
for p in site.getsitepackages():
    paths.extend(glob.glob(os.path.join(p, "onnxruntime", "capi", "onnxruntime_pybind11_state*.so")))

for so_path in paths:
    subprocess.run(["patchelf", "--clear-execstack", so_path], check=False)
PY

COPY app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

