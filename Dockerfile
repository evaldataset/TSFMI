# TSFMI Reproducibility Container
# Usage:
#   docker build -t tsfmi:neurips2026 .
#   docker run --gpus all --rm -v $(pwd)/outputs:/app/outputs tsfmi:neurips2026 make test
#
# For reviewer reproduction:
#   docker run --gpus all --rm tsfmi:neurips2026 \
#       python -m pytest tests/ -q -m "not slow"

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        git build-essential \
        texlive-latex-extra latexmk \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt && \
    pip install pytest==9.0.2 pytest-cov==7.0.0 ruff==0.14.3

COPY . /app

# Smoke test during build
RUN pytest tests/ -q -m "not slow" --no-header -x || true

CMD ["bash"]
