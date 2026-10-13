FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# 1. Python deps. torch comes from PyTorch's CPU-only index: the PyPI aarch64
#    wheel is a CUDA build that dlopens libcublas at import (worthless and
#    ~1.6GB heavier in a CPU-only container). The +cpu wheel's own deps are the
#    usual small pure-python packages.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple "torch==2.14.1+cpu" \
    && pip install -r requirements.txt \
    && python -c "import torch, pandas, numpy, einops, safetensors, huggingface_hub, tqdm; print('deps ok, torch', torch.__version__)"

# 2. Bake models into the image so runtime needs no Hugging Face access.
#    Runtime uses HF_HUB_OFFLINE=1 -> sub-second load from this cache.
#    Both tiers are baked; choose at runtime with KRONOS_MODEL.
RUN python -c "from huggingface_hub import snapshot_download; \
    [print(snapshot_download(r)) for r in \
    ['NeoQuasar/Kronos-Tokenizer-base', 'NeoQuasar/Kronos-small', \
     'NeoQuasar/Kronos-mini', 'NeoQuasar/Kronos-Tokenizer-2k']]"

# 3. App code (chmod: files come from an exFAT context with no useful modes)
COPY . .
RUN chmod -R a+rX /app

# 4. Runtime config. SQLite MUST live on a real filesystem (the exFAT SSD can
#    silently corrupt/lose SQLite writes; /data is a docker volume -> host fs).
ENV KRONOS_DB_PATH=/data/kronos.db \
    HF_HUB_OFFLINE=1 \
    KRONOS_BIND_HOST=0.0.0.0

RUN useradd -m appuser && mkdir -p /data && chown -R appuser:appuser /opt/hf-cache /data
USER appuser

VOLUME ["/data"]
EXPOSE 8787

CMD ["python", "sidecar/daemon.py"]
