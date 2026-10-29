# LifeMate CUDA runner — release candidate, A30 runtime UNVERIFIED
# Models must be pre-staged in model_bundle/ via scripts/package_models.py.
# Dataset/state are NEVER copied into the image; supplied at runtime via /data.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app/src \
    NVIDIA_VISIBLE_DEVICES=all
WORKDIR /app
COPY requirements_cuda.txt /app/requirements_cuda.txt
RUN python -m pip install --no-cache-dir -r requirements_cuda.txt
# Large pinned weights in a stable layer, BEFORE frequently edited Python files.
COPY model_bundle/ /models/
RUN test -f /models/llm/config.json && test -f /models/asr/config.json
COPY src/ /app/src/
RUN python -m compileall -q /app/src
CMD ["python","/app/src/run_cuda.py","--data-root","/data","--split","auto","--llm","/models/llm","--asr","/models/asr","--out","/result/submission.jsonl"]
