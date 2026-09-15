FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/ml/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.14.0 torchvision==0.29.0 \
    && python -m pip install --no-cache-dir -r /app/requirements.txt
COPY ml/src /app/ml/src
COPY contracts/schemas /app/contracts/schemas
COPY ml/weights/current/best.pt /app/ml/weights/current/best.pt
COPY model-selection.json /app/model-selection.json
COPY start.sh /app/start.sh
RUN python -c "from holospex_ml.current_model import CHECKPOINT_PATH, verify_checkpoint; verify_checkpoint(CHECKPOINT_PATH)" \
    && chmod 555 /app/start.sh \
    && python -m pip freeze > /app/installed-packages.txt
USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["/app/start.sh"]
