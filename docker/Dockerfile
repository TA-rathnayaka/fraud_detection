# Shared image for the scoring API and the streaming producer/consumer -- the entrypoint
# command is overridden per-service in docker-compose.yml. Built from the project root so
# `src/` (and, at runtime, `data/` and `models/` via volume mounts) are all in context.

FROM python:3.11-slim

WORKDIR /app

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models
ENV DATA_DIR=/app/data

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
