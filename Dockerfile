# Stage 1: Builder — compile wheels for all dependencies
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt \
    && apt-get purge -y --auto-remove gcc g++

# Stage 2: Runtime — lean image with pre-built wheels
FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

COPY . .

# Default: FastAPI API
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
