FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml ./

RUN pip install --no-cache-dir .

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000 8001

CMD ["sh", "-c", "adk api_server --host 0.0.0.0 --port 8000 adk & adk web --host 0.0.0.0 --port 8001 adk"]