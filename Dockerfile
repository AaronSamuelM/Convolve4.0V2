FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "gunicorn app:app \
  -k uvicorn.workers.UvicornWorker \
  -w 1 \
  --threads 1 \
  --timeout 120 \
  --max-requests 100 \
  --max-requests-jitter 20 \
  -b 0.0.0.0:8000"]

