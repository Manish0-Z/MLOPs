FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY mlops_pipeline/requirements.txt .
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

COPY mlops_pipeline/ .

EXPOSE 8000

CMD ["uvicorn", "stages.model_deployment:app", "--host", "0.0.0.0", "--port", "8000"]
