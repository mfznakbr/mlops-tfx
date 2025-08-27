FROM python:3.9.15-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir tensorflow-cpu==2.10.1 fastapi uvicorn numpy==1.23.5 prometheus_fastapi_instrumentator

COPY app.py .
COPY fznabr-pipeline/serving_model/1756283062 ./model/

CMD ["python", "app.py"]
