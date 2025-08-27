FROM python:3.9.15-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir tensorflow-cpu==2.10.1 fastapi uvicorn numpy==1.23.5 prometheus_fastapi_instrumentator

COPY app.py .
COPY op/pp/Trainer/model/20/Format-Serving/ ./model/

CMD ["python", "app.py"]
