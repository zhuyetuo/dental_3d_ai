# 构建上下文是仓库根目录：docker build -f backend/Dockerfile .
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY ml /src/ml
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install /src/ml -r /tmp/requirements.txt

COPY backend/app /app/app

ENV DENTAL_DATA_DIR=/data/cases
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
