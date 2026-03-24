FROM python:3.11-slim

LABEL maintainer="tclf"
LABEL description="Trade Classification with Python - Docker Image"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml version ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pandas \
    numpy \
    python-multipart

COPY app.py .
COPY entrypoint.py .
COPY examples/ ./examples/

EXPOSE 8000

ENTRYPOINT ["python", "entrypoint.py"]
CMD ["demo"]
