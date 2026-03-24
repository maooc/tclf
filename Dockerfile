# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN pip install --no-cache-dir uv

# Copy pyproject.toml and uv.lock
COPY pyproject.toml uv.lock ./

# Copy source code
COPY src/ ./src/

# Install the package and dependencies
RUN uv pip install --system -e .

# Install FastAPI and uvicorn for the API
RUN uv pip install --system fastapi uvicorn python-multipart

# Copy app.py and example data
COPY app.py ./
COPY examples/ ./examples/

# Expose port for the API
EXPOSE 8000

# Default command: run the demo script
CMD ["python", "/app/examples/demo.py"]
