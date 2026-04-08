FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Copy dependency specification first for layer caching
COPY pyproject.toml ./

# Install all dependencies explicitly (no uv, no bash process substitution)
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "pydantic>=2.0.0" \
    "uvicorn>=0.24.0" \
    "openai>=1.0.0" \
    "requests>=2.31.0" \
    "openenv-core[core]>=0.2.2"

# Copy all source code
COPY . .

# Install the project itself
RUN pip install --no-cache-dir -e .

# Set PYTHONPATH so imports resolve correctly from root
ENV PYTHONPATH="/app"

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
