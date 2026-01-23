# Dockerfile for Cloud Batch GPU sweeps
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY experiments/04_cloud_sweep/ experiments/04_cloud_sweep/
COPY judges/ judges/

# Install dependencies
RUN uv sync --frozen --extra grading

# Install gsutil for GCS uploads
RUN pip install google-cloud-storage

# Set Python path
ENV PYTHONPATH=/app

# Default: show help
CMD ["uv", "run", "python", "experiments/04_cloud_sweep/sweep.py", "--help"]
