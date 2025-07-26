# Multi-stage build for miStudioTrain service
# Stage 1: Build dependencies
FROM python:3.11-slim as builder

# Set build arguments
ARG CUDA_VERSION=cu121
ARG PYTORCH_VERSION=2.5.1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==${PYTORCH_VERSION}+${CUDA_VERSION} torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Install other requirements
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 2: Production image
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r mistudio && useradd -r -g mistudio mistudio

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create data directories with proper permissions
RUN mkdir -p /data/{samples,models,activations,artifacts} && \
    chown -R mistudio:mistudio /data /app

# Environment variables
ENV PYTHONPATH=/app/src
ENV DATA_PATH=/data
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER mistudio

# Expose API port
EXPOSE 8000

# Default command
CMD ["python", "src/main.py"]