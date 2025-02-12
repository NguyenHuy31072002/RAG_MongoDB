# Base image with CUDA support
FROM nvidia/cuda:11.1.1-runtime-ubuntu18.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies for GPU support
RUN pip3 install --no-cache-dir \
    torch==2.2.0+cu121 \
    transformers==4.37.2 \
    sentence-transformers==2.5.1

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 8000

# Run with multiple workers for load balancing
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]