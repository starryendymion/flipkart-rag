#!/bin/bash
set -e

echo "🚀 Building Flipkart Product RAG System"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check for NVIDIA Docker support (optional)
if command -v nvidia-docker &> /dev/null || docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA Docker support detected"
    GPU_SUPPORT=true
else
    echo "⚠️  NVIDIA Docker support not detected. Running in CPU mode."
    GPU_SUPPORT=false
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data output vector_db models

# Build Docker image
echo "🏗️  Building Docker image..."
docker-compose build

# Verify build
echo "🔍 Verifying build..."
if docker images | grep -q flipkart-product-rag; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker image build failed"
    exit 1
fi