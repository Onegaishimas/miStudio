#!/bin/bash
# Build script for miStudioExplain Docker image

set -e

IMAGE_NAME="mistudio/explain"
IMAGE_TAG="v1.0.0"
REGISTRY="localhost:32000"

echo "🐳 Building miStudioExplain Docker image..."

# Build image
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f deployment/docker/Dockerfile .

# Tag for local registry
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

# Push to local registry
echo "📤 Pushing to local registry..."
docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

echo "✅ Image built and pushed: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

