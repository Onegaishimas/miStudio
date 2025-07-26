#!/bin/bash
# Deployment script for miStudioExplain

set -e

NAMESPACE="mistudio-services"

echo "🚀 Deploying miStudioExplain to Kubernetes..."

# Create namespace if it doesnt exist
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes manifests
echo "📋 Applying Kubernetes manifests..."
kubectl apply -f deployment/kubernetes/

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/mistudio-explain -n ${NAMESPACE}

# Get service information
echo "📊 Service information:"
kubectl get services -n ${NAMESPACE} | grep mistudio-explain

echo "✅ Deployment complete!"
echo "🌐 Service available at: http://$(kubectl get nodes -o jsonpath={.items[0].status.addresses[0].address}):30802"

