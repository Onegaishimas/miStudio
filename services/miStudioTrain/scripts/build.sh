#!/bin/bash
# miStudioTrain Service Build and Deployment Scripts

# build.sh - Build Docker image for miStudioTrain service
build_image() {
    echo "🔨 Building miStudioTrain Docker image..."
    
    # Set variables
    SERVICE_NAME="mistudio-train"
    VERSION="v1.1.0"
    REGISTRY="localhost:32000"
    IMAGE_NAME="${REGISTRY}/mistudio/train:${VERSION}"
    LATEST_TAG="${REGISTRY}/mistudio/train:latest"
    
    # Build multi-stage Docker image
    docker build \
        --build-arg CUDA_VERSION=cu121 \
        --build-arg PYTORCH_VERSION=2.5.1 \
        --tag ${IMAGE_NAME} \
        --tag ${LATEST_TAG} \
        --file Dockerfile \
        .
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully built ${IMAGE_NAME}"
        
        # Push to MicroK8s registry
        echo "📤 Pushing to MicroK8s registry..."
        docker push ${IMAGE_NAME}
        docker push ${LATEST_TAG}
        
        if [ $? -eq 0 ]; then
            echo "✅ Successfully pushed to registry"
        else
            echo "❌ Failed to push to registry"
            exit 1
        fi
    else
        echo "❌ Docker build failed"
        exit 1
    fi
}

# test_local.sh - Test service locally before containerization
test_local() {
    echo "🧪 Testing miStudioTrain service locally..."
    
    # Check GPU availability
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
    
    if [ $? -ne 0 ]; then
        echo "❌ GPU test failed"
        exit 1
    fi
    
    # Create test corpus if it doesn't exist
    mkdir -p data/samples
    if [ ! -f "data/samples/test_corpus.txt" ]; then
        echo "Creating test corpus..."
        cat > data/samples/test_corpus.txt << 'EOF'
The artificial intelligence research community has made remarkable progress in recent years.
Large language models demonstrate emergent capabilities across diverse tasks and domains.
Understanding the internal mechanisms of these models is crucial for AI safety and alignment.
Sparse autoencoders provide a promising approach for mechanistic interpretability research.
Feature extraction and analysis help researchers understand what neural networks learn.
Transparency in AI systems builds trust and enables better human-AI collaboration.
The goal of interpretability research is to make AI systems more controllable and predictable.
Feature visualization techniques reveal the concepts learned by different model components.
EOF
    fi
    
    # Test API locally
    echo "🚀 Starting test server..."
    DATA_PATH="./data" python src/main.py &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 10
    
    # Test health endpoint
    curl -f http://localhost:8000/health
    if [ $? -eq 0 ]; then
        echo "✅ Health check passed"
    else
        echo "❌ Health check failed"
        kill $SERVER_PID
        exit 1
    fi
    
    # Test training endpoint
    echo "📊 Testing training endpoint..."
    curl -X POST "http://localhost:8000/api/v1/train" \
        -H "Content-Type: application/json" \
        -d '{
            "corpus_file": "test_corpus.txt",
            "model_name": "EleutherAI/pythia-160m",
            "layer_number": 6,
            "hidden_dim": 128,
            "max_epochs": 3,
            "batch_size": 4
        }'
    
    if [ $? -eq 0 ]; then
        echo "✅ Training endpoint test passed"
    else
        echo "❌ Training endpoint test failed"
    fi
    
    # Cleanup
    kill $SERVER_PID
    echo "🧹 Local test completed"
}

# deploy.sh - Deploy to MicroK8s cluster
deploy_k8s() {
    echo "🚀 Deploying miStudioTrain to MicroK8s..."
    
    # Check if MicroK8s is running
    microk8s status --wait-ready
    if [ $? -ne 0 ]; then
        echo "❌ MicroK8s not ready"
        exit 1
    fi
    
    # Apply Kubernetes manifests
    echo "📝 Applying Kubernetes manifests..."
    microk8s kubectl apply -f k8s/
    
    if [ $? -eq 0 ]; then
        echo "✅ Kubernetes manifests applied successfully"
    else
        echo "❌ Failed to apply Kubernetes manifests"
        exit 1
    fi
    
    # Wait for deployment to be ready
    echo "⏳ Waiting for deployment to be ready..."
    microk8s kubectl wait --for=condition=available --timeout=300s deployment/mistudio-train-deployment -n mistudio-services
    
    if [ $? -eq 0 ]; then
        echo "✅ Deployment is ready"
        
        # Show deployment status
        echo "📊 Deployment status:"
        microk8s kubectl get pods -n mistudio-services -l app=mistudio-train
        microk8s kubectl get services -n mistudio-services -l app=mistudio-train
        
        # Get external access info
        NODE_PORT=$(microk8s kubectl get service mistudio-train-external -n mistudio-services -o jsonpath='{.spec.ports[0].nodePort}')
        NODE_IP=$(microk8s kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
        
        echo "🌐 Service accessible at: http://${NODE_IP}:${NODE_PORT}"
        echo "📚 API documentation: http://${NODE_IP}:${NODE_PORT}/docs"
        
    else
        echo "❌ Deployment failed to become ready"
        exit 1
    fi
}

# monitor.sh - Monitor deployment and logs
monitor_deployment() {
    echo "📊 Monitoring miStudioTrain deployment..."
    
    # Show resource usage
    echo "💾 Resource usage:"
    microk8s kubectl top pods -n mistudio-services -l app=mistudio-train
    
    # Show recent logs
    echo "📋 Recent logs:"
    microk8s kubectl logs -n mistudio-services -l app=mistudio-train --tail=50
    
    # Show events
    echo "📅 Recent events:"
    microk8s kubectl get events -n mistudio-services --sort-by='.lastTimestamp' | tail -10
    
    # GPU monitoring if available
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 GPU status:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    fi
}

# update.sh - Update deployment with new image
update_deployment() {
    echo "🔄 Updating miStudioTrain deployment..."
    
    local new_version=${1:-"v1.1.0"}
    local registry="localhost:32000"
    
    # Update deployment image
    microk8s kubectl set image deployment/mistudio-train-deployment \
        mistudio-train=${registry}/mistudio/train:${new_version} \
        -n mistudio-services
    
    # Wait for rollout to complete
    microk8s kubectl rollout status deployment/mistudio-train-deployment -n mistudio-services
    
    if [ $? -eq 0 ]; then
        echo "✅ Deployment updated successfully"
    else
        echo "❌ Deployment update failed"
        exit 1
    fi
}

# cleanup.sh - Clean up deployment
cleanup_deployment() {
    echo "🧹 Cleaning up miStudioTrain deployment..."
    
    # Delete Kubernetes resources
    microk8s kubectl delete -f k8s/ --ignore-not-found=true
    
    # Remove images from local registry
    docker rmi localhost:32000/mistudio/train:v1.1.0 --force 2>/dev/null || true
    docker rmi localhost:32000/mistudio/train:latest --force 2>/dev/null || true
    
    echo "✅ Cleanup completed"
}

# setup_dev_environment.sh - Setup development environment
setup_dev_environment() {
    echo "🛠️ Setting up miStudioTrain development environment..."
    
    # Create directory structure
    mkdir -p {src,k8s,scripts,tests,data/{samples,models,activations,artifacts}}
    
    # Create sample corpus for testing
    cat > data/samples/dev_corpus.txt << 'EOF'
Artificial intelligence systems process information through complex neural networks.
Machine learning models learn patterns from large datasets to make predictions.
Deep learning uses multiple layers to extract hierarchical features from data.
Natural language processing enables computers to understand and generate human language.
Computer vision allows machines to interpret and analyze visual information.
Reinforcement learning trains agents to make decisions through trial and error.
Transfer learning applies knowledge from one domain to solve problems in another.
Ensemble methods combine multiple models to improve prediction accuracy.
Feature engineering transforms raw data into meaningful inputs for algorithms.
Cross-validation techniques assess model performance and prevent overfitting.
EOF
    
    # Create test script
    cat > scripts/test_api.py << 'EOF'
#!/usr/bin/env python3
"""Test script for miStudioTrain API"""

import requests
import time
import json

def test_mistudio_train_api():
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Health check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # Test training endpoint
    print("\nStarting training job...")
    training_request = {
        "corpus_file": "dev_corpus.txt",
        "model_name": "EleutherAI/pythia-160m",
        "layer_number": 6,
        "hidden_dim": 256,
        "max_epochs": 3,
        "batch_size": 4
    }
    
    response = requests.post(f"{base_url}/api/v1/train", json=training_request)
    print(f"Training request: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    if response.status_code == 200:
        job_id = result["job_id"]
        
        # Monitor job progress
        print(f"\nMonitoring job {job_id}...")
        while True:
            response = requests.get(f"{base_url}/api/v1/train/{job_id}/status")
            status = response.json()
            print(f"Status: {status['status']} - Progress: {status['progress']:.2%} - {status['message']}")
            
            if status["status"] in ["completed", "failed"]:
                break
            
            time.sleep(5)
        
        if status["status"] == "completed":
            # Get final result
            response = requests.get(f"{base_url}/api/v1/train/{job_id}/result")
            result = response.json()
            print("\nTraining completed!")
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_mistudio_train_api()
EOF
    
    chmod +x scripts/test_api.py
    
    echo "✅ Development environment setup completed"
    echo "📁 Directory structure created"
    echo "📝 Sample corpus created: data/samples/dev_corpus.txt"
    echo "🧪 Test script created: scripts/test_api.py"
}

# Main script dispatcher
case "$1" in
    "build")
        build_image
        ;;
    "test")
        test_local
        ;;
    "deploy")
        deploy_k8s
        ;;
    "monitor")
        monitor_deployment
        ;;
    "update")
        update_deployment "$2"
        ;;
    "cleanup")
        cleanup_deployment
        ;;
    "setup")
        setup_dev_environment
        ;;
    "all")
        echo "🚀 Running complete build and deployment pipeline..."
        setup_dev_environment
        test_local
        build_image
        deploy_k8s
        monitor_deployment
        ;;
    *)
        echo "Usage: $0 {build|test|deploy|monitor|update|cleanup|setup|all}"
        echo ""
        echo "Commands:"
        echo "  build   - Build Docker image and push to registry"
        echo "  test    - Test service locally before containerization"
        echo "  deploy  - Deploy to MicroK8s cluster"
        echo "  monitor - Monitor deployment status and logs"
        echo "  update  - Update deployment with new version"
        echo "  cleanup - Remove deployment and clean up resources"
        echo "  setup   - Setup development environment"
        echo "  all     - Run complete pipeline (setup + test + build + deploy)"
        echo ""
        echo "Examples:"
        echo "  $0 setup    # Setup development environment"
        echo "  $0 test     # Test locally"
        echo "  $0 build    # Build and push image"
        echo "  $0 deploy   # Deploy to Kubernetes"
        echo "  $0 all      # Run everything"
        exit 1
        ;;
esac