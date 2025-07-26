#!/bin/bash
# Monitoring script for miStudioExplain

echo "📊 miStudioExplain Service Monitoring"
echo "======================================"

# Check service status
echo "🔍 Service Status:"
kubectl get pods -n mistudio-services -l app=mistudio-explain

echo ""
echo "🔍 Service Endpoints:"
kubectl get services -n mistudio-services | grep mistudio-explain

echo ""
echo "🔍 GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "🔍 Recent Logs (last 20 lines):"
POD_NAME=$(kubectl get pods -n mistudio-services -l app=mistudio-explain -o jsonpath="{.items[0].metadata.name}" 2>/dev/null)
if [ ! -z "$POD_NAME" ]; then
    kubectl logs ${POD_NAME} -n mistudio-services --tail=20
else
    echo "No pods found"
fi

