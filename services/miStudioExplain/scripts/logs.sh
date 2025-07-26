#!/bin/bash
# Log viewing script for miStudioExplain

NAMESPACE="mistudio-services"
POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l app=mistudio-explain -o jsonpath="{.items[0].metadata.name}")

if [ -z "$POD_NAME" ]; then
    echo "❌ No miStudioExplain pods found in namespace ${NAMESPACE}"
    exit 1
fi

echo "📋 Following logs for pod: ${POD_NAME}"
kubectl logs -f ${POD_NAME} -n ${NAMESPACE}

