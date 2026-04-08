---
sidebar_position: 2
title: "System Requirements & Installation"
description: "Hardware requirements and software installation guide"
---

# System Requirements & Installation

## Hardware Requirements

| Tier | VRAM | Capability |
|------|------|-----------|
| **Minimum** | 8 GB | TinyLlama (1.1B), Phi-2, Phi-4-mini |
| **Recommended** | 16–24 GB (RTX 3090/4090) | Models up to 9B, wide SAEs (16k–131k features) |
| **Multi-GPU** | 2×24 GB+ | Dedicated inference + training partitions |

:::warning VRAM vs. System RAM
System RAM cannot compensate for low VRAM. Model weights and activations must reside on the GPU for acceptable speed. If a job exceeds VRAM, you'll get an "Out of Memory" (OOM) crash — the most common failure mode in local research.
:::

## Software Installation

miStudio is packaged as a Docker Compose project:

1. **Prerequisites:** Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. **Network Setup:** Add the domain to your hosts file:
   ```bash
   sudo bash -c 'echo "127.0.0.1  mistudio.mcslab.io" >> /etc/hosts'
   ```
3. **Start all services:**
   ```bash
   ./start-mistudio.sh
   ```

This launches six services:

| Service | Purpose |
|---------|---------|
| **PostgreSQL** | Stores all experiment metadata, labels, metrics, and settings |
| **Redis** | Message broker for the Celery task queue |
| **Celery Worker** | Performs GPU-intensive training, extraction, and labeling tasks |
| **Celery Beat** | Schedules periodic tasks (system monitoring, cleanup) |
| **FastAPI Backend** | API orchestrator with WebSocket support for real-time updates |
| **React Frontend** | Interactive dashboard at `http://mistudio.mcslab.io` |

:::info Why Docker?
A MechInterp environment requires exact versions of PyTorch, Transformers, spaCy, and CUDA kernels. Docker freezes these into a reproducible image — miStudio runs identically on a Jetson Orin and a datacenter server.
:::

## Kubernetes

Kubernetes is the recommended deployment method for shared lab environments and multi-user research clusters. The manifest at `k8s/mistudio-deployment.yaml` deploys the full miStudio stack into a dedicated `mistudio` namespace.

### Architecture

```
┌─────────────────────────────────────────────────┐
│  Namespace: mistudio                             │
│                                                  │
│  ┌──────────┐  ┌──────────┐                     │
│  │ postgres │  │  redis   │  (persistent storage)│
│  └──────────┘  └──────────┘                     │
│                                                  │
│  ┌──────────────────────────────────────┐        │
│  │  mistudio-backend Pod (GPU node)     │        │
│  │  ├── backend      (FastAPI :8000)    │        │
│  │  ├── celery-worker (GPU tasks)       │        │
│  │  └── celery-beat  (scheduled tasks) │        │
│  └──────────────────────────────────────┘        │
│                                                  │
│  ┌────────────────────┐                          │
│  │ mistudio-frontend  │  (React/Nginx :80)       │
│  └────────────────────┘                          │
│                                                  │
│  ┌────────────────────┐                          │
│  │ ollama-proxy       │  (ExternalName service)  │
│  └────────────────────┘                          │
└─────────────────────────────────────────────────┘
         │
    NGINX Ingress
    ├── /api  → mistudio-backend:8000
    ├── /ws   → mistudio-backend:8000 (WebSocket)
    ├── /ollama → ollama-proxy:11434
    └── /    → mistudio-frontend:80
```

The backend pod runs three containers sharing a single GPU and a shared `/data` volume — FastAPI handles API requests, Celery Worker runs training/extraction/labeling jobs, and Celery Beat fires scheduled tasks like system monitoring.

### Prerequisites

**Cluster requirements:**
- Kubernetes 1.25+ (MicroK8s, k3s, or full K8s)
- NGINX Ingress Controller (`ingressClassName: public`)
- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin) for GPU scheduling
- At least one node with an NVIDIA GPU and the NVIDIA Container Toolkit installed

**Local tooling:**
```bash
# Verify kubectl is connected to your cluster
kubectl cluster-info

# Verify NVIDIA device plugin is running
kubectl get pods -n kube-system | grep nvidia

# Verify GPU is schedulable
kubectl describe node <gpu-node> | grep nvidia.com/gpu
```

### Step 1: Prepare Host Storage

miStudio uses `hostPath` volumes for persistent data. Create the required directories on the GPU node before deploying:

```bash
# Run on the GPU node (or via ssh)
sudo mkdir -p /data/mistudio/postgres
sudo mkdir -p /data/mistudio/redis
sudo mkdir -p /data/mistudio/data
sudo chown -R 1000:1000 /data/mistudio
```

The `/data/mistudio/data` directory holds all miStudio working data — downloaded models, datasets, SAE weights, activations, and checkpoints. Size this volume accordingly (500 GB+ recommended for active research).

### Step 2: Configure the Manifest

Open `k8s/mistudio-deployment.yaml` and update the following before applying:

**Node selector** — pin all GPU pods to your GPU node:
```yaml
nodeSelector:
  kubernetes.io/hostname: your-gpu-node-name   # replace mcs-lnxgpu01
```

**Domain names** — update the ingress hosts and hostAlias to match your environment:
```yaml
# In hostAliases:
- ip: "192.168.x.x"           # Your GPU node IP
  hostnames:
    - "k8s-mistudio.yourdomain.com"

# In Ingress rules:
- host: k8s-mistudio.yourdomain.com
```

**Secrets** — change all default credentials before deploying to a shared environment:
```yaml
# PostgreSQL
- name: POSTGRES_PASSWORD
  value: "change-me"           # also update DATABASE_URL and DATABASE_URL_SYNC

# Backend secret key (used for AES-256-GCM encryption of API keys in settings)
- name: SECRET_KEY
  value: "change-me-to-a-long-random-string"
```

**Optional integrations:**
```yaml
# Ollama (for local LLM labeling) — comment out if not used
- name: OLLAMA_URL
  value: http://ollama-proxy:11434

# Neuronpedia local instance — comment out if not used
- name: NEURONPEDIA_LOCAL_URL
  value: http://k8s-neuron.yourdomain.com
- name: NEURONPEDIA_LOCAL_DB_URL
  value: postgresql://neuronpedia:password@host/neuronpedia
```

### Step 3: Deploy

```bash
# Apply the full manifest
kubectl apply -f k8s/mistudio-deployment.yaml

# Watch pods come up
kubectl get pods -n mistudio -w
```

Expected output once healthy:
```
NAME                                  READY   STATUS    RESTARTS   AGE
mistudio-backend-xxxxxxxxx-xxxxx      3/3     Running   0          60s
mistudio-frontend-xxxxxxxxx-xxxxx     1/1     Running   0          60s
postgres-xxxxxxxxx-xxxxx              1/1     Running   0          60s
redis-xxxxxxxxx-xxxxx                 1/1     Running   0          60s
```

:::info 3/3 on the backend pod
The backend pod runs three containers: `backend`, `celery-worker`, and `celery-beat`. All three must show Ready before the application is fully functional. Database migrations run automatically on first start via the entrypoint.
:::

### Step 4: Configure DNS

Add the ingress hostname to your DNS or local hosts file:

```bash
# On each client machine
echo "192.168.x.x  k8s-mistudio.yourdomain.com" | sudo tee -a /etc/hosts
```

Then access miStudio at `http://k8s-mistudio.yourdomain.com`.

### Verifying the Deployment

```bash
# Pod status
kubectl get pods -n mistudio

# Check backend logs (API container)
kubectl logs -n mistudio deployment/mistudio-backend -c backend --tail=50

# Check Celery worker logs
kubectl logs -n mistudio deployment/mistudio-backend -c celery-worker --tail=50

# Check Celery beat logs
kubectl logs -n mistudio deployment/mistudio-backend -c celery-beat --tail=50

# Verify GPU is allocated
kubectl exec -n mistudio deployment/mistudio-backend -c backend -- nvidia-smi

# Confirm API is responding
curl http://k8s-mistudio.yourdomain.com/api/v1/health
```

### Updating to New Images

miStudio publishes new images to DockerHub on every push to `main`. To update a running cluster:

```bash
# Pull latest images on the node and restart
kubectl rollout restart deployment/mistudio-backend -n mistudio
kubectl rollout restart deployment/mistudio-frontend -n mistudio

# Wait for rollout to complete
kubectl rollout status deployment/mistudio-backend -n mistudio --timeout=180s
kubectl rollout status deployment/mistudio-frontend -n mistudio --timeout=180s
```

:::info Recreate strategy
The backend uses `strategy: Recreate` — the old pod terminates completely before the new one starts. This prevents two pods from competing for the same GPU and the same data directory simultaneously.
:::

### Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_TYPE` | `api` | Container role: `api`, `celery-worker`, or `celery-beat` |
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async PostgreSQL connection string |
| `DATABASE_URL_SYNC` | `postgresql+psycopg2://...` | Sync PostgreSQL connection string (Alembic) |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `CELERY_BROKER_URL` | `redis://redis:6379/0` | Celery broker |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/0` | Celery result store |
| `SECRET_KEY` | *(change this)* | AES-256-GCM key for encrypting stored API keys |
| `DATA_DIR` | `/data` | Root for all miStudio data on the pod |
| `INTERNAL_API_URL` | `http://mistudio-backend:8000` | Internal URL for Celery→API callbacks |
| `OLLAMA_URL` | `http://ollama-proxy:11434` | Ollama endpoint for local LLM labeling |
| `NEURONPEDIA_LOCAL_URL` | *(optional)* | Local Neuronpedia instance for feature export |
| `NEURONPEDIA_LOCAL_DB_URL` | *(optional)* | Direct DB connection to local Neuronpedia |
