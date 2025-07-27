# miStudio API Documentation

## Overview

miStudio is an AI interpretability platform that provides services for training sparse autoencoders (SAEs), analyzing features, and generating explanations for AI model behaviors. The platform consists of multiple microservices that work together to provide comprehensive AI interpretability capabilities.

## Architecture

The miStudio platform follows a microservices architecture with the following main services:

- **miStudioTrain**: Sparse Autoencoder Training Service
- **miStudioFind**: Feature Analysis Service  
- **miStudioExplain**: Explanation Generation Service
- **miStudioScore**: Feature Scoring Service
- **miStudioCorrelate**: Real-time Correlation Service
- **miStudioMonitor**: Live Monitoring Service
- **miStudioSteer**: Model Steering Service

## Service Details

### miStudioTrain Service

**Description**: The core training service that trains sparse autoencoders to discover interpretable features in AI models.

**Base URL**: `http://<host>:8001`

**Version**: 1.2.0

#### Authentication

For private models, you may need to provide a HuggingFace authentication token:

```json
{
  "huggingface_token": "hf_your_token_here"
}
```

#### Core Endpoints

##### Health Check

```http
GET /health
```

**Description**: Enhanced health check with memory and GPU information.

**Response**:
```json
{
  "service": "miStudioTrain",
  "status": "healthy",
  "version": "1.2.0",
  "gpu_available": true,
  "gpu_count": 2,
  "memory_status": {
    "gpu_0": {"free_gb": 22.1, "total_gb": 24.0},
    "gpu_1": {"free_gb": 10.8, "total_gb": 12.0}
  }
}
```

##### GPU Status

```http
GET /gpu/status
```

**Description**: Get detailed GPU status and memory information.

**Response**:
```json
{
  "cuda_available": true,
  "device_count": 2,
  "current_device": 0,
  "gpus": {
    "0": {
      "name": "NVIDIA GeForce RTX 3090",
      "memory_total": 25757220864,
      "memory_allocated": 1048576,
      "memory_free": 25756172288,
      "free_gb": 24.0,
      "utilization": 0.04
    }
  }
}
```

##### Start Training Job

```http
POST /api/v1/train
```

**Description**: Start a new sparse autoencoder training job.

**Request Body**:
```json
{
  "model_name": "microsoft/Phi-4",
  "corpus_file": "customer_reviews.txt",
  "layer_number": 16,
  "huggingface_token": "hf_optional_token",
  "hidden_dim": 1024,
  "sparsity_coeff": 0.001,
  "learning_rate": 0.0001,
  "batch_size": 8,
  "max_epochs": 50,
  "min_loss": 0.01,
  "gpu_id": null,
  "max_sequence_length": 512
}
```

**Request Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_name` | string | Yes | - | HuggingFace model name |
| `corpus_file` | string | Yes | - | Input corpus file name in samples/ |
| `layer_number` | integer | No | 12 | Layer to extract activations from (0-48) |
| `huggingface_token` | string | No | null | HuggingFace authentication token |
| `hidden_dim` | integer | No | 1024 | SAE hidden dimension |
| `sparsity_coeff` | float | No | 0.001 | L1 sparsity coefficient |
| `learning_rate` | float | No | 0.0001 | Training learning rate |
| `batch_size` | integer | No | 16 | Training batch size (1-128) |
| `max_epochs` | integer | No | 50 | Maximum training epochs |
| `min_loss` | float | No | 0.01 | Early stopping loss threshold |
| `gpu_id` | integer | No | null | Specific GPU ID (auto-select if null) |
| `max_sequence_length` | integer | No | 512 | Maximum sequence length for tokenization |

**Response**:
```json
{
  "job_id": "train_20250725_224918_1982",
  "status": "queued",
  "message": "Training job started for microsoft/Phi-4",
  "model_name": "microsoft/Phi-4",
  "requires_token": false,
  "memory_check": "passed",
  "optimizations_applied": "Applied optimizations for microsoft/Phi-4: true"
}
```

##### Get Training Status

```http
GET /api/v1/train/{job_id}/status
```

**Description**: Get current training job status with progress information.

**Path Parameters**:
- `job_id`: Unique job identifier

**Response**:
```json
{
  "job_id": "train_20250725_224918_1982",
  "status": "running",
  "progress": 0.65,
  "current_epoch": 33,
  "current_loss": 0.0234,
  "estimated_time_remaining": 420,
  "message": "Training in progress - Epoch 33/50",
  "model_info": {
    "model_name": "microsoft/Phi-4",
    "actual_model_loaded": "microsoft/Phi-4",
    "architecture": "phi3",
    "total_layers": 40,
    "selected_layer": 16,
    "hidden_size": 5120,
    "vocab_size": 32000,
    "requires_token": false
  },
  "last_checkpoint": "/data/models/train_20250725_224918_1982_checkpoint_30.pt"
}
```

**Status Values**:
- `queued`: Job is waiting to start
- `running`: Job is currently executing
- `completed`: Job finished successfully
- `failed`: Job encountered an error
- `cancelled`: Job was cancelled by user

##### Get Training Result

```http
GET /api/v1/train/{job_id}/result
```

**Description**: Get final training results and generated artifacts.

**Path Parameters**:
- `job_id`: Unique job identifier

**Response**:
```json
{
  "job_id": "train_20250725_224918_1982",
  "status": "completed",
  "model_path": "/data/models/train_20250725_224918_1982_sae.pt",
  "activations_path": "/data/activations/train_20250725_224918_1982_activations.pt",
  "metadata_path": "/data/activations/train_20250725_224918_1982_metadata.json",
  "training_stats": {
    "final_loss": 0.0156,
    "total_epochs": 42,
    "training_time_seconds": 2847,
    "convergence_epoch": 38,
    "feature_statistics": {
      "total_features": 1024,
      "active_features": 876,
      "dead_features": 148,
      "mean_activation": 0.0023,
      "sparsity_achieved": 0.0019
    }
  },
  "feature_count": 1024,
  "model_info": {
    "model_name": "microsoft/Phi-4",
    "actual_model_loaded": "microsoft/Phi-4",
    "architecture": "phi3",
    "total_layers": 40,
    "selected_layer": 16,
    "hidden_size": 5120,
    "vocab_size": 32000,
    "requires_token": false
  },
  "ready_for_find_service": true,
  "checkpoints": [
    "/data/models/train_20250725_224918_1982_checkpoint_30.pt",
    "/data/models/train_20250725_224918_1982_checkpoint_40.pt"
  ]
}
```

##### List Jobs

```http
GET /api/v1/jobs
```

**Description**: List all training jobs with their current status.

**Response**:
```json
{
  "total_jobs": 3,
  "jobs": [
    {
      "job_id": "train_20250725_224918_1982",
      "status": "completed",
      "model_name": "microsoft/Phi-4",
      "created_at": "2025-07-25T22:49:18Z",
      "completed_at": "2025-07-25T23:36:45Z"
    }
  ]
}
```

##### Check Memory Requirements

```http
GET /api/v1/check-memory/{model_name}
```

**Description**: Check memory requirements and GPU compatibility for a specific model.

**Path Parameters**:
- `model_name`: HuggingFace model name (URL-encoded)

**Example**: `/api/v1/check-memory/microsoft%2FPhi-4`

**Response**:
```json
{
  "model_name": "microsoft/Phi-4",
  "memory_check": {
    "sufficient": true,
    "required_gb": 18.5,
    "available_gb": 24.0,
    "recommendations": {
      "min_gpu_memory_gb": 16,
      "recommended_gpu_memory_gb": 24,
      "is_large_model": true,
      "use_4bit_quantization": true,
      "use_gradient_checkpointing": true
    }
  },
  "optimization_suggestions": {
    "batch_size": 4,
    "max_sequence_length": 512,
    "enable_4bit": true,
    "enable_checkpointing": true
  },
  "gpu_compatibility": {
    "gpu_0": {
      "compatible": true,
      "memory_sufficient": true,
      "recommended": true
    },
    "gpu_1": {
      "compatible": true,
      "memory_sufficient": false,
      "recommended": false
    }
  }
}
```

##### Upload Corpus File

```http
POST /api/v1/upload
```

**Description**: Upload a text corpus file for training.

**Request**: Multipart form data
- `file`: Text file containing the training corpus

**Response**:
```json
{
  "message": "File uploaded successfully",
  "filename": "customer_reviews.txt",
  "size_bytes": 2048576,
  "lines_count": 10000,
  "path": "/data/samples/customer_reviews.txt"
}
```

##### Validate Model

```http
POST /api/v1/validate-model
```

**Description**: Validate model accessibility and requirements before training.

**Request Body**:
```json
{
  "model_name": "microsoft/Phi-4",
  "huggingface_token": "hf_optional_token"
}
```

**Response**:
```json
{
  "model_name": "microsoft/Phi-4",
  "valid": true,
  "accessible": true,
  "requires_token": false,
  "model_info": {
    "architecture": "phi3",
    "total_layers": 40,
    "hidden_size": 5120,
    "vocab_size": 32000
  },
  "memory_requirements": {
    "minimum_gb": 16,
    "recommended_gb": 24
  }
}
```

#### Error Responses

All endpoints return appropriate HTTP status codes and error messages:

**400 Bad Request**:
```json
{
  "detail": "Corpus file 'nonexistent.txt' not found. Please upload to /data/samples/"
}
```

**404 Not Found**:
```json
{
  "detail": "Job train_invalid_job_id not found"
}
```

**500 Internal Server Error**:
```json
{
  "detail": "CUDA out of memory. Required: 20.5GB, Available: 18.2GB"
}
```

### Data Models

#### TrainingRequest

```typescript
interface TrainingRequest {
  model_name: string;              // HuggingFace model identifier
  corpus_file: string;             // Input corpus filename
  layer_number?: number;           // Target layer (default: 12)
  huggingface_token?: string;      // Authentication token
  hidden_dim?: number;             // SAE dimension (default: 1024)
  sparsity_coeff?: number;         // Sparsity coefficient (default: 0.001)
  learning_rate?: number;          // Learning rate (default: 0.0001)
  batch_size?: number;             // Batch size (default: 16)
  max_epochs?: number;             // Max epochs (default: 50)
  min_loss?: number;               // Early stopping threshold (default: 0.01)
  gpu_id?: number;                 // Specific GPU ID
  max_sequence_length?: number;    // Max sequence length (default: 512)
}
```

#### ModelInfo

```typescript
interface ModelInfo {
  model_name: string;              // Original model name
  actual_model_loaded: string;     // Actually loaded model
  architecture: string;           // Model architecture type
  total_layers: number;            // Total layers in model
  selected_layer: number;          // Selected layer for analysis
  hidden_size: number;             // Hidden dimension size
  vocab_size: number;              // Vocabulary size
  requires_token: boolean;         // Whether auth token is required
}
```

#### TrainingStatus

```typescript
interface TrainingStatus {
  job_id: string;                  // Unique job identifier
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;                // Progress (0.0 to 1.0)
  current_epoch: number;           // Current training epoch
  current_loss: number;            // Current training loss
  estimated_time_remaining?: number; // Seconds remaining
  message: string;                 // Status message
  model_info?: ModelInfo;          // Model information
  last_checkpoint?: string;        // Latest checkpoint path
}
```

## Best Practices

### Model Selection

1. **Small Models** (< 1B parameters): Use default settings
2. **Medium Models** (1B-7B parameters): Consider reducing batch_size to 8-16
3. **Large Models** (7B+ parameters): Use batch_size 4-8, enable optimizations

### Memory Management

- Always check memory requirements before starting training
- Use the `/api/v1/check-memory/{model_name}` endpoint
- Monitor GPU status during training
- Consider using smaller batch sizes for large models

### File Management

- Upload corpus files to `/data/samples/` using the upload endpoint
- Ensure corpus files are plain text with reasonable line lengths
- Use UTF-8 encoding for text files

### Error Handling

- Implement retry logic for transient errors
- Check job status regularly during training
- Handle memory errors by reducing batch size or using smaller models

## Rate Limits

- Training jobs are queued and processed sequentially
- No explicit rate limits on API calls
- GPU memory is the primary constraint

## Support

For technical support and advanced configuration:
- Check the comprehensive health endpoints
- Monitor GPU status and memory usage
- Review job logs through the status endpoint
- Use memory checking endpoints before starting large jobs

## Related Services

This documentation covers the miStudioTrain service. For complete AI interpretability workflows, also see:

- **miStudioFind**: Feature analysis and discovery
- **miStudioExplain**: Natural language explanations
- **miStudioScore**: Feature importance scoring

## Change Log

**Version 1.2.0**:
- Added memory optimization for large models
- Enhanced GPU management and monitoring
- Improved error handling and recovery
- Added model validation endpoints
- Dynamic model loading capabilities