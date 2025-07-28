# miStudioFind API Reference Documentation

## Overview

miStudioFind is the **feature discovery engine** in the miStudio AI Interpretability Platform. It analyzes trained Sparse Autoencoder (SAE) models from miStudioTrain to identify specific text patterns that activate each learned feature, transforming abstract mathematical vectors into human-understandable examples.

**Service Status**: ✅ **Production Ready**  
**Version**: 1.0.0  
**Base URL**: `http://<host>:8001`  
**Documentation**: `/docs` (OpenAPI/Swagger)

## Service Architecture

miStudioFind processes SAE training outputs through a sophisticated pipeline:

```
miStudioTrain Output → Feature Analysis → Pattern Categorization → Multi-Format Export
```

### Core Capabilities

- **Feature Analysis**: Identifies top activating text patterns for each feature
- **Quality Assessment**: Coherence scoring and interpretability evaluation
- **Pattern Categorization**: Advanced behavioral and semantic classification
- **Multi-Format Export**: JSON, CSV, XML, PyTorch, and ZIP outputs
- **Background Processing**: Asynchronous job handling with progress tracking

## Quick Start

### 1. Validate Input Files

Before starting analysis, validate your miStudioTrain outputs:

```bash
curl -X GET "http://localhost:8001/api/v1/validate/train_20250725_222505_9775"
```

### 2. Start Feature Analysis

```bash
curl -X POST "http://localhost:8001/api/v1/find/start" \
  -H "Content-Type: application/json" \
  -d '{
    "source_job_id": "train_20250725_222505_9775",
    "top_k": 20,
    "coherence_threshold": 0.7
  }'
```

### 3. Monitor Progress

```bash
curl -X GET "http://localhost:8001/api/v1/find/{job_id}/status"
```

### 4. Retrieve Results

```bash
curl -X GET "http://localhost:8001/api/v1/find/{job_id}/results"
```

## API Endpoints

### Core Service Endpoints

#### Service Information

```http
GET /
```

**Description**: Get service information and feature availability.

**Response Example**:
```json
{
  "service": "miStudioFind", 
  "status": "running",
  "version": "1.0.0",
  "description": "Feature Analysis Service for miStudio AI Interpretability Platform",
  "documentation": "/docs",
  "health_check": "/health",
  "features_available": {
    "basic_analysis": true,
    "file_persistence": true,
    "advanced_filtering": true
  }
}
```

#### Health Check

```http
GET /health
```

**Description**: Comprehensive health check with deep system validation.

**Response Example**:
```json
{
  "service": "miStudioFind",
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-07-26T15:13:21.123456",
  "system_health": {
    "processing_service": "available",
    "enhanced_persistence": "available", 
    "advanced_filtering": "available",
    "data_path_accessible": true,
    "free_disk_space_gb": 1024.5
  },
  "performance_metrics": {
    "last_job_processing_time": 1.1,
    "average_features_per_second": 465,
    "memory_usage_mb": 2048
  }
}
```

### File Validation

#### Validate Input Files

```http
GET /api/v1/validate/{job_id}
```

**Description**: Validate miStudioTrain output files for analysis readiness.

**Path Parameters**:
- `job_id` (string): Job ID from miStudioTrain

**Response Example**:
```json
{
  "job_id": "train_20250725_222505_9775",
  "validation_timestamp": "2025-07-26T15:13:21.123456",
  "files": {
    "sae_model": {
      "path": "/data/models/train_20250725_222505_9775_sae.pt",
      "exists": true,
      "readable": true,
      "size_mb": 10.5,
      "status": "valid"
    },
    "feature_activations": {
      "path": "/data/activations/train_20250725_222505_9775_activations.pt",
      "exists": true,
      "readable": true,
      "size_mb": 117.6,
      "status": "valid"
    },
    "metadata": {
      "path": "/data/activations/train_20250725_222505_9775_metadata.json",
      "exists": true,
      "readable": true,
      "size_mb": 0.035,
      "status": "valid"
    }
  },
  "summary": {
    "all_files_present": true,
    "all_files_readable": true,
    "reasonable_file_sizes": true,
    "total_size_mb": 128.14,
    "ready_for_analysis": true
  },
  "metadata_info": {
    "model_name": "microsoft/phi-4",
    "layer_number": 16,
    "feature_count": 512,
    "hidden_size": 5120,
    "ready_for_find": true,
    "training_loss": 0.0234
  },
  "next_steps": {
    "can_start_analysis": true,
    "recommendation": "Files look good - ready for feature analysis!"
  }
}
```

**Error Response (404)**:
```json
{
  "detail": "Training job train_invalid_id not found. Check job ID or ensure training completed successfully."
}
```

### Job Management

#### Start Analysis Job

```http
POST /api/v1/find/start
```

**Description**: Start a new feature analysis job.

**Request Body**:
```json
{
  "source_job_id": "train_20250725_222505_9775",
  "top_k": 20,
  "coherence_threshold": 0.7,
  "include_statistics": true
}
```

**Request Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_job_id` | string | Yes | - | Job ID from miStudioTrain |
| `top_k` | integer | No | 20 | Number of top activating texts per feature (1-100) |
| `coherence_threshold` | float | No | 0.7 | Minimum coherence score for quality assessment (0.0-1.0) |
| `include_statistics` | boolean | No | true | Whether to include detailed statistical analysis |

**Response Example (202 Accepted)**:
```json
{
  "job_id": "find_20250726_151321_d9675dce",
  "status": "queued", 
  "message": "Feature analysis started for train_20250725_222505_9775",
  "source_job_id": "train_20250725_222505_9775",
  "parameters": {
    "top_k": 20,
    "coherence_threshold": 0.7
  },
  "timestamp": "2025-07-26T15:13:21.123456",
  "next_steps": {
    "check_status": "/api/v1/find/find_20250726_151321_d9675dce/status",
    "get_results": "/api/v1/find/find_20250726_151321_d9675dce/results"
  }
}
```

**Error Responses**:
```json
// 400 Bad Request - Invalid parameters
{
  "detail": "top_k must be between 1 and 100"
}

// 503 Service Unavailable - Processing service not available
{
  "detail": "Processing service not available. Check core module imports."
}
```

#### Get Job Status

```http
GET /api/v1/find/{job_id}/status
```

**Description**: Get current status and progress of an analysis job.

**Path Parameters**:
- `job_id` (string): Analysis job identifier

**Response Example**:
```json
{
  "job_id": "find_20250726_151321_d9675dce",
  "status": "running",
  "source_job_id": "train_20250725_222505_9775",
  "progress": {
    "features_processed": 256,
    "total_features": 512,
    "current_feature": 257,
    "estimated_time_remaining": 0.6
  },
  "progress_percentage": 50.0,
  "start_time": "2025-07-26T15:13:21.123456",
  "timestamp": "2025-07-26T15:13:21.789012",
  "message": "Analyzing feature 257/512 (50.0% complete, ETA: 0.6s)",
  "performance_metrics": {
    "features_per_second": 426.7,
    "memory_usage_mb": 3072
  }
}
```

**Status Values**:
- `queued`: Job is waiting to start
- `running`: Job is currently executing  
- `completed`: Job finished successfully
- `failed`: Job encountered an error
- `cancelled`: Job was cancelled

**Error Response (404)**:
```json
{
  "detail": "Job find_invalid_id not found"
}
```

#### Get Job Results

```http
GET /api/v1/find/{job_id}/results
```

**Description**: Get complete results of a finished analysis job.

**Path Parameters**:
- `job_id` (string): Analysis job identifier

**Response Example**:
```json
{
  "job_id": "find_20250726_151321_d9675dce",
  "source_job_id": "train_20250725_222505_9775",
  "status": "completed",
  "processing_time_seconds": 1.1,
  "timestamp": "2025-07-26T15:13:22.234567",
  "summary": {
    "total_features_analyzed": 512,
    "quality_distribution": {
      "high": 0,
      "medium": 7, 
      "low": 505
    },
    "mean_coherence_score": 0.234,
    "high_quality_features": 0,
    "interpretable_features": 7
  },
  "results": [
    {
      "feature_id": 348,
      "coherence_score": 0.501,
      "quality_level": "medium",
      "pattern_keywords": ["json", "schema", "validation"],
      "pattern_category": "technical",
      "top_activations": [
        {
          "text": "JSON schema validation patterns for API...",
          "activation_strength": 0.89,
          "text_index": 142,
          "ranking": 1
        }
      ],
      "activation_statistics": {
        "mean": 0.15,
        "std": 0.08,
        "frequency": 0.023,
        "max_activation": 0.92
      },
      "behavioral_indicators": ["structured_data", "validation_logic"],
      "complexity_score": 0.67
    }
  ],
  "ready_for_explain_service": true,
  "output_files": {
    "json": "/data/results/find_20250726_151321_d9675dce_results.json",
    "csv": "/data/results/find_20250726_151321_d9675dce_results.csv",
    "xml": "/data/results/find_20250726_151321_d9675dce_results.xml",
    "pytorch": "/data/results/find_20250726_151321_d9675dce_tensors.pt",
    "summary": "/data/results/find_20250726_151321_d9675dce_summary.txt",
    "archive": "/data/results/find_20250726_151321_d9675dce_complete.zip"
  }
}
```

**Error Responses**:
```json
// 409 Conflict - Job not completed
{
  "detail": "Job find_20250726_151321_d9675dce is not completed yet. Current status: running"
}

// 404 Not Found - Job doesn't exist
{
  "detail": "Job find_invalid_id not found"
}
```

#### List All Jobs

```http
GET /api/v1/find/jobs
```

**Description**: List all analysis jobs (active and completed).

**Query Parameters** (optional):
- `status` (string): Filter by job status
- `limit` (integer): Maximum number of jobs to return
- `offset` (integer): Number of jobs to skip

**Response Example**:
```json
{
  "total_jobs": 3,
  "active_jobs": 0,
  "completed_jobs": 3,
  "jobs": [
    {
      "job_id": "find_20250726_151321_d9675dce",
      "source_job_id": "train_20250725_222505_9775", 
      "status": "completed",
      "features_analyzed": 512,
      "interpretable_features": 7,
      "start_time": "2025-07-26T15:13:21.123456",
      "completion_time": "2025-07-26T15:13:22.234567",
      "processing_time_seconds": 1.1
    }
  ],
  "pagination": {
    "limit": 50,
    "offset": 0,
    "has_more": false
  }
}
```

### Export and Download

#### Export Results

```http
GET /api/v1/find/{job_id}/export?format={format}
```

**Description**: Download analysis results in specified format.

**Path Parameters**:
- `job_id` (string): Analysis job identifier

**Query Parameters**:
- `format` (string): Export format - `json`, `csv`, `xml`, `pytorch`, `summary`, `all`

**Response**: File download or streaming response

**Format Descriptions**:

| Format | Description | Size (typical) | Use Case |
|--------|-------------|---------------|----------|
| `json` | Complete structured results | 21.7MB | API integration, detailed analysis |
| `csv` | Spreadsheet-friendly format | 260KB | Business analysis, Excel import |
| `xml` | Structured data exchange | 4.8MB | System integration, SOAP APIs |
| `pytorch` | Tensor format for ML | 3.4MB | Downstream ML processing |
| `summary` | Human-readable report | 1.2KB | Quick overview, reports |
| `all` | ZIP bundle of all formats | 4.4MB | Complete archive |

**Example Requests**:
```bash
# Download JSON results
curl -X GET "http://localhost:8001/api/v1/find/find_20250726_151321_d9675dce/export?format=json" \
  -o results.json

# Download complete ZIP bundle
curl -X GET "http://localhost:8001/api/v1/find/find_20250726_151321_d9675dce/export?format=all" \
  -o complete_results.zip

# Download CSV for spreadsheet analysis
curl -X GET "http://localhost:8001/api/v1/find/find_20250726_151321_d9675dce/export?format=csv" \
  -o analysis.csv
```

**Error Response (404)**:
```json
{
  "detail": "Job find_invalid_id not found or results not available"
}
```

### Advanced Filtering

#### Get Pattern Categories

```http
GET /api/v1/find/{job_id}/categories
```

**Description**: Get available pattern categories and quality tiers from analysis.

**Response Example**:
```json
{
  "job_id": "find_20250726_151321_d9675dce",
  "pattern_categories": {
    "technical": {
      "count": 245,
      "description": "Technical terminology, APIs, structured data",
      "examples": ["API endpoints", "JSON schemas", "system functions"]
    },
    "conversational": {
      "count": 189,
      "description": "Chat patterns, user interactions",
      "examples": ["greeting patterns", "question formats", "politeness markers"]
    },
    "behavioral": {
      "count": 67,
      "description": "AI decision-making patterns",
      "examples": ["tool usage", "capability assessment", "reasoning chains"]
    },
    "linguistic": {
      "count": 11,
      "description": "Language patterns and grammar",
      "examples": ["syntax patterns", "semantic structures"]
    }
  },
  "quality_tiers": {
    "excellent": {"count": 0, "threshold": ">0.8 coherence"},
    "good": {"count": 0, "threshold": "0.6-0.8 coherence"},
    "fair": {"count": 7, "threshold": "0.4-0.6 coherence"}, 
    "poor": {"count": 505, "threshold": "<0.4 coherence"}
  },
  "semantic_tags": [
    "json_schema", "api_validation", "chat_interaction", 
    "time_management", "tool_selection", "reasoning_pattern"
  ]
}
```

#### Filter Results

```http
GET /api/v1/find/{job_id}/results/filtered
```

**Description**: Get filtered analysis results based on categories and quality.

**Query Parameters**:
- `categories` (string): Comma-separated list of categories to include
- `min_coherence` (float): Minimum coherence score (0.0-1.0)
- `max_coherence` (float): Maximum coherence score (0.0-1.0)
- `quality_tiers` (string): Comma-separated quality tiers
- `semantic_tags` (string): Comma-separated semantic tags
- `limit` (integer): Maximum number of features to return

**Example Request**:
```bash
curl -X GET "http://localhost:8001/api/v1/find/find_20250726_151321_d9675dce/results/filtered?categories=behavioral,technical&min_coherence=0.4&limit=10"
```

**Response Example**:
```json
{
  "job_id": "find_20250726_151321_d9675dce",
  "filters_applied": {
    "categories": ["behavioral", "technical"],
    "min_coherence": 0.4,
    "limit": 10
  },
  "total_matches": 7,
  "filtered_results": [
    {
      "feature_id": 348,
      "coherence_score": 0.501,
      "quality_level": "fair",
      "pattern_category": "technical",
      "semantic_tags": ["json_schema", "api_validation"],
      "behavioral_indicators": ["structured_data", "validation_logic"],
      "complexity_score": 0.67,
      "top_activations": [
        {
          "text": "JSON schema validation patterns for API documentation...",
          "activation_strength": 0.89,
          "ranking": 1
        }
      ]
    }
  ],
  "summary": {
    "category_distribution": {
      "technical": 4,
      "behavioral": 3
    },
    "quality_distribution": {
      "fair": 7,
      "good": 0
    },
    "mean_coherence": 0.456
  }
}
```

### Archive Management

#### Get Archive Listing

```http
GET /api/v1/find/results/archive
```

**Description**: List all archived analysis results.

**Query Parameters**:
- `limit` (integer): Maximum number of items
- `offset` (integer): Number of items to skip
- `sort_by` (string): Sort criteria - `date`, `features`, `quality`
- `order` (string): Sort order - `asc`, `desc`

**Response Example**:
```json
{
  "total_archived_jobs": 15,
  "archive_entries": [
    {
      "job_id": "find_20250726_151321_d9675dce",
      "source_job_id": "train_20250725_222505_9775",
      "model_name": "microsoft/phi-4",
      "completion_date": "2025-07-26T15:13:22.234567",
      "features_analyzed": 512,
      "interpretable_features": 7,
      "mean_coherence": 0.234,
      "available_formats": ["json", "csv", "xml", "pytorch", "summary", "zip"],
      "archive_size_mb": 30.2
    }
  ],
  "pagination": {
    "limit": 50,
    "offset": 0,
    "has_more": false
  }
}
```

#### Get Archived Results

```http
GET /api/v1/find/archived/{job_id}
```

**Description**: Retrieve archived analysis results.

**Query Parameters**:
- `format` (string): Desired format - `json`, `csv`, `xml`, `pytorch`, `summary`

**Example**:
```bash
curl -X GET "http://localhost:8001/api/v1/find/archived/find_20250726_151321_d9675dce?format=csv" \
  -o archived_analysis.csv
```

## Data Models

### FindRequest

```typescript
interface FindRequest {
  source_job_id: string;           // Job ID from miStudioTrain
  top_k?: number;                  // Number of top activations (1-100, default: 20)
  coherence_threshold?: number;    // Minimum coherence score (0.0-1.0, default: 0.7)
  include_statistics?: boolean;    // Include detailed stats (default: true)
}
```

### FeatureAnalysis

```typescript
interface FeatureAnalysis {
  feature_id: number;              // Feature identifier
  coherence_score: number;         // Coherence score (0.0-1.0)
  quality_level: "excellent" | "good" | "fair" | "poor";
  pattern_category: string;        // Primary pattern category
  semantic_tags: string[];         // Semantic classification tags
  behavioral_indicators: string[]; // AI behavioral patterns
  complexity_score: number;        // Pattern complexity (0.0-1.0)
  top_activations: FeatureActivation[];
  activation_statistics: FeatureStatistics;
}
```

### FeatureActivation

```typescript
interface FeatureActivation {
  text: string;                    // Text snippet that activated feature
  activation_strength: number;    // Activation value (0.0-1.0)
  text_index: number;             // Index in original dataset
  ranking: number;                // Ranking among top activations
}
```

### FeatureStatistics

```typescript
interface FeatureStatistics {
  mean: number;                    // Mean activation across dataset
  std: number;                     // Standard deviation
  frequency: number;               // Activation frequency
  max_activation: number;          // Maximum activation value
}
```

### JobStatus

```typescript
interface JobStatus {
  job_id: string;                  // Unique job identifier
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  source_job_id: string;          // Source training job
  progress?: {
    features_processed: number;
    total_features: number;
    current_feature: number;
    estimated_time_remaining: number;
  };
  progress_percentage: number;     // Completion percentage (0-100)
  start_time: string;             // ISO timestamp
  message: string;                // Human-readable status message
}
```

## Error Handling

### HTTP Status Codes

| Code | Description | When It Occurs |
|------|-------------|----------------|
| 200 | OK | Successful request |
| 202 | Accepted | Job started successfully |
| 400 | Bad Request | Invalid parameters or malformed request |
| 404 | Not Found | Job or resource not found |
| 409 | Conflict | Resource in incompatible state |
| 500 | Internal Server Error | Server-side processing error |
| 503 | Service Unavailable | Service components not available |

### Error Response Format

All errors follow a consistent format:

```json
{
  "detail": "Human-readable error description",
  "error_code": "ERROR_TYPE",
  "timestamp": "2025-07-26T15:13:21.123456",
  "job_id": "find_20250726_151321_d9675dce",
  "suggestion": "Recommended next action"
}
```

### Common Error Scenarios

#### Invalid Job ID
```json
{
  "detail": "Training job train_invalid_id not found. Check job ID or ensure training completed successfully.",
  "error_code": "JOB_NOT_FOUND"
}
```

#### Job Not Ready
```json
{
  "detail": "Job find_20250726_151321_d9675dce is not completed yet. Current status: running",
  "error_code": "JOB_NOT_COMPLETED",
  "current_status": "running",
  "suggestion": "Wait for job completion or check status endpoint"
}
```

#### Service Unavailable
```json
{
  "detail": "Processing service not available. Check core module imports.",
  "error_code": "SERVICE_UNAVAILABLE",
  "suggestion": "Contact system administrator"
}
```

#### Invalid Parameters
```json
{
  "detail": "top_k must be between 1 and 100",
  "error_code": "INVALID_PARAMETER",
  "parameter": "top_k",
  "provided_value": 150,
  "valid_range": "1-100"
}
```

## Code Examples

### Python Client Example

```python
import requests
import time
import json

class MiStudioFindClient:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        
    def validate_files(self, source_job_id):
        """Validate miStudioTrain output files."""
        response = requests.get(f"{self.base_url}/api/v1/validate/{source_job_id}")
        response.raise_for_status()
        return response.json()
    
    def start_analysis(self, source_job_id, top_k=20, coherence_threshold=0.7):
        """Start feature analysis job."""
        data = {
            "source_job_id": source_job_id,
            "top_k": top_k,
            "coherence_threshold": coherence_threshold
        }
        response = requests.post(f"{self.base_url}/api/v1/find/start", json=data)
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, job_id, poll_interval=2):
        """Wait for job completion with progress monitoring."""
        while True:
            status = self.get_status(job_id)
            print(f"Status: {status['status']} - {status['message']}")
            
            if status['status'] in ['completed', 'failed', 'cancelled']:
                return status
            
            time.sleep(poll_interval)
    
    def get_status(self, job_id):
        """Get job status."""
        response = requests.get(f"{self.base_url}/api/v1/find/{job_id}/status")
        response.raise_for_status()
        return response.json()
    
    def get_results(self, job_id):
        """Get analysis results."""
        response = requests.get(f"{self.base_url}/api/v1/find/{job_id}/results")
        response.raise_for_status()
        return response.json()
    
    def download_export(self, job_id, format_type="json", output_file=None):
        """Download results in specified format."""
        response = requests.get(
            f"{self.base_url}/api/v1/find/{job_id}/export",
            params={"format": format_type},
            stream=True
        )
        response.raise_for_status()
        
        if output_file:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            return response.content

# Usage example
client = MiStudioFindClient()

# 1. Validate input files
validation = client.validate_files("train_20250725_222505_9775")
if validation['summary']['ready_for_analysis']:
    print("✅ Files ready for analysis")
    
    # 2. Start analysis
    job = client.start_analysis("train_20250725_222505_9775", top_k=30)
    job_id = job['job_id']
    print(f"Started job: {job_id}")
    
    # 3. Wait for completion
    final_status = client.wait_for_completion(job_id)
    
    if final_status['status'] == 'completed':
        # 4. Get results
        results = client.get_results(job_id)
        print(f"Analysis complete: {results['summary']['interpretable_features']} interpretable features")
        
        # 5. Download different formats
        client.download_export(job_id, "json", "results.json")
        client.download_export(job_id, "csv", "results.csv") 
        client.download_export(job_id, "all", "complete_results.zip")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');
const fs = require('fs');

class MiStudioFindClient {
    constructor(baseUrl = 'http://localhost:8001') {
        this.baseUrl = baseUrl;
        this.client = axios.create({ baseURL: baseUrl });
    }
    
    async validateFiles(sourceJobId) {
        const response = await this.client.get(`/api/v1/validate/${sourceJobId}`);
        return response.data;
    }
    
    async startAnalysis(sourceJobId, options = {}) {
        const data = {
            source_job_id: sourceJobId,
            top_k: options.topK || 20,
            coherence_threshold: options.coherenceThreshold || 0.7
        };
        const response = await this.client.post('/api/v1/find/start', data);
        return response.data;
    }
    
    async getStatus(jobId) {
        const response = await this.client.get(`/api/v1/find/${jobId}/status`);
        return response.data;
    }
    
    async waitForCompletion(jobId, pollInterval = 2000) {
        while (true) {
            const status = await this.getStatus(jobId);
            console.log(`Status: ${status.status} - ${status.message}`);
            
            if (['completed', 'failed', 'cancelled'].includes(status.status)) {
                return status;
            }
            
            await new Promise(resolve => setTimeout(resolve, pollInterval));
        }
    }
    
    async getResults(jobId) {
        const response = await this.client.get(`/api/v1/find/${jobId}/results`);
        return response.data;
    }
    
    async downloadExport(jobId, format = 'json', outputFile = null) {
        const response = await this.client.get(`/api/v1/find/${jobId}/export`, {
            params: { format },
            responseType: 'stream'
        });
        
        if (outputFile) {
            const writer = fs.createWriteStream(outputFile);
            response.data.pipe(writer);
            
            return new Promise((resolve, reject) => {
                writer.on('finish', resolve);
                writer.on('error', reject);
            });
        }
        
        return response.data;
    }
    
    async getFilteredResults(jobId, filters = {}) {
        const params = {};
        if (filters.categories) params.categories = filters.categories.join(',');
        if (filters.minCoherence) params.min_coherence = filters.minCoherence;
        if (filters.maxCoherence) params.max_coherence = filters.maxCoherence;
        if (filters.qualityTiers) params.quality_tiers = filters.qualityTiers.join(',');
        if (filters.limit) params.limit = filters.limit;
        
        const response = await this.client.get(`/api/v1/find/${jobId}/results/filtered`, { params });
        return response.data;
    }
}

// Usage example
async function analyzeFeatures() {
    const client = new MiStudioFindClient();
    
    try {
        // 1. Validate input files
        const validation = await client.validateFiles('train_20250725_222505_9775');
        
        if (!validation.summary.ready_for_analysis) {
            console.error('❌ Files not ready for analysis');
            return;
        }
        
        console.log('✅ Files validated successfully');
        
        // 2. Start analysis
        const job = await client.startAnalysis('train_20250725_222505_9775', {
            topK: 25,
            coherenceThreshold: 0.5
        });
        
        console.log(`🚀 Started analysis job: ${job.job_id}`);
        
        // 3. Wait for completion
        const finalStatus = await client.waitForCompletion(job.job_id);
        
        if (finalStatus.status === 'completed') {
            console.log('✅ Analysis completed successfully');
            
            // 4. Get results
            const results = await client.getResults(job.job_id);
            console.log(`📊 Found ${results.summary.interpretable_features} interpretable features`);
            
            // 5. Get high-quality features only
            const filtered = await client.getFilteredResults(job.job_id, {
                categories: ['behavioral', 'technical'],
                minCoherence: 0.4,
                limit: 10
            });
            
            console.log(`🎯 High-quality features: ${filtered.total_matches}`);
            
            // 6. Download results
            await client.downloadExport(job.job_id, 'json', 'analysis_results.json');
            await client.downloadExport(job.job_id, 'csv', 'analysis_results.csv');
            
            console.log('💾 Results downloaded successfully');
            
        } else {
            console.error(`❌ Analysis failed: ${finalStatus.message}`);
        }
        
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

// Run the analysis
analyzeFeatures();
```

### cURL Examples

#### Complete Workflow
```bash
#!/bin/bash

# Configuration
BASE_URL="http://localhost:8001"
SOURCE_JOB_ID="train_20250725_222505_9775"

echo "🔍 miStudioFind Analysis Workflow"
echo "================================="

# 1. Health check
echo "1. Checking service health..."
curl -s "$BASE_URL/health" | jq '.'

# 2. Validate input files
echo -e "\n2. Validating input files..."
VALIDATION=$(curl -s "$BASE_URL/api/v1/validate/$SOURCE_JOB_ID")
echo "$VALIDATION" | jq '.'

READY=$(echo "$VALIDATION" | jq -r '.summary.ready_for_analysis')
if [ "$READY" != "true" ]; then
    echo "❌ Files not ready for analysis"
    exit 1
fi

echo "✅ Files validated successfully"

# 3. Start analysis
echo -e "\n3. Starting analysis..."
JOB_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/find/start" \
    -H "Content-Type: application/json" \
    -d "{
        \"source_job_id\": \"$SOURCE_JOB_ID\",
        \"top_k\": 30,
        \"coherence_threshold\": 0.6
    }")

echo "$JOB_RESPONSE" | jq '.'
JOB_ID=$(echo "$JOB_RESPONSE" | jq -r '.job_id')

if [ "$JOB_ID" == "null" ]; then
    echo "❌ Failed to start analysis"
    exit 1
fi

echo "🚀 Started job: $JOB_ID"

# 4. Monitor progress
echo -e "\n4. Monitoring progress..."
while true; do
    STATUS_RESPONSE=$(curl -s "$BASE_URL/api/v1/find/$JOB_ID/status")
    STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')
    MESSAGE=$(echo "$STATUS_RESPONSE" | jq -r '.message')
    PROGRESS=$(echo "$STATUS_RESPONSE" | jq -r '.progress_percentage // 0')
    
    echo "Status: $STATUS - $MESSAGE ($PROGRESS%)"
    
    if [ "$STATUS" == "completed" ]; then
        echo "✅ Analysis completed successfully"
        break
    elif [ "$STATUS" == "failed" ]; then
        echo "❌ Analysis failed"
        exit 1
    fi
    
    sleep 2
done

# 5. Get results
echo -e "\n5. Retrieving results..."
RESULTS=$(curl -s "$BASE_URL/api/v1/find/$JOB_ID/results")
echo "$RESULTS" | jq '.summary'

INTERPRETABLE=$(echo "$RESULTS" | jq -r '.summary.interpretable_features')
echo "📊 Found $INTERPRETABLE interpretable features"

# 6. Download exports
echo -e "\n6. Downloading results..."
curl -s "$BASE_URL/api/v1/find/$JOB_ID/export?format=json" -o "results_$JOB_ID.json"
curl -s "$BASE_URL/api/v1/find/$JOB_ID/export?format=csv" -o "results_$JOB_ID.csv"
curl -s "$BASE_URL/api/v1/find/$JOB_ID/export?format=all" -o "complete_$JOB_ID.zip"

echo "💾 Results downloaded:"
echo "  - results_$JOB_ID.json (detailed results)"
echo "  - results_$JOB_ID.csv (spreadsheet format)"
echo "  - complete_$JOB_ID.zip (all formats)"

# 7. Get filtered high-quality features
echo -e "\n7. Getting high-quality features..."
FILTERED=$(curl -s "$BASE_URL/api/v1/find/$JOB_ID/results/filtered?categories=behavioral,technical&min_coherence=0.4&limit=5")
echo "$FILTERED" | jq '.summary'

echo -e "\n🎉 Analysis workflow completed successfully!"
```

#### Batch Processing Multiple Jobs
```bash
#!/bin/bash

# Batch process multiple training jobs
TRAINING_JOBS=("train_20250725_222505_9775" "train_20250725_223015_8821" "train_20250725_224525_7743")
BASE_URL="http://localhost:8001"

for SOURCE_JOB in "${TRAINING_JOBS[@]}"; do
    echo "Processing $SOURCE_JOB..."
    
    # Start analysis
    JOB_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/find/start" \
        -H "Content-Type: application/json" \
        -d "{\"source_job_id\": \"$SOURCE_JOB\", \"top_k\": 20}")
    
    JOB_ID=$(echo "$JOB_RESPONSE" | jq -r '.job_id')
    echo "Started: $JOB_ID"
    
    # Store job ID for later monitoring
    echo "$JOB_ID" >> batch_jobs.txt
done

echo "All jobs started. Monitor with: watch 'curl -s $BASE_URL/api/v1/find/jobs | jq \".jobs[]\"'"
```

## Best Practices

### 1. Input Validation
Always validate input files before starting analysis:

```python
def safe_analysis_start(client, source_job_id):
    """Safely start analysis with proper validation."""
    
    # 1. Validate files first
    validation = client.validate_files(source_job_id)
    
    if not validation['summary']['ready_for_analysis']:
        print("❌ Files not ready:")
        for file_type, info in validation['files'].items():
            if not info['exists']:
                print(f"  - Missing: {file_type}")
            elif not info['readable']:
                print(f"  - Unreadable: {file_type}")
        return None
    
    # 2. Check metadata compatibility
    metadata = validation.get('metadata_info', {})
    if not metadata.get('ready_for_find', False):
        print("⚠️ Warning: Training may not be complete")
    
    # 3. Start analysis with appropriate parameters
    feature_count = metadata.get('feature_count', 512)
    top_k = min(30, max(10, feature_count // 20))  # Adaptive top_k
    
    return client.start_analysis(source_job_id, top_k=top_k)
```

### 2. Progress Monitoring
Implement robust progress monitoring with error handling:

```python
async def monitor_with_retry(client, job_id, max_retries=3):
    """Monitor job with retry logic for transient failures."""
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            status = await client.get_status(job_id)
            
            if status['status'] in ['completed', 'failed', 'cancelled']:
                return status
            
            # Log progress
            progress = status.get('progress_percentage', 0)
            message = status.get('message', 'Processing...')
            print(f"[{job_id[:8]}...] {progress:.1f}% - {message}")
            
            await asyncio.sleep(2)
            retry_count = 0  # Reset on success
            
        except Exception as e:
            retry_count += 1
            print(f"⚠️ Status check failed (attempt {retry_count}): {e}")
            
            if retry_count >= max_retries:
                raise
            
            await asyncio.sleep(5)  # Longer wait on failure
```

### 3. Result Processing
Process results efficiently with filtering:

```python
def extract_insights(results):
    """Extract key insights from analysis results."""
    
    summary = results['summary']
    features = results['results']
    
    # Filter for interpretable features
    interpretable = [f for f in features if f['quality_level'] in ['excellent', 'good', 'fair']]
    
    # Group by category
    by_category = {}
    for feature in interpretable:
        category = feature.get('pattern_category', 'unknown')
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(feature)
    
    # Find top patterns
    top_features = sorted(interpretable, 
                         key=lambda x: x['coherence_score'], 
                         reverse=True)[:10]
    
    return {
        'total_interpretable': len(interpretable),
        'categories': {k: len(v) for k, v in by_category.items()},
        'top_features': [(f['feature_id'], f['coherence_score'], f['pattern_category']) 
                        for f in top_features],
        'behavioral_patterns': [f for f in interpretable 
                               if f.get('pattern_category') == 'behavioral']
    }
```

### 4. Export Management
Efficiently handle different export formats:

```python
def download_optimized_exports(client, job_id, output_dir="./results"):
    """Download exports in optimal order and formats."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    exports = {
        'summary': f"{output_dir}/{job_id}_summary.txt",      # Quick overview
        'csv': f"{output_dir}/{job_id}_analysis.csv",        # Spreadsheet analysis  
        'json': f"{output_dir}/{job_id}_complete.json",      # Full results
        'pytorch': f"{output_dir}/{job_id}_tensors.pt",      # ML processing
    }
    
    # Download in order of typical usage
    for format_type, filepath in exports.items():
        try:
            print(f"Downloading {format_type}...")
            client.download_export(job_id, format_type, filepath)
            print(f"✅ Saved: {filepath}")
        except Exception as e:
            print(f"⚠️ Failed to download {format_type}: {e}")
    
    # Download complete archive as backup
    try:
        client.download_export(job_id, 'all', f"{output_dir}/{job_id}_complete.zip")
        print("✅ Complete archive saved")
    except Exception as e:
        print(f"⚠️ Archive download failed: {e}")
```

### 5. Error Recovery
Implement comprehensive error handling:

```python
class AnalysisError(Exception):
    """Custom exception for analysis errors."""
    pass

def robust_analysis_workflow(client, source_job_id, retry_failed=True):
    """Complete analysis workflow with error recovery."""
    
    try:
        # Step 1: Validation
        validation = client.validate_files(source_job_id)
        if not validation['summary']['ready_for_analysis']:
            raise AnalysisError(f"Files not ready: {validation['summary']}")
        
        # Step 2: Start analysis
        job = client.start_analysis(source_job_id)
        job_id = job['job_id']
        print(f"🚀 Started: {job_id}")
        
        # Step 3: Monitor with timeout
        start_time = time.time()
        timeout = 600  # 10 minutes
        
        while time.time() - start_time < timeout:
            status = client.get_status(job_id)
            
            if status['status'] == 'completed':
                print("✅ Analysis completed")
                return client.get_results(job_id)
            
            elif status['status'] == 'failed':
                error_msg = status.get('message', 'Unknown error')
                
                if retry_failed and 'memory' in error_msg.lower():
                    print("🔄 Retrying with reduced parameters...")
                    # Retry with smaller top_k
                    return robust_analysis_workflow(
                        client, source_job_id, retry_failed=False
                    )
                else:
                    raise AnalysisError(f"Analysis failed: {error_msg}")
            
            time.sleep(2)
        
        raise AnalysisError(f"Analysis timed out after {timeout}s")
        
    except requests.RequestException as e:
        raise AnalysisError(f"API communication error: {e}")
    
    except Exception as e:
        raise AnalysisError(f"Unexpected error: {e}")
```

## Performance Considerations

### Optimal Parameters

#### Feature Count vs Top-K Selection
```python
def calculate_optimal_top_k(feature_count, analysis_depth="standard"):
    """Calculate optimal top_k based on feature count and analysis depth."""
    
    base_ratios = {
        "quick": 0.02,      # 2% for rapid analysis
        "standard": 0.04,   # 4% for balanced analysis  
        "detailed": 0.08,   # 8% for comprehensive analysis
        "exhaustive": 0.15  # 15% for research purposes
    }
    
    ratio = base_ratios.get(analysis_depth, 0.04)
    calculated = int(feature_count * ratio)
    
    # Apply constraints
    return max(5, min(100, calculated))

# Usage
feature_count = 512  # From validation
top_k = calculate_optimal_top_k(feature_count, "standard")  # Returns 20
```

#### Coherence Threshold Guidelines
- **0.8+**: Excellent interpretability (rare, typically <1% of features)
- **0.6-0.8**: Good interpretability (5-15% of features)  
- **0.4-0.6**: Fair interpretability (15-25% of features)
- **<0.4**: Poor interpretability (majority of features)

### Memory Management

```python
def estimate_memory_usage(feature_count, top_k, sequence_length=512):
    """Estimate memory requirements for analysis."""
    
    # Base memory for service
    base_mb = 1024
    
    # Feature activation storage
    activation_mb = (feature_count * sequence_length * 4) / (1024 * 1024)  # 4 bytes per float
    
    # Top-k text storage  
    text_mb = (feature_count * top_k * 200) / (1024 * 1024)  # ~200 bytes per text
    
    # Processing overhead
    overhead_mb = activation_mb * 0.5
    
    total_mb = base_mb + activation_mb + text_mb + overhead_mb
    
    return {
        'total_mb': round(total_mb, 1),
        'breakdown': {
            'base': base_mb,
            'activations': round(activation_mb, 1),
            'text_storage': round(text_mb, 1),
            'overhead': round(overhead_mb, 1)
        },
        'recommendation': 'sufficient' if total_mb < 8192 else 'increase_memory'
    }
```

### Batch Processing

```python
async def process_multiple_jobs(client, source_jobs, max_concurrent=3):
    """Process multiple analysis jobs with concurrency control."""
    
    import asyncio
    from asyncio import Semaphore
    
    semaphore = Semaphore(max_concurrent)
    
    async def process_single_job(source_job_id):
        async with semaphore:
            try:
                # Validate
                validation = await client.validate_files(source_job_id)
                if not validation['summary']['ready_for_analysis']:
                    return {'source_job_id': source_job_id, 'status': 'validation_failed'}
                
                # Start
                job = await client.start_analysis(source_job_id)
                job_id = job['job_id']
                
                # Wait
                final_status = await client.wait_for_completion(job_id)
                
                if final_status['status'] == 'completed':
                    results = await client.get_results(job_id)
                    return {
                        'source_job_id': source_job_id,
                        'job_id': job_id,
                        'status': 'completed',
                        'interpretable_features': results['summary']['interpretable_features']
                    }
                else:
                    return {
                        'source_job_id': source_job_id,
                        'job_id': job_id,
                        'status': 'failed',
                        'error': final_status.get('message')
                    }
                    
            except Exception as e:
                return {
                    'source_job_id': source_job_id,
                    'status': 'error',
                    'error': str(e)
                }
    
    # Process all jobs concurrently
    tasks = [process_single_job(job_id) for job_id in source_jobs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results
```

## Integration with miStudio Pipeline

### Upstream Integration (miStudioTrain)

```python
def seamless_train_to_find_workflow(train_client, find_client, training_config):
    """Complete workflow from training to feature analysis."""
    
    # 1. Start training
    train_job = train_client.start_training(training_config)
    train_job_id = train_job['job_id']
    
    print(f"🏋️ Training started: {train_job_id}")
    
    # 2. Monitor training
    train_status = train_client.wait_for_completion(train_job_id)
    
    if train_status['status'] != 'completed':
        raise Exception(f"Training failed: {train_status.get('message')}")
    
    train_result = train_client.get_results(train_job_id)
    
    if not train_result.get('ready_for_find_service'):
        raise Exception("Training result not ready for feature analysis")
    
    print(f"✅ Training completed: {train_result['feature_count']} features")
    
    # 3. Automatically start feature analysis
    find_job = find_client.start_analysis(
        source_job_id=train_job_id,
        top_k=calculate_optimal_top_k(train_result['feature_count'])
    )
    
    print(f"🔍 Feature analysis started: {find_job['job_id']}")
    
    # 4. Monitor analysis
    find_status = find_client.wait_for_completion(find_job['job_id'])
    
    if find_status['status'] == 'completed':
        find_results = find_client.get_results(find_job['job_id'])
        
        return {
            'train_job_id': train_job_id,
            'find_job_id': find_job['job_id'],
            'model_name': training_config['model_name'],
            'total_features': train_result['feature_count'],
            'interpretable_features': find_results['summary']['interpretable_features'],
            'ready_for_explain': find_results['ready_for_explain_service']
        }
    else:
        raise Exception(f"Feature analysis failed: {find_status.get('message')}")
```

### Downstream Integration (miStudioExplain)

```python
def prepare_for_explanation_service(find_results, focus_categories=None):
    """Prepare miStudioFind results for miStudioExplain input."""
    
    # Filter for explanation-ready features
    features = find_results['results']
    
    # Default to behavioral and technical patterns
    if focus_categories is None:
        focus_categories = ['behavioral', 'technical', 'conversational']
    
    explanation_candidates = []
    
    for feature in features:
        # Quality filter
        if feature['quality_level'] in ['excellent', 'good', 'fair']:
            # Category filter
            if feature.get('pattern_category') in focus_categories:
                # Coherence filter
                if feature['coherence_score'] >= 0.4:
                    explanation_candidates.append({
                        'feature_id': feature['feature_id'],
                        'coherence_score': feature['coherence_score'],
                        'pattern_category': feature['pattern_category'],
                        'top_texts': [act['text'] for act in feature['top_activations'][:5]],
                        'pattern_keywords': feature['pattern_keywords'],
                        'complexity_score': feature.get('complexity_score', 0.5)
                    })
    
    # Sort by coherence and complexity for prioritization
    explanation_candidates.sort(
        key=lambda x: (x['coherence_score'], x['complexity_score']), 
        reverse=True
    )
    
    return {
        'source_job_id': find_results['job_id'],
        'model_name': find_results.get('model_name', 'unknown'),
        'total_features_analyzed': find_results['summary']['total_features_analyzed'],
        'explanation_candidates': explanation_candidates[:20],  # Top 20 for explanation
        'categories_available': list(set(f['pattern_category'] for f in explanation_candidates)),
        'ready_for_explanation': len(explanation_candidates) > 0
    }
```

## Troubleshooting

### Common Issues

#### 1. Job Not Found Error
```
Error: Job train_invalid_id not found
```

**Causes**:
- Invalid job ID format
- Training job not completed
- Files moved or deleted

**Solutions**:
```bash
# Check available training jobs
curl "$BASE_URL/api/v1/train/jobs" | jq '.jobs[].job_id'

# Verify training completion
curl "$BASE_URL/api/v1/train/{job_id}/status"

# Check file system
ls -la /data/models/
ls -la /data/activations/
```

#### 2. Memory Errors
```
Error: Insufficient memory for analysis
```

**Solutions**:
- Reduce `top_k` parameter
- Process features in smaller batches
- Increase system memory
- Check for memory leaks

#### 3. Processing Timeouts
```
Error: Analysis timed out after 600s
```

**Causes**:
- Large feature count
- High `top_k` value
- System resource constraints

**Solutions**:
```python
# Reduce parameters
job = client.start_analysis(
    source_job_id, 
    top_k=10,  # Reduced from default 20
    coherence_threshold=0.8  # Higher threshold = fewer features
)
```

#### 4. File Access Errors
```
Error: Permission denied accessing /data/...
```

**Solutions**:
```bash
# Check permissions
sudo chown -R $USER:$USER /data/
sudo chmod -R 755 /data/

# Check disk space
df -h /data/
```

### Debugging Tools

#### Enable Debug Logging
```bash
# Set debug environment
export LOG_LEVEL=DEBUG

# Check detailed logs
docker logs mistudio-find-service
```

#### Health Diagnostics
```python
def diagnose_system_health(client):
    """Comprehensive system diagnostics."""
    
    print("🔍 miStudioFind System Diagnostics")
    print("=" * 40)
    
    # 1. Service health
    try:
        health = client.get_health()
        print(f"✅ Service: {health['status']}")
        print(f"📊 Version: {health['version']}")
        
        for component, status in health.get('system_health', {}).items():
            icon = "✅" if status == "available" else "❌"
            print(f"{icon} {component}: {status}")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # 2. Recent jobs
    try:
        jobs = client.list_jobs()
        active_jobs = sum(1 for job in jobs['jobs'] if job['status'] in ['queued', 'running'])
        completed_jobs = sum(1 for job in jobs['jobs'] if job['status'] == 'completed')
        failed_jobs = sum(1 for job in jobs['jobs'] if job['status'] == 'failed')
        
        print(f"\n📋 Job Statistics:")
        print(f"   Active: {active_jobs}")
        print(f"   Completed: {completed_jobs}")
        print(f"   Failed: {failed_jobs}")
        
    except Exception as e:
        print(f"❌ Job listing failed: {e}")
    
    # 3. System resources
    try:
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/data')
        
        print(f"\n💾 System Resources:")
        print(f"   Memory: {memory.percent}% used ({memory.available // (1024**3)}GB free)")
        print(f"   Disk: {disk.percent}% used ({disk.free // (1024**3)}GB free)")
        
    except ImportError:
        print("⚠️ psutil not available for resource monitoring")
    except Exception as e:
        print(f"❌ Resource check failed: {e}")
```

## Security Considerations

### API Security
- **Input Validation**: All parameters are validated server-side
- **Path Traversal Protection**: File paths are sanitized  
- **Resource Limits**: Request size and processing time limits enforced
- **Error Information**: Error messages don't leak sensitive information

### Data Security
- **File Permissions**: Proper access controls on data directories
- **Temporary Files**: Automatic cleanup of processing artifacts
- **Archive Security**: Compressed downloads are safe from zip bombs

### Deployment Security
```yaml
# Kubernetes security context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  capabilities:
    drop: ["ALL"]
```

## Conclusion

This API reference provides comprehensive documentation for integrating with the miStudioFind service. The service is production-ready with proven performance on real-world data (Phi-4 analysis), offering:

- **Fast Processing**: 512 features analyzed in 1.1 seconds
- **Multiple Export Formats**: JSON, CSV, XML, PyTorch, ZIP
- **Advanced Filtering**: Pattern categorization and quality assessment
- **Robust Error Handling**: Comprehensive error recovery and retry logic
- **Complete Integration**: Seamless pipeline from miStudioTrain to miStudioExplain

For additional support or advanced use cases, refer to the interactive API documentation at `/docs` when the service is running.

---

**Service Status**: ✅ Production Ready  
**Last Updated**: July 26, 2025  
**API Version**: 1.0.0