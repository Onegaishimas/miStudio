# miStudioExplain API Reference Documentation

## Overview

miStudioExplain is the **AI explanation engine** in the miStudio AI Interpretability Platform. It transforms abstract feature mappings from miStudioFind into human-readable, actionable descriptions of AI behavioral patterns. The service generates clear, understandable explanations that reveal what the AI model has learned and how it makes decisions.

**Service Status**: 📋 **Specification Complete - Ready for Development**  
**Version**: 1.0.0  
**Base URL**: `http://<host>:8003`  
**Documentation**: `/docs` (OpenAPI/Swagger)

## Service Architecture

miStudioExplain processes feature analysis results through a sophisticated AI-powered pipeline:

```
miStudioFind Output → Feature Prioritization → Context Building → LLM Generation → Quality Validation → Structured Explanations
```

### Core Capabilities

- **Feature Analysis**: Processes miStudioFind JSON outputs with 512+ features
- **Intelligent Prioritization**: Selects most important features for explanation
- **Local LLM Integration**: Uses Ollama with Llama 3.1, CodeLlama, and Phi-4 models
- **Quality Validation**: Automated explanation accuracy and coherence assessment
- **Multi-Format Output**: JSON, HTML, PDF, CSV, and TXT exports

## Quick Start

### 1. Health Check

Verify service status and LLM availability:

```bash
curl -X GET "http://localhost:8003/health"
```

### 2. Generate Explanation

Submit feature analysis for explanation:

```bash
curl -X POST "http://localhost:8003/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "explain_001",
    "analysis_type": "complex_behavioral",
    "complexity": "medium",
    "input_data": {
      "find_job_id": "find_20250726_151321_d9675dce",
      "feature_analysis": {
        "feature_348": {
          "coherence_score": 0.501,
          "pattern_category": "technical",
          "pattern_keywords": ["json", "schema", "validation"]
        }
      },
      "summary_report": "Analysis of 512 features from Phi-4 model"
    }
  }'
```

### 3. Monitor Progress

Check explanation generation status:

```bash
curl -X GET "http://localhost:8003/jobs/{job_id}/status"
```

### 4. Retrieve Results

Download completed explanations:

```bash
curl -X GET "http://localhost:8003/jobs/{job_id}/results"
```

## API Endpoints

### Core Service Endpoints

#### Service Information

```http
GET /
```

**Description**: Get service information and component status.

**Response Example**:
```json
{
  "service": "miStudioExplain",
  "status": "running",
  "version": "1.0.0",
  "description": "Generates explanations for complex data patterns",
  "components": {
    "ollama_manager": "available",
    "input_manager": "ready",
    "feature_prioritizer": "ready",
    "context_builder": "ready",
    "explanation_generator": "ready",
    "quality_validator": "ready",
    "result_manager": "ready"
  },
  "llm_status": {
    "endpoint": "http://ollama.mcslab.io",
    "available_models": ["llama3.1:8b", "llama3.1:70b", "codellama:13b"],
    "health_check_passed": true
  }
}
```

#### Health Check

```http
GET /health
```

**Description**: Comprehensive health check including LLM connectivity.

**Response Example**:
```json
{
  "service": "miStudioExplain",
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-07-26T15:30:45.123456",
  "system_health": {
    "ollama_connection": "healthy",
    "model_availability": "ready",
    "gpu_resources": "available",
    "storage_access": "ready"
  },
  "llm_diagnostics": {
    "endpoint_reachable": true,
    "models_loaded": 3,
    "gpu_memory_available": "18.2GB",
    "last_generation_time": "0.8s"
  },
  "performance_metrics": {
    "explanations_generated": 45,
    "average_generation_time": "12.3s",
    "success_rate": 0.96
  }
}
```

### Explanation Generation

#### Generate Explanation

```http
POST /explain
```

**Description**: Generate natural language explanations for AI features.

**Request Body**:
```json
{
  "request_id": "explain_20250726_153045_abc123",
  "analysis_type": "complex_behavioral",
  "complexity": "medium",
  "model": "llama3.1:8b",
  "input_data": {
    "find_job_id": "find_20250726_151321_d9675dce",
    "feature_analysis": {
      "feature_348": {
        "coherence_score": 0.501,
        "quality_level": "medium",
        "pattern_category": "technical",
        "pattern_keywords": ["json", "schema", "validation"],
        "top_activations": [
          {
            "text": "JSON schema validation patterns for API documentation...",
            "activation_strength": 0.89,
            "context": "API documentation"
          }
        ],
        "activation_statistics": {
          "mean": 0.15,
          "std": 0.08,
          "frequency": 0.023
        }
      },
      "feature_410": {
        "coherence_score": 0.488,
        "quality_level": "medium",
        "pattern_category": "behavioral",
        "pattern_keywords": ["time", "interval", "scheduling"]
      }
    },
    "summary_report": "Analysis of 512 features from microsoft/phi-4 model showing technical and behavioral patterns"
  }
}
```

**Request Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `request_id` | string | Yes | - | Unique identifier for explanation request |
| `analysis_type` | string | No | "complex_behavioral" | Type of analysis ("technical_patterns", "complex_behavioral", "default") |
| `complexity` | string | No | "medium" | Explanation complexity ("low", "medium", "high") |
| `model` | string | No | auto-select | Specific LLM model to use |
| `input_data` | object | Yes | - | Feature analysis data or raw text |

**Input Data Types**:

1. **FindResultInput** (miStudioFind output):
```json
{
  "find_job_id": "find_20250726_151321_d9675dce",
  "feature_analysis": { /* feature data */ },
  "summary_report": "Text summary from analysis"
}
```

2. **RawTextInput** (direct text):
```json
{
  "text_corpus": "Text to be explained (minimum 100 characters)",
  "source_description": "Description of text source"
}
```

**Response Example (202 Accepted)**:
```json
{
  "status": "success",
  "request_id": "explain_20250726_153045_abc123",
  "validation_passed": true,
  "result_location": "/data/output/explain_20250726_153045_abc123_explanation.json",
  "explanation": {
    "success": true,
    "explanation_text": "This feature detects JSON schema validation patterns in API documentation and code examples. It activates when the AI encounters structured data validation rules, indicating the model has learned to recognize formal data specification patterns. This is significant for understanding how the AI processes technical documentation and API-related content.",
    "model_used": "llama3.1:8b",
    "token_count": 89,
    "validation_failures": []
  },
  "processing_info": {
    "features_prioritized": ["json", "schema", "validation"],
    "generation_time_seconds": 8.7,
    "quality_score": 0.89
  }
}
```

**Alternative Response Formats**:

**Streaming Response** (for long explanations):
```http
POST /explain?stream=true
```

**Batch Processing** (multiple features):
```json
{
  "request_id": "batch_001",
  "batch_mode": true,
  "max_features": 20,
  "input_data": { /* full feature set */ }
}
```

**Error Responses**:
```json
// 400 Bad Request - Invalid input
{
  "detail": "Invalid request payload: field 'request_id' is required",
  "error_code": "VALIDATION_ERROR",
  "timestamp": "2025-07-26T15:30:45.123456"
}

// 500 Internal Server Error - LLM failure
{
  "detail": "LLM failed: Model llama3.1:8b not available",
  "error_code": "LLM_UNAVAILABLE",
  "suggestion": "Try with a different model or check Ollama service"
}

// 503 Service Unavailable - Ollama not accessible
{
  "detail": "Ollama service not available",
  "error_code": "SERVICE_UNAVAILABLE",
  "ollama_endpoint": "http://ollama.mcslab.io",
  "suggestion": "Check Ollama service status and connectivity"
}
```

### Job Management

#### Get Job Status

```http
GET /jobs/{job_id}/status
```

**Description**: Monitor explanation generation progress.

**Path Parameters**:
- `job_id` (string): Explanation job identifier

**Response Example**:
```json
{
  "job_id": "explain_20250726_153045_abc123",
  "status": "processing",
  "request_id": "explain_20250726_153045_abc123",
  "progress": {
    "current_step": "llm_generation",
    "steps_completed": 3,
    "total_steps": 6,
    "progress_percentage": 50.0,
    "estimated_time_remaining": "8.2s"
  },
  "processing_details": {
    "features_to_process": 12,
    "features_completed": 6,
    "current_feature": "feature_410",
    "model_used": "llama3.1:8b",
    "quality_checks_passed": 5
  },
  "performance_metrics": {
    "average_time_per_feature": "1.4s",
    "tokens_generated": 1245,
    "validation_success_rate": 0.83
  },
  "timestamp": "2025-07-26T15:30:45.123456"
}
```

**Status Values**:
- `queued`: Job waiting to start
- `processing`: Actively generating explanations
- `validating`: Quality validation in progress
- `completed`: Successfully finished
- `failed`: Encountered an error
- `cancelled`: Job was cancelled

#### Get Job Results

```http
GET /jobs/{job_id}/results
```

**Description**: Retrieve completed explanation results.

**Query Parameters**:
- `format` (string): Response format - `json`, `html`, `summary`
- `include_metadata` (boolean): Include processing metadata

**Response Example**:
```json
{
  "job_id": "explain_20250726_153045_abc123",
  "request_id": "explain_20250726_153045_abc123",
  "status": "completed",
  "completion_time": "2025-07-26T15:31:18.456789",
  "processing_summary": {
    "total_features_processed": 12,
    "high_quality_explanations": 8,
    "medium_quality_explanations": 3,
    "failed_explanations": 1,
    "average_explanation_quality": 0.87,
    "total_processing_time": "24.3 minutes"
  },
  "explanations": [
    {
      "feature_id": 348,
      "original_coherence": 0.501,
      "explanation": "This feature detects JSON schema validation patterns in API documentation and code examples. It activates when the AI encounters structured data validation rules, indicating the model has learned to recognize formal data specification patterns. This is significant for understanding how the AI processes technical documentation and API-related content.",
      "explanation_quality": 0.89,
      "explanation_confidence": 0.92,
      "pattern_verification": "confirmed",
      "safety_assessment": "neutral",
      "business_relevance": "high",
      "model_used": "llama3.1:8b",
      "generation_time": "8.7s",
      "token_count": 89
    },
    {
      "feature_id": 410,
      "original_coherence": 0.488,
      "explanation": "This feature identifies time interval specifications and scheduling patterns in text. It responds to temporal expressions, calendar references, and time-based planning language, suggesting the model has developed an understanding of temporal reasoning and time management concepts.",
      "explanation_quality": 0.84,
      "explanation_confidence": 0.88,
      "pattern_verification": "confirmed",
      "safety_assessment": "neutral",
      "business_relevance": "medium"
    }
  ],
  "summary_insights": {
    "high_quality_explanations": 8,
    "safety_concerns_identified": 0,
    "business_critical_features": 3,
    "technical_pattern_count": 7,
    "behavioral_pattern_count": 5,
    "key_themes": ["API processing", "temporal reasoning", "data validation", "structured content"]
  },
  "output_files": {
    "json": "/data/output/explain_20250726_153045_abc123_explanations.json",
    "html": "/data/output/explain_20250726_153045_abc123_report.html",
    "pdf": "/data/output/explain_20250726_153045_abc123_report.pdf",
    "csv": "/data/output/explain_20250726_153045_abc123_data.csv",
    "summary": "/data/output/explain_20250726_153045_abc123_summary.txt"
  }
}
```

#### List Jobs

```http
GET /jobs
```

**Description**: List all explanation jobs.

**Query Parameters**:
- `status` (string): Filter by job status
- `limit` (integer): Maximum jobs to return (default: 50)
- `offset` (integer): Number of jobs to skip
- `sort_by` (string): Sort criteria - `created`, `completed`, `quality`

**Response Example**:
```json
{
  "total_jobs": 15,
  "active_jobs": 2,
  "completed_jobs": 12,
  "failed_jobs": 1,
  "jobs": [
    {
      "job_id": "explain_20250726_153045_abc123",
      "request_id": "explain_20250726_153045_abc123",
      "status": "completed",
      "analysis_type": "complex_behavioral",
      "features_processed": 12,
      "quality_average": 0.87,
      "created_at": "2025-07-26T15:30:45.123456",
      "completed_at": "2025-07-26T15:54:33.789012",
      "processing_time": "24.3 minutes"
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
GET /jobs/{job_id}/export
```

**Description**: Download explanation results in various formats.

**Query Parameters**:
- `format` (string): Export format - `json`, `html`, `pdf`, `csv`, `txt`, `all`
- `features` (string): Comma-separated feature IDs to include
- `quality_filter` (string): Minimum quality level - `high`, `medium`, `low`

**Format Descriptions**:

| Format | Description | Size (typical) | Use Case |
|--------|-------------|---------------|----------|
| `json` | Complete structured data | 2.1MB | API integration, further processing |
| `html` | Interactive web report | 850KB | Human review, presentations |
| `pdf` | Professional document | 1.2MB | Reports, documentation |
| `csv` | Spreadsheet format | 120KB | Data analysis, Excel import |
| `txt` | Plain text summary | 45KB | Quick overview, text processing |
| `all` | ZIP bundle of all formats | 3.8MB | Complete archive |

**Example Requests**:
```bash
# Download complete JSON results
curl -X GET "http://localhost:8003/jobs/explain_20250726_153045_abc123/export?format=json" \
  -o explanations.json

# Download high-quality explanations only
curl -X GET "http://localhost:8003/jobs/explain_20250726_153045_abc123/export?format=html&quality_filter=high" \
  -o high_quality_report.html

# Download specific features as CSV
curl -X GET "http://localhost:8003/jobs/explain_20250726_153045_abc123/export?format=csv&features=348,410,256" \
  -o selected_features.csv
```

### Model Management

#### List Available Models

```http
GET /models
```

**Description**: Get available LLM models and their capabilities.

**Response Example**:
```json
{
  "available_models": [
    {
      "name": "llama3.1:8b",
      "status": "ready",
      "gpu_memory_mb": 8192,
      "target_gpu": "RTX_3080_Ti",
      "use_cases": ["simple_patterns", "quick_explanations"],
      "max_concurrent": 2,
      "parameters": {
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 200
      },
      "performance": {
        "average_generation_time": "1.2s",
        "tokens_per_second": 85,
        "quality_score": 0.82
      }
    },
    {
      "name": "llama3.1:70b",
      "status": "ready",
      "gpu_memory_mb": 20480,
      "target_gpu": "RTX_3090",
      "use_cases": ["complex_behavioral", "detailed_analysis"],
      "max_concurrent": 1,
      "parameters": {
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 300
      },
      "performance": {
        "average_generation_time": "8.7s",
        "tokens_per_second": 24,
        "quality_score": 0.91
      }
    },
    {
      "name": "codellama:13b",
      "status": "ready",
      "gpu_memory_mb": 12288,
      "target_gpu": "RTX_3080_Ti",
      "use_cases": ["technical_patterns", "code_analysis"],
      "max_concurrent": 1,
      "parameters": {
        "temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": 250
      }
    }
  ],
  "model_selection_strategy": {
    "technical_patterns": "codellama:13b",
    "complex_behavioral": "llama3.1:70b",
    "simple_explanations": "llama3.1:8b",
    "default": "llama3.1:8b"
  },
  "gpu_resources": {
    "rtx_3090": {
      "memory_total": "24GB",
      "memory_available": "18.2GB",
      "current_load": "RTX_3090 running llama3.1:70b"
    },
    "rtx_3080_ti": {
      "memory_total": "12GB", 
      "memory_available": "8.5GB",
      "current_load": "RTX_3080_Ti running llama3.1:8b"
    }
  }
}
```

#### Model Health Check

```http
GET /models/{model_name}/health
```

**Description**: Check specific model health and performance.

**Response Example**:
```json
{
  "model_name": "llama3.1:8b",
  "status": "healthy",
  "availability": "ready",
  "last_generation": "2025-07-26T15:29:12.345678",
  "performance_metrics": {
    "generations_completed": 127,
    "average_response_time": "1.2s",
    "success_rate": 0.98,
    "tokens_generated": 15420
  },
  "resource_usage": {
    "gpu_memory_used": "7.8GB",
    "gpu_utilization": "85%",
    "temperature": "72°C"
  },
  "configuration": {
    "temperature": 0.1,
    "top_p": 0.9,
    "max_tokens": 200
  }
}
```

## Data Models

### ExplanationRequest

```typescript
interface ExplanationRequest {
  request_id: string;                    // Unique request identifier
  analysis_type?: string;                // "technical_patterns" | "complex_behavioral" | "default"
  complexity?: string;                   // "low" | "medium" | "high"
  model?: string;                        // Specific model name (optional)
  input_data: FindResultInput | RawTextInput;  // Input data
}
```

### FindResultInput

```typescript
interface FindResultInput {
  find_job_id: string;                   // miStudioFind job ID
  feature_analysis: {                    // Feature analysis data
    [feature_id: string]: {
      coherence_score: number;
      quality_level: string;
      pattern_category: string;
      pattern_keywords: string[];
      top_activations: FeatureActivation[];
      activation_statistics: ActivationStats;
    };
  };
  summary_report: string;                // Text summary from analysis
}
```

### RawTextInput

```typescript
interface RawTextInput {
  text_corpus: string;                   // Raw text (min 100 chars)
  source_description?: string;           // Description of text source
}
```

### ExplanationResult

```typescript
interface ExplanationResult {
  job_id: string;                        // Job identifier
  request_id: string;                    // Original request ID
  status: "completed" | "failed";       // Job status
  processing_summary: {
    total_features_processed: number;
    high_quality_explanations: number;
    average_explanation_quality: number;
    total_processing_time: string;
  };
  explanations: FeatureExplanation[];   // Individual explanations
  summary_insights: SummaryInsights;    // Overall insights
  output_files: OutputFiles;            // Generated file paths
}
```

### FeatureExplanation

```typescript
interface FeatureExplanation {
  feature_id: number;                    // Feature identifier
  original_coherence: number;            // Original coherence score
  explanation: string;                   // Generated explanation text
  explanation_quality: number;           // Quality score (0-1)
  explanation_confidence: number;        // Confidence score (0-1)
  pattern_verification: "confirmed" | "uncertain" | "rejected";
  safety_assessment: "safe" | "concerning" | "requires_review";
  business_relevance: "high" | "medium" | "low";
  model_used: string;                    // LLM model used
  generation_time: string;               // Time taken
  token_count: number;                   // Tokens generated
}
```

## Code Examples

### Python Client

```python
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional

class MiStudioExplainClient:
    def __init__(self, base_url="http://localhost:8003"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health and LLM availability."""
        async with self.session.get(f"{self.base_url}/health") as response:
            response.raise_for_status()
            return await response.json()
    
    async def list_models(self) -> Dict[str, Any]:
        """Get available LLM models and capabilities."""
        async with self.session.get(f"{self.base_url}/models") as response:
            response.raise_for_status()
            return await response.json()
    
    async def generate_explanation(
        self,
        request_id: str,
        input_data: Dict[str, Any],
        analysis_type: str = "complex_behavioral",
        complexity: str = "medium",
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate explanation for feature analysis data."""
        
        payload = {
            "request_id": request_id,
            "analysis_type": analysis_type,
            "complexity": complexity,
            "input_data": input_data
        }
        
        if model:
            payload["model"] = model
        
        async with self.session.post(
            f"{self.base_url}/explain",
            json=payload
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get explanation job status."""
        async with self.session.get(f"{self.base_url}/jobs/{job_id}/status") as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_job_results(self, job_id: str, format_type: str = "json") -> Dict[str, Any]:
        """Get explanation job results."""
        params = {"format": format_type}
        async with self.session.get(
            f"{self.base_url}/jobs/{job_id}/results",
            params=params
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def download_export(
        self,
        job_id: str,
        format_type: str = "json",
        output_file: Optional[str] = None
    ) -> bytes:
        """Download explanation results in specified format."""
        params = {"format": format_type}
        async with self.session.get(
            f"{self.base_url}/jobs/{job_id}/export",
            params=params
        ) as response:
            response.raise_for_status()
            
            content = await response.read()
            
            if output_file:
                with open(output_file, 'wb') as f:
                    f.write(content)
            
            return content
    
    async def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 5,
        timeout: int = 1800  # 30 minutes
    ) -> Dict[str, Any]:
        """Wait for job completion with progress monitoring."""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = await self.get_job_status(job_id)
            
            print(f"Status: {status['status']} - {status.get('progress', {}).get('progress_percentage', 0):.1f}%")
            
            if status['status'] in ['completed', 'failed', 'cancelled']:
                return status
            
            await asyncio.sleep(poll_interval)
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

# Usage example
async def explain_features():
    async with MiStudioExplainClient() as client:
        # 1. Check service health
        health = await client.health_check()
        print(f"Service status: {health['status']}")
        
        # 2. Check available models
        models = await client.list_models()
        print(f"Available models: {[model['name'] for model in models['available_models']]}")
        
        # 3. Prepare feature analysis data
        feature_data = {
            "find_job_id": "find_20250726_151321_d9675dce",
            "feature_analysis": {
                "feature_348": {
                    "coherence_score": 0.501,
                    "quality_level": "medium",
                    "pattern_category": "technical",
                    "pattern_keywords": ["json", "schema", "validation"],
                    "top_activations": [
                        {
                            "text": "JSON schema validation patterns for API documentation...",
                            "activation_strength": 0.89,
                            "context": "API documentation"
                        }
                    ],
                    "activation_statistics": {
                        "mean": 0.15,
                        "std": 0.08,
                        "frequency": 0.023
                    }
                }
            },
            "summary_report": "Analysis of 512 features from microsoft/phi-4 model"
        }
        
        # 4. Generate explanation
        result = await client.generate_explanation(
            request_id="explain_001",
            input_data=feature_data,
            analysis_type="technical_patterns",
            complexity="medium",
            model="codellama:13b"  # Use code-specialized model
        )
        
        job_id = result["request_id"]
        print(f"Started explanation job: {job_id}")
        
        # 5. Wait for completion
        final_status = await client.wait_for_completion(job_id)
        
        if final_status['status'] == 'completed':
            # 6. Get results
            results = await client.get_job_results(job_id)
            
            print(f"\n✅ Explanation completed!")
            print(f"Features explained: {results['processing_summary']['total_features_processed']}")
            print(f"Average quality: {results['processing_summary']['average_explanation_quality']:.2f}")
            
            # 7. Download different formats
            await client.download_export(job_id, "json", "explanations.json")
            await client.download_export(job_id, "html", "report.html")
            await client.download_export(job_id, "pdf", "report.pdf")
            
            print("📄 Results downloaded in multiple formats")
            
            # 8. Display sample explanation
            if results['explanations']:
                sample = results['explanations'][0]
                print(f"\n📝 Sample Explanation (Feature {sample['feature_id']}):")
                print(f"Quality: {sample['explanation_quality']:.2f}")
                print(f"Text: {sample['explanation'][:200]}...")
        
        else:
            print(f"❌ Job failed: {final_status.get('error', 'Unknown error')}")

# Run the example
asyncio.run(explain_features())
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');
const fs = require('fs');

class MiStudioExplainClient {
    constructor(baseUrl = 'http://localhost:8003') {
        this.baseUrl = baseUrl;
        this.client = axios.create({ 
            baseURL: baseUrl,
            timeout: 60000  // 60 seconds timeout
        });
    }
    
    async healthCheck() {
        const response = await this.client.get('/health');
        return response.data;
    }
    
    async listModels() {
        const response = await this.client.get('/models');
        return response.data;
    }
    
    async generateExplanation({
        requestId,
        inputData,
        analysisType = 'complex_behavioral',
        complexity = 'medium',
        model = null
    }) {
        const payload = {
            request_id: requestId,
            analysis_type: analysisType,
            complexity: complexity,
            input_data: inputData
        };
        
        if (model) {
            payload.model = model;
        }
        
        const response = await this.client.post('/explain', payload);
        return response.data;
    }
    
    async getJobStatus(jobId) {
        const response = await this.client.get(`/jobs/${jobId}/status`);
        return response.data;
    }
    
    async getJobResults(jobId, format = 'json') {
        const response = await this.client.get(`/jobs/${jobId}/results`, {
            params: { format }
        });
        return response.data;
    }
    
    async downloadExport(jobId, format = 'json', outputFile = null) {
        const response = await this.client.get(`/jobs/${jobId}/export`, {
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
    
    async waitForCompletion(jobId, pollInterval = 5000, timeout = 1800000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            const status = await this.getJobStatus(jobId);
            
            console.log(`Status: ${status.status} - ${status.progress?.progress_percentage || 0}%`);
            
            if (['completed', 'failed', 'cancelled'].includes(status.status)) {
                return status;
            }
            
            await new Promise(resolve => setTimeout(resolve, pollInterval));
        }
        
        throw new Error(`Job ${jobId} did not complete within ${timeout}ms`);
    }
}

// Usage example
async function explainBehavioralPatterns() {
    const client = new MiStudioExplainClient();
    
    try {
        // 1. Health check
        const health = await client.healthCheck();
        console.log(`🟢 Service status: ${health.status}`);
        
        // 2. Check LLM availability
        if (!health.llm_diagnostics.endpoint_reachable) {
            throw new Error('LLM service not available');
        }
        
        // 3. Prepare behavioral analysis data
        const behavioralData = {
            find_job_id: "find_20250726_151321_d9675dce",
            feature_analysis: {
                "feature_335": {
                    "coherence_score": 0.443,
                    "quality_level": "medium",
                    "pattern_category": "conversational",
                    "pattern_keywords": ["chat", "conversation", "dialogue"],
                    "top_activations": [
                        {
                            "text": "User: Hello, how can I help you today? Assistant: I'd be happy to help...",
                            "activation_strength": 0.76,
                            "context": "Chat conversation"
                        }
                    ]
                },
                "feature_254": {
                    "coherence_score": 0.437,
                    "quality_level": "medium", 
                    "pattern_category": "behavioral",
                    "pattern_keywords": ["personality", "traits", "assessment"],
                    "top_activations": [
                        {
                            "text": "The individual demonstrates high conscientiousness and analytical thinking...",
                            "activation_strength": 0.82,
                            "context": "Personality assessment"
                        }
                    ]
                }
            },
            summary_report: "Behavioral pattern analysis showing conversational and personality assessment capabilities"
        };
        
        // 4. Generate explanation with behavioral focus
        const result = await client.generateExplanation({
            requestId: 'behavioral_analysis_001',
            inputData: behavioralData,
            analysisType: 'complex_behavioral',
            complexity: 'high',
            model: 'llama3.1:70b'  // Use larger model for complex behavioral analysis
        });
        
        console.log(`🚀 Started behavioral analysis: ${result.request_id}`);
        
        // 5. Monitor progress
        const finalStatus = await client.waitForCompletion(result.request_id);
        
        if (finalStatus.status === 'completed') {
            // 6. Get detailed results
            const results = await client.getJobResults(result.request_id);
            
            console.log('\n✅ Behavioral Analysis Complete!');
            console.log(`📊 Features explained: ${results.processing_summary.total_features_processed}`);
            console.log(`🎯 Quality average: ${results.processing_summary.average_explanation_quality.toFixed(2)}`);
            
            // 7. Display behavioral insights
            const behavioralFeatures = results.explanations.filter(
                exp => exp.pattern_verification === 'confirmed' && 
                       exp.business_relevance === 'high'
            );
            
            console.log(`\n🧠 High-Value Behavioral Patterns (${behavioralFeatures.length}):`);
            behavioralFeatures.forEach(feature => {
                console.log(`\n🔍 Feature ${feature.feature_id}:`);
                console.log(`   Quality: ${feature.explanation_quality.toFixed(2)}`);
                console.log(`   Safety: ${feature.safety_assessment}`);
                console.log(`   Text: ${feature.explanation.substring(0, 150)}...`);
            });
            
            // 8. Download comprehensive report
            await client.downloadExport(result.request_id, 'html', 'behavioral_report.html');
            await client.downloadExport(result.request_id, 'pdf', 'behavioral_report.pdf');
            
            console.log('\n📄 Behavioral analysis reports downloaded');
            
        } else {
            console.error(`❌ Analysis failed: ${finalStatus.error || 'Unknown error'}`);
        }
        
    } catch (error) {
        console.error('Error during behavioral analysis:', error.message);
    }
}

// Run behavioral analysis
explainBehavioralPatterns();
```

### cURL Examples

#### Complete Workflow Script

```bash
#!/bin/bash

# miStudioExplain Complete Workflow
# Demonstrates explanation generation for AI behavioral patterns

BASE_URL="http://localhost:8003"
REQUEST_ID="explain_$(date +%Y%m%d_%H%M%S)_$(shuf -i 1000-9999 -n 1)"

echo "🤖 miStudioExplain Workflow - AI Behavioral Analysis"
echo "=================================================="

# 1. Health check and service validation
echo -e "\n1. 🔍 Checking service health..."
HEALTH_RESPONSE=$(curl -s "$BASE_URL/health")
echo "$HEALTH_RESPONSE" | jq '.'

# Validate LLM connectivity
LLM_HEALTHY=$(echo "$HEALTH_RESPONSE" | jq -r '.llm_diagnostics.endpoint_reachable')
if [ "$LLM_HEALTHY" != "true" ]; then
    echo "❌ LLM service not available"
    exit 1
fi

echo "✅ Service and LLM connectivity confirmed"

# 2. Check available models
echo -e "\n2. 🤖 Checking available models..."
MODELS_RESPONSE=$(curl -s "$BASE_URL/models")
echo "$MODELS_RESPONSE" | jq '.available_models[] | {name: .name, status: .status, use_cases: .use_cases}'

# 3. Prepare complex behavioral analysis data
echo -e "\n3. 📊 Preparing behavioral analysis data..."
cat > behavioral_features.json << 'EOF'
{
  "request_id": "PLACEHOLDER_REQUEST_ID",
  "analysis_type": "complex_behavioral",
  "complexity": "high",
  "model": "llama3.1:70b",
  "input_data": {
    "find_job_id": "find_20250726_151321_d9675dce",
    "feature_analysis": {
      "feature_335": {
        "coherence_score": 0.443,
        "quality_level": "medium",
        "pattern_category": "conversational",
        "pattern_keywords": ["chat", "conversation", "dialogue", "role"],
        "top_activations": [
          {
            "text": "User: Hello, how can I help you today? Assistant: I'd be happy to help you with any questions or tasks you have. What would you like to know?",
            "activation_strength": 0.76,
            "context": "Chat conversation structure"
          },
          {
            "text": "The conversation follows a clear pattern: greeting, inquiry, response, and follow-up engagement.",
            "activation_strength": 0.71,
            "context": "Dialogue analysis"
          }
        ],
        "activation_statistics": {
          "mean": 0.18,
          "std": 0.12,
          "frequency": 0.034
        }
      },
      "feature_254": {
        "coherence_score": 0.437,
        "quality_level": "medium",
        "pattern_category": "behavioral",
        "pattern_keywords": ["personality", "traits", "assessment", "psychology"],
        "top_activations": [
          {
            "text": "The individual demonstrates high conscientiousness and analytical thinking patterns, showing systematic approach to problem-solving.",
            "activation_strength": 0.82,
            "context": "Personality assessment"
          },
          {
            "text": "Behavioral indicators suggest strong attention to detail and preference for structured environments.",
            "activation_strength": 0.78,
            "context": "Trait analysis"
          }
        ],
        "activation_statistics": {
          "mean": 0.21,
          "std": 0.09,
          "frequency": 0.028
        }
      },
      "feature_410": {
        "coherence_score": 0.488,
        "quality_level": "medium",
        "pattern_category": "temporal",
        "pattern_keywords": ["time", "schedule", "planning", "calendar"],
        "top_activations": [
          {
            "text": "Meeting scheduled for 2:00 PM next Tuesday, followed by review session at 3:30 PM.",
            "activation_strength": 0.89,
            "context": "Time management"
          }
        ]
      }
    },
    "summary_report": "Comprehensive behavioral pattern analysis of microsoft/phi-4 model showing sophisticated understanding of conversational dynamics, personality assessment, and temporal reasoning. Analysis identified 3 high-coherence features demonstrating human-like behavioral pattern recognition."
  }
}
EOF

# Replace placeholder with actual request ID
sed -i "s/PLACEHOLDER_REQUEST_ID/$REQUEST_ID/g" behavioral_features.json

echo "✅ Behavioral analysis data prepared"

# 4. Submit explanation request
echo -e "\n4. 🚀 Submitting explanation request..."
EXPLANATION_RESPONSE=$(curl -s -X POST "$BASE_URL/explain" \
    -H "Content-Type: application/json" \
    -d @behavioral_features.json)

echo "$EXPLANATION_RESPONSE" | jq '.'

# Extract job details
JOB_ID=$(echo "$EXPLANATION_RESPONSE" | jq -r '.request_id')
if [ "$JOB_ID" == "null" ]; then
    echo "❌ Failed to start explanation job"
    exit 1
fi

echo "🎯 Job started: $JOB_ID"

# 5. Monitor explanation progress
echo -e "\n5. ⏱️  Monitoring explanation generation..."
START_TIME=$(date +%s)
TIMEOUT=1800  # 30 minutes

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo "⏰ Timeout reached after ${TIMEOUT}s"
        break
    fi
    
    STATUS_RESPONSE=$(curl -s "$BASE_URL/jobs/$JOB_ID/status")
    STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')
    PROGRESS=$(echo "$STATUS_RESPONSE" | jq -r '.progress.progress_percentage // 0')
    CURRENT_STEP=$(echo "$STATUS_RESPONSE" | jq -r '.progress.current_step // "unknown"')
    
    echo "Status: $STATUS | Progress: $PROGRESS% | Step: $CURRENT_STEP | Elapsed: ${ELAPSED}s"
    
    if [ "$STATUS" == "completed" ]; then
        echo "✅ Explanation generation completed!"
        break
    elif [ "$STATUS" == "failed" ]; then
        echo "❌ Explanation generation failed"
        echo "$STATUS_RESPONSE" | jq '.error'
        exit 1
    fi
    
    sleep 10
done

# 6. Retrieve comprehensive results
echo -e "\n6. 📋 Retrieving explanation results..."
RESULTS=$(curl -s "$BASE_URL/jobs/$JOB_ID/results?include_metadata=true")

# Display summary
echo "📊 Processing Summary:"
echo "$RESULTS" | jq '.processing_summary'

echo -e "\n🧠 Summary Insights:"
echo "$RESULTS" | jq '.summary_insights'

# Display high-quality explanations
echo -e "\n🎯 High-Quality Behavioral Explanations:"
echo "$RESULTS" | jq '.explanations[] | select(.explanation_quality > 0.8) | {
    feature_id: .feature_id,
    quality: .explanation_quality,
    safety: .safety_assessment,
    relevance: .business_relevance,
    explanation: .explanation[:150] + "..."
}'

# 7. Download multiple formats
echo -e "\n7. 💾 Downloading explanation results..."

# JSON - Complete structured data
curl -s "$BASE_URL/jobs/$JOB_ID/export?format=json" -o "behavioral_explanations_$JOB_ID.json"
echo "✅ JSON results saved"

# HTML - Interactive report
curl -s "$BASE_URL/jobs/$JOB_ID/export?format=html" -o "behavioral_report_$JOB_ID.html"
echo "✅ HTML report saved"

# PDF - Professional document
curl -s "$BASE_URL/jobs/$JOB_ID/export?format=pdf" -o "behavioral_report_$JOB_ID.pdf"
echo "✅ PDF report saved"

# CSV - Data analysis format
curl -s "$BASE_URL/jobs/$JOB_ID/export?format=csv&quality_filter=medium" -o "behavioral_data_$JOB_ID.csv"
echo "✅ CSV data saved"

# Complete archive
curl -s "$BASE_URL/jobs/$JOB_ID/export?format=all" -o "complete_behavioral_analysis_$JOB_ID.zip"
echo "✅ Complete archive saved"

# 8. Quality and safety assessment
echo -e "\n8. 🔒 Safety and Quality Assessment:"
SAFETY_CONCERNS=$(echo "$RESULTS" | jq '.summary_insights.safety_concerns_identified')
HIGH_QUALITY_COUNT=$(echo "$RESULTS" | jq '.summary_insights.high_quality_explanations')
BUSINESS_CRITICAL=$(echo "$RESULTS" | jq '.summary_insights.business_critical_features')

echo "Safety concerns identified: $SAFETY_CONCERNS"
echo "High-quality explanations: $HIGH_QUALITY_COUNT"  
echo "Business-critical features: $BUSINESS_CRITICAL"

if [ "$SAFETY_CONCERNS" -gt 0 ]; then
    echo "⚠️  Safety concerns detected - review required"
    echo "$RESULTS" | jq '.explanations[] | select(.safety_assessment != "safe") | {
        feature_id: .feature_id,
        safety: .safety_assessment,
        explanation: .explanation[:100] + "..."
    }'
fi

# 9. Integration preparation
echo -e "\n9. 🔗 Preparing for downstream integration..."
INTEGRATION_DATA=$(echo "$RESULTS" | jq '{
    job_id: .job_id,
    explanations_ready_for_scoring: [.explanations[] | select(.explanation_quality > 0.7) | .feature_id],
    behavioral_patterns_identified: .summary_insights.behavioral_pattern_count,
    technical_patterns_identified: .summary_insights.technical_pattern_count,
    ready_for_next_step: (.summary_insights.high_quality_explanations > 5)
}')

echo "$INTEGRATION_DATA" | jq '.'

READY_FOR_SCORING=$(echo "$INTEGRATION_DATA" | jq -r '.ready_for_next_step')
EXPLANATION_COUNT=$(echo "$INTEGRATION_DATA" | jq -r '.explanations_ready_for_scoring | length')

if [ "$READY_FOR_SCORING" == "true" ]; then
    echo "✅ Ready for miStudioScore integration ($EXPLANATION_COUNT explanations)"
else
    echo "⚠️  May need additional explanation refinement"
fi

# Cleanup
rm behavioral_features.json

echo -e "\n🎉 Behavioral Analysis Workflow Complete!"
echo "Files generated:"
echo "  - behavioral_explanations_$JOB_ID.json (structured data)"
echo "  - behavioral_report_$JOB_ID.html (interactive report)" 
echo "  - behavioral_report_$JOB_ID.pdf (professional document)"
echo "  - behavioral_data_$JOB_ID.csv (analysis data)"
echo "  - complete_behavioral_analysis_$JOB_ID.zip (complete archive)"

echo -e "\n🔬 Next Steps:"
echo "1. Review high-quality explanations in HTML report"
echo "2. Address any safety concerns identified"
echo "3. Use JSON output for miStudioScore integration"
echo "4. Share PDF report with stakeholders"
```

#### Batch Processing Script

```bash
#!/bin/bash

# Batch process multiple feature sets
FEATURE_SETS=("technical_patterns" "behavioral_analysis" "conversational_dynamics")
BASE_URL="http://localhost:8003"

echo "🔄 Batch Processing Multiple Feature Sets"
echo "========================================"

declare -A JOB_IDS
declare -A FEATURE_COUNTS

# Start all jobs
for FEATURE_SET in "${FEATURE_SETS[@]}"; do
    echo "Starting analysis for $FEATURE_SET..."
    
    REQUEST_ID="batch_${FEATURE_SET}_$(date +%Y%m%d_%H%M%S)"
    
    # Create appropriate payload for each feature set
    case $FEATURE_SET in
        "technical_patterns")
            MODEL="codellama:13b"
            ANALYSIS_TYPE="technical_patterns"
            FEATURE_COUNT=15
            ;;
        "behavioral_analysis")
            MODEL="llama3.1:70b"
            ANALYSIS_TYPE="complex_behavioral"
            FEATURE_COUNT=8
            ;;
        "conversational_dynamics")
            MODEL="llama3.1:8b"
            ANALYSIS_TYPE="complex_behavioral"
            FEATURE_COUNT=12
            ;;
    esac
    
    PAYLOAD=$(cat << EOF
{
  "request_id": "$REQUEST_ID",
  "analysis_type": "$ANALYSIS_TYPE",
  "complexity": "medium",
  "model": "$MODEL",
  "input_data": {
    "find_job_id": "find_20250726_151321_d9675dce",
    "feature_analysis": {},
    "summary_report": "Batch processing $FEATURE_SET with $FEATURE_COUNT features"
  }
}
EOF
)
    
    RESPONSE=$(curl -s -X POST "$BASE_URL/explain" \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD")
    
    JOB_ID=$(echo "$RESPONSE" | jq -r '.request_id')
    JOB_IDS[$FEATURE_SET]=$JOB_ID
    FEATURE_COUNTS[$FEATURE_SET]=$FEATURE_COUNT
    
    echo "Started $FEATURE_SET: $JOB_ID"
done

echo -e "\n⏱️  Monitoring all jobs..."

# Monitor all jobs
while true; do
    ALL_COMPLETE=true
    
    for FEATURE_SET in "${FEATURE_SETS[@]}"; do
        JOB_ID=${JOB_IDS[$FEATURE_SET]}
        
        STATUS_RESPONSE=$(curl -s "$BASE_URL/jobs/$JOB_ID/status")
        STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')
        PROGRESS=$(echo "$STATUS_RESPONSE" | jq -r '.progress.progress_percentage // 0')
        
        echo "[$FEATURE_SET] Status: $STATUS | Progress: $PROGRESS%"
        
        if [ "$STATUS" != "completed" ] && [ "$STATUS" != "failed" ]; then
            ALL_COMPLETE=false
        fi
    done
    
    if [ "$ALL_COMPLETE" = true ]; then
        break
    fi
    
    echo "Waiting for completion..."
    sleep 15
done

echo -e "\n📊 Batch Processing Results Summary:"

# Collect results from all jobs
for FEATURE_SET in "${FEATURE_SETS[@]}"; do
    JOB_ID=${JOB_IDS[$FEATURE_SET]}
    
    RESULTS=$(curl -s "$BASE_URL/jobs/$JOB_ID/results")
    STATUS=$(echo "$RESULTS" | jq -r '.status // "unknown"')
    
    if [ "$STATUS" == "completed" ]; then
        QUALITY=$(echo "$RESULTS" | jq -r '.processing_summary.average_explanation_quality // 0')
        PROCESSED=$(echo "$RESULTS" | jq -r '.processing_summary.total_features_processed // 0')
        
        echo "✅ $FEATURE_SET: $PROCESSED features, avg quality: $QUALITY"
        
        # Download results
        curl -s "$BASE_URL/jobs/$JOB_ID/export?format=json" -o "batch_${FEATURE_SET}_results.json"
        curl -s "$BASE_URL/jobs/$JOB_ID/export?format=html" -o "batch_${FEATURE_SET}_report.html"
    else
        echo "❌ $FEATURE_SET: Failed"
    fi
done

echo -e "\n🎯 Batch processing complete! Check individual reports for detailed analysis."
```

## Best Practices

### 1. Input Preparation

Optimize feature analysis data for best explanation quality:

```python
def prepare_optimal_feature_data(find_results):
    """Prepare feature data optimized for explanation generation."""
    
    # Filter for high-quality features
    quality_features = {
        feature_id: data for feature_id, data in find_results['features'].items()
        if data.get('coherence_score', 0) >= 0.4 and
           data.get('quality_level') in ['excellent', 'good', 'fair']
    }
    
    # Prioritize by business relevance and interpretability
    prioritized_features = {}
    for feature_id, data in quality_features.items():
        # Enhance with additional context
        enhanced_data = data.copy()
        
        # Add business context based on pattern category
        category = data.get('pattern_category', 'unknown')
        if category == 'behavioral':
            enhanced_data['business_context'] = 'AI decision-making pattern'
        elif category == 'technical':
            enhanced_data['business_context'] = 'Technical capability indicator'
        elif category == 'conversational':
            enhanced_data['business_context'] = 'Communication pattern'
        
        # Limit top activations for focused analysis
        if 'top_activations' in enhanced_data:
            enhanced_data['top_activations'] = enhanced_data['top_activations'][:5]
        
        prioritized_features[feature_id] = enhanced_data
    
    return {
        'find_job_id': find_results['job_id'],
        'feature_analysis': prioritized_features,
        'summary_report': f"Optimized analysis of {len(prioritized_features)} high-quality features"
    }
```

### 2. Model Selection Strategy

Choose optimal models based on analysis requirements:

```python
def select_optimal_model(features, complexity='medium'):
    """Select the best model based on feature characteristics."""
    
    # Analyze feature composition
    categories = [f.get('pattern_category', 'unknown') for f in features.values()]
    category_counts = {cat: categories.count(cat) for cat in set(categories)}
    
    # Count technical vs behavioral patterns
    technical_count = category_counts.get('technical', 0)
    behavioral_count = category_counts.get('behavioral', 0)
    total_features = len(features)
    
    # Model selection logic
    if technical_count > total_features * 0.6:
        return 'codellama:13b'  # Technical patterns
    elif behavioral_count > total_features * 0.4 or complexity == 'high':
        return 'llama3.1:70b'   # Complex behavioral analysis
    else:
        return 'llama3.1:8b'    # General purpose
```

### 3. Quality Assessment

Implement quality checks for generated explanations:

```python
def assess_explanation_quality(explanation_result):
    """Assess the quality of generated explanations."""
    
    quality_metrics = {
        'completeness': 0.0,
        'clarity': 0.0,
        'accuracy': 0.0,
        'actionability': 0.0
    }
    
    explanations = explanation_result.get('explanations', [])
    
    for explanation in explanations:
        exp_text = explanation.get('explanation', '')
        
        # Completeness check
        if len(exp_text) >= 100 and 'pattern' in exp_text.lower():
            quality_metrics['completeness'] += 1
        
        # Clarity check
        if any(word in exp_text.lower() for word in ['detects', 'identifies', 'recognizes']):
            quality_metrics['clarity'] += 1
        
        # Accuracy check (based on confidence scores)
        if explanation.get('explanation_confidence', 0) >= 0.8:
            quality_metrics['accuracy'] += 1
        
        # Actionability check
        if any(word in exp_text.lower() for word in ['significant', 'important', 'indicates']):
            quality_metrics['actionability'] += 1
    
    # Normalize scores
    total_explanations = len(explanations)
    if total_explanations > 0:
        for metric in quality_metrics:
            quality_metrics[metric] /= total_explanations
    
    return quality_metrics
```

### 4. Error Recovery

Implement robust error handling:

```python
async def robust_explanation_workflow(client, feature_data, max_retries=3):
    """Generate explanations with comprehensive error recovery."""
    
    for attempt in range(max_retries):
        try:
            # Select model based on attempt
            if attempt == 0:
                model = None  # Auto-select
            elif attempt == 1:
                model = 'llama3.1:8b'  # Fallback to smaller model
            else:
                model = 'llama3.1:70b'  # Try larger model
            
            result = await client.generate_explanation(
                request_id=f"retry_{attempt}_{int(time.time())}",
                input_data=feature_data,
                model=model
            )
            
            # Wait for completion
            status = await client.wait_for_completion(result['request_id'])
            
            if status['status'] == 'completed':
                return await client.get_job_results(result['request_id'])
            
            # If failed, analyze the error
            error_msg = status.get('error', 'Unknown error')
            
            if 'memory' in error_msg.lower() and attempt < max_retries - 1:
                print(f"Memory error on attempt {attempt + 1}, reducing complexity...")
                # Reduce feature count for next attempt
                feature_count = len(feature_data.get('feature_analysis', {}))
                reduced_count = max(5, feature_count // 2)
                feature_data = reduce_feature_set(feature_data, reduced_count)
                continue
            
            elif 'timeout' in error_msg.lower() and attempt < max_retries - 1:
                print(f"Timeout on attempt {attempt + 1}, trying simpler analysis...")
                continue
            
            else:
                raise Exception(f"Explanation failed: {error_msg}")
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"All retry attempts failed: {e}")
            
            print(f"Attempt {attempt + 1} failed: {e}, retrying...")
            await asyncio.sleep(5 * (attempt + 1))  # Exponential backoff

def reduce_feature_set(feature_data, target_count):
    """Reduce feature set to most important features."""
    features = feature_data.get('feature_analysis', {})
    
    # Sort by coherence score
    sorted_features = sorted(
        features.items(),
        key=lambda x: x[1].get('coherence_score', 0),
        reverse=True
    )
    
    # Take top features
    reduced_features = dict(sorted_features[:target_count])
    
    return {
        **feature_data,
        'feature_analysis': reduced_features,
        'summary_report': f"Reduced analysis of {target_count} highest-coherence features"
    }
```

### 5. Performance Optimization

Optimize explanation generation for large feature sets:

```python
async def batch_explanation_processor(client, feature_data, batch_size=10):
    """Process large feature sets in optimized batches."""
    
    features = feature_data.get('feature_analysis', {})
    feature_items = list(features.items())
    
    # Split into batches
    batches = [
        dict(feature_items[i:i + batch_size])
        for i in range(0, len(feature_items), batch_size)
    ]
    
    all_results = []
    
    for i, batch_features in enumerate(batches):
        print(f"Processing batch {i + 1}/{len(batches)} ({len(batch_features)} features)...")
        
        batch_data = {
            **feature_data,
            'feature_analysis': batch_features,
            'summary_report': f"Batch {i + 1} of {len(batches)} - {len(batch_features)} features"
        }
        
        # Select model based on batch characteristics
        model = select_optimal_model(batch_features)
        
        try:
            result = await client.generate_explanation(
                request_id=f"batch_{i + 1}_{int(time.time())}",
                input_data=batch_data,
                model=model,
                complexity='medium'
            )
            
            status = await client.wait_for_completion(result['request_id'])
            
            if status['status'] == 'completed':
                batch_results = await client.get_job_results(result['request_id'])
                all_results.extend(batch_results.get('explanations', []))
                print(f"✅ Batch {i + 1} completed successfully")
            else:
                print(f"❌ Batch {i + 1} failed: {status.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"❌ Batch {i + 1} error: {e}")
            continue
    
    return {
        'total_batches_processed': len(batches),
        'total_explanations_generated': len(all_results),
        'explanations': all_results,
        'processing_summary': {
            'batch_size': batch_size,
            'success_rate': len(all_results) / len(feature_items) if feature_items else 0
        }
    }
```

## Integration with miStudio Pipeline

### Upstream Integration (miStudioFind)

```python
def seamless_find_to_explain_workflow(find_client, explain_client, find_job_id):
    """Complete workflow from feature analysis to explanation generation."""
    
    # 1. Get miStudioFind results
    find_results = find_client.get_job_results(find_job_id)
    
    if not find_results.get('ready_for_explain_service'):
        raise Exception("Find results not ready for explanation")
    
    print(f"🔍 Find analysis complete: {find_results['summary']['interpretable_features']} interpretable features")
    
    # 2. Prepare explanation input
    explanation_input = prepare_optimal_feature_data(find_results)
    
    # 3. Start explanation generation
    explain_job = explain_client.generate_explanation(
        request_id=f"auto_explain_{find_job_id}",
        input_data=explanation_input,
        analysis_type="complex_behavioral",
        complexity="medium"
    )
    
    print(f"💭 Explanation generation started: {explain_job['request_id']}")
    
    # 4. Monitor explanation progress
    explain_status = explain_client.wait_for_completion(explain_job['request_id'])
    
    if explain_status['status'] == 'completed':
        explain_results = explain_client.get_job_results(explain_job['request_id'])
        
        return {
            'find_job_id': find_job_id,
            'explain_job_id': explain_job['request_id'],
            'original_features': find_results['summary']['total_features_analyzed'],
            'interpretable_features': find_results['summary']['interpretable_features'],
            'explained_features': explain_results['processing_summary']['total_features_processed'],
            'high_quality_explanations': explain_results['processing_summary']['high_quality_explanations'],
            'ready_for_scoring': explain_results['processing_summary']['high_quality_explanations'] >= 5
        }
    else:
        raise Exception(f"Explanation generation failed: {explain_status.get('error')}")
```

### Downstream Integration (miStudioScore)

```python
def prepare_for_scoring_service(explain_results, quality_threshold=0.7):
    """Prepare miStudioExplain results for miStudioScore input."""
    
    explanations = explain_results.get('explanations', [])
    
    # Filter for scoring-ready explanations
    scoring_candidates = []
    
    for explanation in explanations:
        if (explanation.get('explanation_quality', 0) >= quality_threshold and
            explanation.get('pattern_verification') == 'confirmed' and
            explanation.get('safety_assessment') in ['safe', 'neutral']):
            
            scoring_candidates.append({
                'feature_id': explanation['feature_id'],
                'explanation_text': explanation['explanation'],
                'explanation_quality': explanation['explanation_quality'],
                'explanation_confidence': explanation['explanation_confidence'],
                'business_relevance': explanation['business_relevance'],
                'pattern_category': explanation.get('pattern_category', 'unknown'),
                'safety_assessment': explanation['safety_assessment'],
                'model_used': explanation['model_used']
            })
    
    return {
        'source_explain_job': explain_results['job_id'],
        'explanations_for_scoring': scoring_candidates,
        'quality_distribution': {
            'high_quality': len([e for e in explanations if e.get('explanation_quality', 0) >= 0.8]),
            'medium_quality': len([e for e in explanations if 0.6 <= e.get('explanation_quality', 0) < 0.8]),
            'low_quality': len([e for e in explanations if e.get('explanation_quality', 0) < 0.6])
        },
        'safety_summary': {
            'safe_explanations': len([e for e in explanations if e.get('safety_assessment') == 'safe']),
            'concerning_explanations': len([e for e in explanations if e.get('safety_assessment') == 'concerning']),
            'review_required': len([e for e in explanations if e.get('safety_assessment') == 'requires_review'])
        },
        'ready_for_scoring': len(scoring_candidates) >= 5
    }
```

## Troubleshooting

### Common Issues

#### 1. LLM Service Unavailable
```
Error: Ollama service not available
```

**Causes**:
- Ollama service not running
- Network connectivity issues
- Model not loaded

**Solutions**:
```bash
# Check Ollama service status
curl -s "http://ollama.mcslab.io/api/tags" | jq '.'

# Restart Ollama service (if you have access)
kubectl restart deployment ollama -n mistudio-services

# Check model availability
curl -s "http://localhost:8003/models" | jq '.available_models'
```

#### 2. Memory Errors During Generation
```
Error: CUDA out of memory during explanation generation
```

**Solutions**:
- Use smaller model (llama3.1:8b instead of llama3.1:70b)
- Reduce batch size
- Process features individually
- Clear GPU memory between requests

```python
# Memory-efficient processing
async def memory_efficient_explanation(client, feature_data):
    # Use smaller model
    result = await client.generate_explanation(
        request_id="memory_safe_001",
        input_data=feature_data,
        model="llama3.1:8b",  # Smaller model
        complexity="medium"   # Reduce complexity
    )
    
    return result
```

#### 3. Quality Validation Failures
```
Warning: Validation failed for explanation - low coherence
```

**Causes**:
- Poor input feature quality
- Model hallucination
- Insufficient context

**Solutions**:
```python
# Improve input quality
def enhance_feature_context(feature_data):
    """Add more context to improve explanation quality."""
    
    for feature_id, data in feature_data['feature_analysis'].items():
        # Add pattern examples
        if 'top_activations' in data:
            # Ensure we have diverse examples
            activations = data['top_activations'][:10]  # More examples
            data['enhanced_context'] = {
                'activation_examples': len(activations),
                'pattern_diversity': calculate_diversity(activations),
                'business_context': get_business_context(data.get('pattern_category'))
            }
    
    return feature_data
```

#### 4. Timeout Issues
```
Error: Explanation generation timed out
```

**Solutions**:
- Increase timeout settings
- Reduce feature complexity
- Use faster models
- Process in smaller batches

### Debugging Tools

#### Enable Debug Logging
```python
import logging

# Enable debug logging for detailed troubleshooting
logging.basicConfig(level=logging.DEBUG)

# Specific component logging
logging.getLogger('miStudioExplain.explanation_generator').setLevel(logging.DEBUG)
logging.getLogger('miStudioExplain.ollama_manager').setLevel(logging.DEBUG)
```

#### Health Diagnostics
```python
async def comprehensive_health_check(client):
    """Perform detailed system diagnostics."""
    
    print("🔍 miStudioExplain Health Diagnostics")
    print("=" * 50)
    
    # 1. Service health
    try:
        health = await client.health_check()
        print(f"✅ Service: {health['status']}")
        print(f"📊 Version: {health['version']}")
        
        # LLM diagnostics
        llm_status = health.get('llm_diagnostics', {})
        if llm_status.get('endpoint_reachable'):
            print(f"✅ LLM: Connected to {llm_status.get('models_loaded', 0)} models")
        else:
            print("❌ LLM: Not reachable")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # 2. Model availability
    try:
        models = await client.list_models()
        available_models = [m for m in models['available_models'] if m['status'] == 'ready']
        
        print(f"\n🤖 Available Models: {len(available_models)}")
        for model in available_models:
            print(f"   ✅ {model['name']}: {model['use_cases']}")
            
    except Exception as e:
        print(f"❌ Model check failed: {e}")
    
    # 3. Recent job performance
    try:
        jobs = await client.list_jobs()
        completed_jobs = [job for job in jobs['jobs'] if job['status'] == 'completed']
        failed_jobs = [job for job in jobs['jobs'] if job['status'] == 'failed']
        
        print(f"\n📈 Recent Performance:")
        print(f"   Completed: {len(completed_jobs)}")
        print(f"   Failed: {len(failed_jobs)}")
        
        if completed_jobs:
            avg_quality = sum(job.get('quality_average', 0) for job in completed_jobs) / len(completed_jobs)
            print(f"   Average Quality: {avg_quality:.2f}")
        
    except Exception as e:
        print(f"❌ Performance check failed: {e}")
    
    # 4. Resource monitoring
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        print(f"\n💾 System Resources:")
        print(f"   Memory: {memory.percent}% used")
        print(f"   Available: {memory.available // (1024**3)}GB")
        
    except ImportError:
        print("⚠️ psutil not available for resource monitoring")
```

## Security Considerations

### API Security
- **Input Validation**: All request data validated server-side
- **Model Access Control**: Only approved models accessible
- **Resource Limits**: GPU memory and processing time limits enforced
- **Safe Content**: Generated explanations filtered for safety

### Data Security
- **Local Processing**: All explanations generated locally (no external APIs)
- **Data Sovereignty**: Complete control over sensitive feature data
- **Audit Trail**: Full logging of explanation generation process
- **Access Control**: Secure API endpoints with proper authentication

### Deployment Security
```yaml
# Kubernetes security context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  capabilities:
    drop: ["ALL"]
    add: ["NET_BIND_SERVICE"]
```

## Performance Optimization

### GPU Utilization

```python
def optimize_gpu_usage(feature_count, available_models):
    """Optimize GPU usage based on workload."""
    
    # Model selection based on GPU capacity
    if feature_count <= 5:
        return 'llama3.1:8b'    # RTX 3080 Ti (lighter load)
    elif feature_count <= 15:
        return 'llama3.1:70b'   # RTX 3090 (heavier model)
    else:
        # Batch processing with mixed models
        return 'batch_processing'

def parallel_processing_strategy(features):
    """Strategy for parallel explanation generation."""
    
    # Group features by complexity
    simple_features = []
    complex_features = []
    
    for feature_id, data in features.items():
        if data.get('pattern_category') == 'technical':
            simple_features.append((feature_id, data))
        else:
            complex_features.append((feature_id, data))
    
    return {
        'simple_batch': {
            'features': dict(simple_features),
            'model': 'llama3.1:8b',
            'gpu': 'RTX_3080_Ti'
        },
        'complex_batch': {
            'features': dict(complex_features),
            'model': 'llama3.1:70b', 
            'gpu': 'RTX_3090'
        }
    }
```

### Caching Strategy

```python
import hashlib
import json

def cache_explanation_result(feature_data, explanation_result):
    """Cache explanation results for reuse."""
    
    # Create cache key from feature data
    feature_hash = hashlib.md5(
        json.dumps(feature_data, sort_keys=True).encode()
    ).hexdigest()
    
    cache_key = f"explanation_{feature_hash}"
    
    # Store in cache (implement with Redis, memcached, etc.)
    cache_store[cache_key] = {
        'explanation_result': explanation_result,
        'timestamp': time.time(),
        'ttl': 86400  # 24 hours
    }

def get_cached_explanation(feature_data):
    """Retrieve cached explanation if available."""
    
    feature_hash = hashlib.md5(
        json.dumps(feature_data, sort_keys=True).encode()
    ).hexdigest()
    
    cache_key = f"explanation_{feature_hash}"
    cached = cache_store.get(cache_key)
    
    if cached and time.time() - cached['timestamp'] < cached['ttl']:
        return cached['explanation_result']
    
    return None
```

## Advanced Features

### Custom Explanation Templates

```python
def create_custom_explanation_template(analysis_type, target_audience):
    """Create customized explanation templates."""
    
    templates = {
        'technical_audience': {
            'system_prompt': "You are a senior AI researcher explaining neural network features to technical experts.",
            'explanation_format': "Provide detailed technical analysis including activation patterns, statistical significance, and implementation implications."
        },
        'business_audience': {
            'system_prompt': "You are an AI consultant explaining AI capabilities to business stakeholders.", 
            'explanation_format': "Focus on business impact, risk assessment, and actionable insights in plain language."
        },
        'safety_review': {
            'system_prompt': "You are an AI safety specialist identifying potential risks and concerns.",
            'explanation_format': "Highlight safety implications, potential misuse, and recommended safeguards."
        }
    }
    
    return templates.get(target_audience, templates['business_audience'])
```

### Explanation Quality Metrics

```python
def advanced_quality_assessment(explanation_text, feature_data):
    """Advanced quality assessment using multiple metrics."""
    
    metrics = {}
    
    # 1. Coherence with original feature
    pattern_keywords = feature_data.get('pattern_keywords', [])
    keyword_coverage = sum(1 for keyword in pattern_keywords 
                          if keyword.lower() in explanation_text.lower())
    metrics['keyword_alignment'] = keyword_coverage / len(pattern_keywords) if pattern_keywords else 0
    
    # 2. Explanation completeness
    required_elements = ['detects', 'pattern', 'significant', 'indicates']
    completeness = sum(1 for element in required_elements 
                      if element in explanation_text.lower())
    metrics['completeness'] = completeness / len(required_elements)
    
    # 3. Readability (Flesch reading ease)
    import textstat
    metrics['readability'] = textstat.flesch_reading_ease(explanation_text) / 100
    
    # 4. Specificity (avoid generic terms)
    generic_terms = ['thing', 'stuff', 'feature', 'pattern', 'data']
    generic_count = sum(explanation_text.lower().count(term) for term in generic_terms)
    word_count = len(explanation_text.split())
    metrics['specificity'] = 1 - (generic_count / word_count) if word_count else 0
    
    # Overall quality score
    metrics['overall_quality'] = (
        metrics['keyword_alignment'] * 0.3 +
        metrics['completeness'] * 0.3 +
        metrics['readability'] * 0.2 +
        metrics['specificity'] * 0.2
    )
    
    return metrics
```

## Conclusion

This comprehensive API reference provides everything needed to integrate with the miStudioExplain service. The service offers:

- **Local LLM Integration**: Zero external dependencies with Ollama
- **High-Quality Explanations**: AI-powered generation with quality validation
- **Flexible Input Support**: Both miStudioFind results and raw text
- **Multi-Format Output**: JSON, HTML, PDF, CSV, and TXT exports
- **Production-Ready**: Comprehensive error handling and monitoring

### Key Benefits

1. **AI Transparency**: Transform abstract features into human understanding
2. **Local Processing**: Complete data sovereignty with no external API calls
3. **Quality Assurance**: Automated validation and safety assessment
4. **Scalable Architecture**: Efficient batch processing and GPU optimization
5. **Integration Ready**: Seamless pipeline from miStudioFind to miStudioScore

### Next Steps

For production deployment:
1. Set up Ollama service with required models
2. Configure GPU resources and model allocation
3. Implement monitoring and alerting
4. Establish quality thresholds and safety policies
5. Create custom explanation templates for your use case

For technical support and advanced configuration, refer to the interactive API documentation at `/docs` when the service is running.

---

**Service Status**: 📋 Specification Complete - Ready for Development  
**Last Updated**: July 26, 2025  
**API Version**: 1.0.0