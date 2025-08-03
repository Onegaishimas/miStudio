---

# main.py - FastAPI Application Entry Point

## Imports and Dependencies

### External Dependencies

* **FastAPI Framework**: Core web framework for API endpoints
* **uvicorn**: ASGI server for running the application
* **torch**: PyTorch for ML operations and GPU management
* **transformers**: HuggingFace transformers for model validation
* **aiofiles**: Async file operations
* **requests**: HTTP client for external API calls

### Internal Dependencies

* `utils.logging\_config`: Logging configuration
* `models.api\_models`: Pydantic models for API request/response
* `core.training\_service`: Main training orchestration service
* `core.gpu\_manager`: GPU resource management utilities

## Application Lifecycle Management

### Lifespan Context Manager

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
```

**Purpose**: Manages application startup and shutdown procedures
**Startup Actions**:

* Clears GPU cache to start with clean memory state
* Logs initial GPU status for all available devices
* Sets up memory optimization environment variables

**Shutdown Actions**:

* Clears GPU cache for clean shutdown
* Logs shutdown completion

### Memory Optimization Setup

```python
os.environ\['PYTORCH\_CUDA\_ALLOC_CONF'] = 'expandable_segments:True'
```

**Purpose**: Configures PyTorch memory allocation for better GPU memory management

## API Endpoints Analysis

### Health and Status Endpoints

#### GET /health

**Handler**: `health\_check()`
**Purpose**: Comprehensive health check with GPU memory information
**Returns**: Service status, GPU information, supported models, and feature flags
**Key Features**:

* Lists all available GPUs with memory status
* Indicates which GPUs are suitable for large models (Phi-4)
* Shows memory optimization capabilities

#### GET /gpu/status

**Handler**: `gpu\_status()`
**Purpose**: Detailed GPU status and memory information
**Returns**: CUDA availability, GPU count, memory usage per GPU
**Use Case**: Monitoring GPU resources before training

#### GET /gpu/clear-cache

**Handler**: `clear\_gpu\_cache()`
**Purpose**: Manual GPU memory cache clearing
**Returns**: Success/failure status
**Use Case**: Memory management during development/debugging

### Training Workflow Endpoints

#### POST /api/v1/train

**Handler**: `start\_training(request: TrainingRequest, background\_tasks: BackgroundTasks)`
**Purpose**: Initiates SAE training job with memory optimization
**Request Model**: `TrainingRequest` (Pydantic model)
**Process Flow**:

1. Validates corpus file exists in `/data/samples/`
2. Checks GPU memory requirements for specified model
3. Creates training job with unique job ID
4. Adds training execution to background tasks
5. Returns job ID and status

**Memory Optimization Features**:

* Pre-validates memory requirements before starting
* Applies model-specific optimizations (batch size, sequence length)
* Uses best available GPU automatically

#### GET /api/v1/train/{job\_id}/status

**Handler**: `get\_training\_status(job\_id: str)`
**Purpose**: Retrieves current training job status
**Returns**: `TrainingStatus` with progress, current epoch, loss, and model info

#### GET /api/v1/train/{job\_id}/result

**Handler**: `get\_training\_result(job\_id: str)`
**Purpose**: Retrieves completed training job results
**Returns**: `TrainingResult` with model paths, statistics, and feature information

#### GET /api/v1/jobs

**Handler**: `list\_jobs()`
**Purpose**: Lists all training jobs with model information
**Returns**: Array of job summaries with status and metadata

### Model Validation Endpoints

#### GET /api/v1/check-memory/{model\_name}

**Handler**: `check\_memory\_requirements(model\_name: str)`
**Purpose**: Validates GPU memory requirements for specific model
**Returns**: Memory check results, optimization recommendations, GPU status
**Key Features**:

* Determines if model is "large" (Phi-4, Llama, etc.)
* Recommends batch size and sequence length
* Suggests quantization and gradient checkpointing

#### POST /api/v1/validate-model

**Handler**: `validate\_model(model\_name: str, huggingface\_token: Optional\[str])`
**Purpose**: Validates model accessibility and extracts metadata
**Process**:

1. Loads tokenizer to verify access
2. Loads model config to extract architecture info
3. Checks memory requirements
4. Returns comprehensive model information

### File Management Endpoints

#### POST /api/v1/upload

**Handler**: `upload\_corpus\_file(file: UploadFile)`
**Purpose**: Uploads training corpus files
**Supported Formats**: .txt, .csv, .json
**Storage Location**: `/data/samples/`
**Returns**: File metadata and upload confirmation

#### GET /api/v1/files

**Handler**: `list\_corpus\_files()`
**Purpose**: Lists available corpus files in samples directory
**Returns**: File listing with metadata (size, modification date)

#### DELETE /api/v1/files/{filename}

**Handler**: `delete\_corpus\_file(filename: str)`
**Purpose**: Deletes corpus files
**Security**: Only operates within samples directory

## Application Configuration

### FastAPI App Configuration

```python
app = FastAPI(
    title="miStudioTrain API - Memory Optimized",
    description="Sparse Autoencoder Training Service with Memory Optimization and Dynamic Model Loading",
    version="1.2.0",
    docs\_url="/docs",
    redoc\_url="/redoc",
    lifespan=lifespan,
)
```

### Service Initialization

```python
data\_path = os.getenv("DATA\_PATH", "/data")
train\_service = MiStudioTrainService(data\_path)
```

**Data Path**: Configurable via environment variable, defaults to `/data`
**Service Instance**: Single global instance of training service

## Error Handling Patterns

### HTTP Exception Handling

* **400 Bad Request**: Invalid parameters, missing files, insufficient memory
* **404 Not Found**: Job not found, file not found
* **500 Internal Server Error**: Training failures, validation errors

### Memory Management Error Handling

* CUDA out-of-memory errors are caught and reported
* GPU cache clearing on errors
* Graceful degradation to CPU when GPU unavailable

## Development Server Configuration

```python
if \_\_name\_\_ == "\_\_main\_\_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log\_level="info", access\_log=True)
```

**Note**: Development configuration uses port 8000, but production uses port 8001 (as configured in dev.sh)



---

# training\_service.py - Core Training Orchestration Service

## Purpose and Architecture

The `MiStudioTrainService` class is the central orchestrator for all SAE training operations. It manages the complete training lifecycle from job creation to result storage, with advanced memory optimization and dynamic model loading capabilities.

## Class: MiStudioTrainService

### Initialization

```python
def \_\_init\_\_(self, base\_data\_path: str = "/data"):
```

**Purpose**: Sets up the training service with configurable data paths
**Key Features**:

* Creates required directory structure
* Initializes job tracking dictionary
* Sets PyTorch CUDA memory optimization environment variables
* Supports configurable data path for different deployment environments

### Directory Structure Management

**Created Directories**:

* `/data/models/{job\_id}/` - Trained SAE models
* `/data/activations/{job\_id}/` - Feature activations for miStudioFind
* `/data/samples/` - Input corpus files

## Job Management System

### Job ID Generation

```python
def generate\_job\_id(self) -> str:
```

**Format**: `train\_{YYYYMMDD}\_{HHMMSS}\_{random\_4\_digits}`
**Purpose**: Creates unique, timestamped job identifiers
**Example**: `train\_20250802\_210454\_2263`

### Job Lifecycle Tracking

```python
self.active\_jobs: Dict\[str, Dict\[str, Any]] = {}
```

**Tracks Per Job**:

* Status: queued, running, completed, failed
* Progress: 0.0 to 1.0 percentage
* Current epoch and loss values
* Start time and estimated completion
* Configuration parameters
* Model information
* Final results

## Training Workflow Implementation

### 1\. Job Initialization

```python
async def start\_training(self, request: TrainingRequest) -> str:
```

**Process**:

1. Generates unique job ID
2. Creates job tracking entry with "queued" status
3. Stores training configuration
4. Returns job ID immediately (non-blocking)

### 2\. Main Training Execution

```python
async def run\_training\_job(self, job\_id: str):
```

**Complete Training Pipeline**:

#### Phase 1: Setup and Validation

* Converts request to TrainConfig
* Validates GPU memory requirements using GPUManager
* Selects optimal GPU device
* Applies model-specific optimizations

#### Phase 2: Model Loading and Activation Extraction

* Loads specified HuggingFace model with memory optimizations
* Extracts activations from specified layer
* Applies quantization for large models (Phi-4, Llama)
* Uses gradient checkpointing for memory efficiency

#### Phase 3: SAE Training

* Initializes Sparse Autoencoder with specified dimensions
* Configures optimizer and mixed precision training
* Implements training loop with memory management
* Applies early stopping based on loss threshold

#### Phase 4: Result Generation and Storage

* Saves trained SAE model with metadata
* Generates feature activations for miStudioFind service
* Computes comprehensive feature statistics
* Creates metadata for service integration

## Memory Optimization Strategies

### Model-Specific Optimizations

```python
model\_opts = GPUManager.optimize\_for\_model(config.model\_name)
```

**Large Model Detection**: Identifies models requiring special handling
**Optimizations Applied**:

* Reduced batch sizes for large models
* Shorter sequence lengths
* 4-bit quantization via BitsAndBytesConfig
* Mixed precision training with GradScaler

### Memory Monitoring and Management

**Throughout Training**:

* Periodic GPU cache clearing (every 50 batches)
* Memory usage logging every 5 epochs
* Out-of-memory error handling with graceful recovery
* Batch-level memory cleanup

### Data Loading Optimization

```python
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch\_size=config.batch\_size,
    shuffle=True,
    num\_workers=0,  # Avoid multiprocessing issues
    pin\_memory=False,  # Save memory
)
```

## Training Algorithm Implementation

### SAE Training Loop

**Key Features**:

* Mixed precision training for large models
* L1 sparsity regularization
* Mean squared error reconstruction loss
* Adaptive batch processing with OOM recovery

### Early Stopping Logic

**Conditions**:

* Loss below minimum threshold
* No improvement for 10 consecutive epochs
* Manual stopping via API

### Progress Tracking

**Real-time Updates**:

* Current epoch and loss
* Training progress percentage
* Estimated time remaining
* Sparsity levels and reconstruction errors

## Result Storage and Integration

### Model Artifacts Saved

```python
torch.save({
    "model\_state\_dict": sae.state\_dict(),
    "config": asdict(config),
    "training\_stats": training\_stats,
    "input\_dim": input\_dim,
    "hidden\_dim": hidden\_dim,
    "sparsity\_coeff": config.sparsity\_coeff,
    "model\_info": model\_info.model\_dump(),
    "memory\_optimizations": model\_opts,
}, model\_path)
```

### Feature Activations for miStudioFind

```python
torch.save({
    "feature\_activations": feature\_activations,
    "original\_activations": original\_activations,
    "texts": texts,
    "feature\_count": hidden\_dim,
    "activation\_dim": input\_dim,
    "model\_info": model\_info.model\_dump(),
}, activations\_path)
```

### Comprehensive Metadata

**JSON Metadata Includes**:

* Job and service information
* Model architecture details
* Training configuration and results
* Feature statistics and analysis
* Integration flags for downstream services

## Status and Result Retrieval

### Status Endpoint Logic

```python
def get\_job\_status(self, job\_id: str) -> TrainingStatus:
```

**Returns**: Real-time training status with safe float conversion for JSON serialization

### Result Endpoint Logic

```python
def get\_job\_result(self, job\_id: str) -> TrainingResult:
```

**Returns**: Complete training results (only for completed jobs)

## Error Handling and Recovery

### Memory Error Handling

* CUDA out-of-memory detection and recovery
* Automatic cache clearing on memory errors
* Graceful batch skipping with continued training
* GPU memory monitoring and alerting

### Job Failure Management

* Detailed error logging and status updates
* GPU cache cleanup on failures
* Proper job status transitions
* Error message propagation to API clients

## Service Integration Features

### miStudioFind Integration

**Prepared Outputs**:

* Feature activations in expected format
* Text-to-activation mappings
* Feature statistics for analysis
* Ready-for-service flags

### Health Monitoring

```python
async def get\_health\_status(self):
```

**Comprehensive Health Check**:

* GPU availability and memory status
* Active job counts
* Service capabilities and supported models
* Memory optimization status

## Performance Characteristics

### Memory Efficiency

* Automatic model-specific optimizations
* Progressive memory management throughout training
* Minimal memory footprint for large models

### Scalability Features

* Background job processing
* Multiple concurrent job support (limited by GPU memory)
* Configurable resource allocation

### Monitoring and Observability

* Detailed progress tracking
* Real-time memory usage monitoring
* Comprehensive job history
* Integration-ready metadata generation



---

# gpu\_manager.py - GPU Resource Management System

## Purpose and Architecture

The `GPUManager` class provides comprehensive GPU resource management with intelligent device selection, memory monitoring, and model-specific optimization recommendations. It serves as the central authority for all GPU-related decisions in the training pipeline.

## Class: GPUManager

### Static Method Design Pattern

All methods are static, making GPUManager a utility class that doesn't require instantiation. This design enables easy access from any part of the codebase without state management.

## Core GPU Selection Logic

### Primary GPU Selection Method

```python
@staticmethod
def get\_best\_gpu(prefer\_large\_memory: bool = True, min\_memory\_gb: float = 4.0) -> int:
```

**Selection Algorithm**:

1. **CUDA Availability Check**: Returns -1 if CUDA unavailable
2. **Multi-GPU Evaluation**: Iterates through all available GPUs
3. **Memory Assessment**: Calculates available memory (total - allocated)
4. **Minimum Threshold**: Filters GPUs below minimum memory requirement
5. **Scoring System**: Combines available memory + total memory bonus
6. **Best Candidate**: Returns GPU ID with highest score

**Key Features**:

* **Large Model Support**: 4GB minimum memory requirement by default
* **Dynamic Memory Calculation**: Uses real-time allocation data
* **Preference Weighting**: Favors GPUs with larger total memory
* **Error Resilience**: Handles individual GPU query failures gracefully

### Memory Requirements by Model Type

**Large Models** (Phi-4, Llama, Mistral): 8GB+ recommended
**Medium Models** (Phi-1, GPT-2, BERT): 2-4GB sufficient
**Small Models**: <2GB acceptable

## Memory Management and Monitoring

### Cache Management

```python
@staticmethod
def clear\_gpu\_cache():
```

**Process**:

1. Calls `torch.cuda.empty\_cache()` to free unused memory
2. Logs memory statistics for all GPUs
3. Provides detailed allocated vs reserved memory reporting
4. Handles per-GPU memory query failures gracefully

### Detailed Memory Information

```python
@staticmethod
def get\_memory\_info(device\_id: int = 0) -> dict:
```

**Returns Comprehensive GPU Data**:

* **Device Properties**: Name, compute capability, multiprocessor count
* **Memory Statistics**: Total, allocated, reserved, free memory in GB
* **Utilization**: Percentage of memory currently in use
* **Hardware Details**: CUDA compute capability version

**Memory Calculation Logic**:

```python
total = props.total\_memory / 1e9
allocated = torch.cuda.memory\_allocated(device\_id) / 1e9
reserved = torch.cuda.memory\_reserved(device\_id) / 1e9
free = total - allocated
utilization = (allocated / total) \* 100
```

### Multi-GPU Monitoring

```python
@staticmethod
def monitor\_memory\_usage(device\_id: int = None) -> dict:
```

**Capabilities**:

* **Single GPU**: Detailed analysis of specific device
* **All GPUs**: Comprehensive system-wide memory status
* **Error Handling**: Graceful degradation for inaccessible devices

## Model-Specific Optimization System

### Optimization Recommendation Engine

```python
@staticmethod
def optimize\_for\_model(model\_name: str) -> dict:
```

**Default Recommendations**:

```python
recommendations = {
    "model\_name": model\_name,
    "is\_large\_model": False,
    "recommended\_batch\_size": 8,
    "recommended\_sequence\_length": 512,
    "use\_quantization": False,
    "use\_gradient\_checkpointing": False,
    "min\_gpu\_memory\_gb": 4.0
}
```

### Large Model Detection Logic

**Identified by Keywords**:

* **Phi Models**: "phi-2", "phi-4"
* **Meta Models**: "llama"
* **Mistral Models**: "mistral"
* **OpenAI Models**: "gpt-3", "gpt-4"

**Large Model Optimizations**:

```python
recommendations.update({
    "is\_large\_model": True,
    "recommended\_batch\_size": 1,      # Conservative batch size
    "recommended\_sequence\_length": 256, # Reduced sequence length
    "use\_quantization": True,         # 4-bit quantization
    "use\_gradient\_checkpointing": True, # Memory optimization
    "min\_gpu\_memory\_gb": 8.0         # Higher memory requirement
})
```

### Medium Model Optimizations

**Models**: Phi-1, GPT-2, BERT, DistilBERT
**Settings**:

* Batch size: 16 (higher throughput)
* Memory requirement: 2GB (lower threshold)
* No quantization needed

## Memory Compatibility System

### Pre-Training Memory Validation

```python
@staticmethod
def check\_memory\_for\_model(model\_name: str, device\_id: int = None) -> dict:
```

**Validation Process**:

1. **Requirement Analysis**: Gets model-specific memory requirements
2. **Device Selection**: Uses specified device or finds best available
3. **Memory Comparison**: Compares available vs required memory
4. **Compatibility Report**: Returns detailed compatibility assessment

**Return Format**:

```python
{
    "sufficient": bool,
    "device\_id": int,
    "available\_gb": float,
    "required\_gb": float,
    "recommendations": dict,
    "reason": str  # If insufficient
}
```

### Memory Insufficiency Handling

**Common Scenarios**:

* No CUDA available â†’ CPU fallback recommendation
* Insufficient memory â†’ Lists memory gap and recommendations
* No suitable GPU â†’ Reports best available option

## Error Handling and Resilience

### GPU Query Error Management

**Error Scenarios Handled**:

* Individual GPU property query failures
* CUDA driver communication issues
* Device enumeration problems
* Memory allocation query failures

**Recovery Strategies**:

* Continue with remaining GPUs if one fails
* Provide partial information when possible
* Default to safe fallback values
* Log warnings for debugging

### Graceful Degradation

**When GPU Unavailable**:

* Returns device\_id = -1 for CPU operation
* Provides appropriate error messages
* Maintains API compatibility
* Suggests alternative approaches

## Integration with Training Pipeline

### Training Service Integration Points

1. **Pre-Training Validation**: Memory check before job starts
2. **Device Selection**: Automatic best GPU selection
3. **Runtime Monitoring**: Memory usage tracking during training
4. **Error Recovery**: Memory cleanup on training failures

### Configuration Influence

**Affects Training Parameters**:

* Batch size adjustment based on memory
* Sequence length optimization
* Quantization configuration
* Gradient checkpointing settings

## Performance Monitoring Features

### Real-Time Memory Tracking

**During Training**:

* Periodic memory usage logging
* GPU utilization monitoring
* Memory leak detection
* Performance bottleneck identification

### Optimization Feedback Loop

**Adaptive Recommendations**:

* Model-specific optimizations based on observed patterns
* Memory usage patterns for different model types
* Performance characteristics per GPU type
* Training efficiency metrics

## Hardware Compatibility

### CUDA Compute Capability Support

**Tracked Information**:

* Compute capability version (e.g., "8.6")
* Multiprocessor count
* Hardware generation identification
* Feature support matrix

### Multi-GPU Environment Support

**Capabilities**:

* Heterogeneous GPU support (different models/memory)
* Intelligent load balancing recommendations
* Per-GPU optimization strategies
* Cluster-aware memory management

This GPU management system provides the foundation for efficient and reliable training across different hardware configurations, ensuring optimal resource utilization while preventing memory-related failures.



---

# activation\_extractor.py - Neural Network Activation Extraction System

## Purpose and Architecture

The `EnhancedActivationExtractor` class is responsible for loading transformer models, extracting internal activations from specific layers, and preparing these activations for SAE training. It includes advanced memory optimizations and supports dynamic model loading from HuggingFace Hub.

## Class: EnhancedActivationExtractor

### Initialization Parameters

```python
def \_\_init\_\_(
    self,
    model\_name: str,              # HuggingFace model identifier
    layer\_number: int,            # Target layer for activation extraction
    device: torch.device,         # GPU/CPU device for model
    huggingface\_token: Optional\[str] = None,  # Authentication token
    max\_sequence\_length: int = 512,           # Maximum input sequence length
):
```

### Instance State Management

**Key Attributes**:

* `self.activations`: List storing extracted activations
* `self.hook`: Forward hook handle for activation capture
* `self.model\_info`: Metadata about loaded model
* `self.model`: Loaded transformer model
* `self.tokenizer`: Text tokenizer for the model

## Dynamic Model Loading System

### Model Loading Pipeline

```python
def \_load\_model(self):
```

**Loading Sequence**:

1. **Authentication Setup**: Configures HuggingFace token if provided
2. **Tokenizer Loading**: Loads and configures tokenizer with padding tokens
3. **Model Size Assessment**: Determines if model requires quantization
4. **Memory-Optimized Loading**: Applies appropriate loading strategy
5. **Post-Load Optimization**: Enables gradient checkpointing and clears cache

### Large Model Detection

**Identified Models**:

* Phi-2, Phi-4: Microsoft's Phi model family
* Llama: Meta's Llama models
* Mistral: Mistral AI models

**Detection Logic**:

```python
is\_large\_model = any(name in self.model\_name.lower() 
                    for name in \["phi-2", "phi-4", "llama", "mistral"])
```

### Quantization Strategy for Large Models

```python
quantization\_config = BitsAndBytesConfig(
    load\_in\_4bit=True,                    # 4-bit quantization
    bnb\_4bit\_compute\_dtype=torch.float16, # Computation dtype
    bnb\_4bit\_use\_double\_quant=True,       # Double quantization
    bnb\_4bit\_quant\_type="nf4"            # NormalFloat4 quantization
)
```

**Benefits**:

* Reduces memory usage by ~75%
* Maintains model performance
* Enables large model training on consumer GPUs
* Automatic fallback if quantization fails

### Memory Optimization Features

**Applied Optimizations**:

* **Mixed Precision**: float16 for CUDA, float32 for CPU
* **Gradient Checkpointing**: Trades computation for memory
* **Device Mapping**: Automatic GPU placement
* **Cache Management**: Periodic GPU memory clearing

## Model Information Extraction

### Architecture Detection

```python
def \_extract\_model\_info(self):
```

**Information Extracted**:

* **Model Type**: Architecture family (phi, gpt, bert, etc.)
* **Hidden Size**: Transformer hidden dimension
* **Vocabulary Size**: Tokenizer vocabulary size
* **Layer Count**: Total number of transformer layers
* **Authentication**: Whether model requires HuggingFace token

### Layer Counting Algorithm

```python
def \_count\_layers(self) -> int:
```

**Layer Detection Strategy**:

1. **Standard Attributes**: Checks common layer attribute names

   * `layers` (Phi-4 style)
   * `h` (GPT style)
   * `layer` (BERT style)
   * `blocks` (Alternative naming)
   * `transformer.h` (GPT-2 style)
   * `encoder.layer` (BERT encoder)

2. **Manual Counting**: Iterates through named modules if standard detection fails
3. **Fallback Default**: Returns 12 if all detection methods fail

## Activation Extraction System

### Forward Hook Registration

```python
def \_register\_hook(self):
```

**Hook Function**:

```python
def hook\_fn(module, input, output):
    # Handle different output formats
    if isinstance(output, tuple):
        hidden\_states = output\[0]  # Usually the hidden states
    elif hasattr(output, "last\_hidden\_state"):
        hidden\_states = output.last\_hidden\_state
    else:
        hidden\_states = output

    # Store activation but immediately move to CPU to save GPU memory
    self.activations.append(hidden\_states.detach().cpu())
```

**Key Features**:

* **Format Flexibility**: Handles various transformer output formats
* **Memory Management**: Immediately moves activations to CPU
* **Error Resilience**: Adapts to different model architectures

### Layer Selection and Validation

**Layer Identification Process**:

1. **Architecture-Specific Search**: Uses known patterns for different model families
2. **Bounds Checking**: Ensures layer number is within valid range
3. **Automatic Adjustment**: Falls back to last layer if specified layer doesn't exist
4. **Hook Registration**: Attaches forward hook to target layer

### Batch Processing with Memory Management

```python
def extract\_activations(self, texts: List\[str], batch\_size: int = 8) -> torch.Tensor:
```

**Processing Pipeline**:

1. **Batch Size Optimization**: Reduces batch size for large models
2. **Memory Monitoring**: Tracks GPU memory usage every 10 batches
3. **Tokenization**: Processes text with truncation and padding
4. **Forward Pass**: Executes model with mixed precision if available
5. **Activation Collection**: Gathers and pools activations across sequence length
6. **Memory Cleanup**: Clears intermediate tensors and GPU cache

### Memory Management During Extraction

**Strategies Applied**:

```python
# Clear GPU cache before each batch
if self.device.type == "cuda":
    torch.cuda.empty\_cache()

# Monitor memory usage every 10 batches
if self.device.type == "cuda" and i % (10 \* batch\_size) == 0:
    allocated = torch.cuda.memory\_allocated(self.device) / 1e9
    reserved = torch.cuda.memory\_reserved(self.device) / 1e9
    logger.info(f"Batch {i//batch\_size}: GPU memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
```

**Out-of-Memory Recovery**:

* Catches CUDA OOM exceptions
* Clears cache and continues with next batch
* Logs warnings for failed batches
* Ensures partial results are still usable

## Activation Processing

### Sequence Length Pooling

**Pooling Strategy**:

```python
# Average pool over sequence length to get fixed-size representation
pooled = torch.mean(batch\_activations, dim=1)  # \[batch\_size, hidden\_dim]
```

**Benefits**:

* **Fixed Output Size**: Consistent dimensions regardless of input length
* **Information Preservation**: Maintains semantic content through averaging
* **Memory Efficiency**: Reduces activation storage requirements

### Sequence Length Optimization

**Large Model Adjustments**:

```python
max\_length = self.max\_sequence\_length
if "phi" in self.model\_name.lower() or "llama" in self.model\_name.lower():
    max\_length = min(max\_length, 256)  # Reduce sequence length for large models
```

## Error Handling and Recovery

### Out-of-Memory Management

**Recovery Strategy**:

1. **Detection**: Catches RuntimeError with "out of memory" message
2. **Cache Clearing**: Immediately clears GPU cache
3. **Batch Skipping**: Continues with next batch
4. **Logging**: Records OOM events for analysis
5. **Graceful Degradation**: Returns partial results if some batches succeed

### Model Loading Failures

**Fallback Mechanisms**:

* Quantization failure â†’ Regular loading
* Authentication failure â†’ Clear error messages
* Model not found â†’ Validation error with suggestions
* Architecture incompatibility â†’ Generic model info

## Integration Points

### Training Service Integration

**Data Flow**:

1. Training service creates extractor with model specifications
2. Extractor loads model and validates layer selection
3. Training service provides text corpus for processing
4. Extractor returns processed activations ready for SAE training
5. Training service receives model metadata for result storage

### Resource Cleanup

```python
def cleanup(self):
    """Remove hooks and clean up memory"""
    if self.hook:
        self.hook.remove()
    self.activations = \[]
    
    # Additional memory cleanup
    if self.device.type == "cuda":
        torch.cuda.empty\_cache()
```

**Cleanup Operations**:

* **Hook Removal**: Unregisters forward hooks to prevent memory leaks
* **Activation Clearing**: Frees stored activation tensors
* **GPU Cache Clearing**: Releases unused GPU memory
* **Reference Cleanup**: Removes circular references for garbage collection

### Model Information Access

```python
def get\_model\_info(self) -> ModelInfo:
```

**Returns**: Complete model metadata for integration with downstream services

## Performance Characteristics

### Memory Efficiency Features

* **Immediate CPU Transfer**: Activations moved to CPU as soon as extracted
* **Batch-Level Cleanup**: GPU memory cleared between batches
* **Quantization Support**: 4-bit quantization for large models
* **Mixed Precision**: Automatic dtype optimization based on device

### Scalability Considerations

* **Adaptive Batch Sizing**: Automatically reduces batch size for large models
* **Progressive Processing**: Handles arbitrarily large text corpora
* **Resource Monitoring**: Real-time memory usage tracking
* **Graceful Degradation**: Continues processing even with partial failures

### Error Resilience

* **Partial Success Handling**: Returns results even if some batches fail
* **Architecture Flexibility**: Adapts to different transformer architectures
* **Memory Recovery**: Automatic recovery from out-of-memory conditions
* **Validation Feedback**: Clear error messages for configuration issues



---

# sae.py - Sparse Autoencoder Implementation

## Purpose and Architecture

The `SparseAutoencoder` class implements a memory-optimized sparse autoencoder designed for learning interpretable features from neural network activations. It includes advanced memory management, feature analysis capabilities, and dead neuron prevention mechanisms.

## Class: SparseAutoencoder

### Network Architecture

```python
def \_\_init\_\_(self, input\_dim: int, hidden\_dim: int, sparsity\_coeff: float = 1e-3):
```

**Network Structure**:

* **Encoder**: `input\_dim â†’ hidden\_dim` with bias and ReLU activation
* **Decoder**: `hidden\_dim â†’ input\_dim` with bias (reconstruction layer)
* **Sparsity Regularization**: L1 penalty on hidden activations

**Mathematical Formulation**:

```
h = ReLU(W\_enc \* x + b\_enc)    # Encoding
xÌ‚ = W\_dec \* h + b\_dec          # Decoding
L = MSE(x, xÌ‚) + Î» \* ||h||â‚     # Loss with sparsity
```

### Weight Initialization Strategy

```python
def \_initialize\_weights(self):
```

**Initialization Scheme**:

1. **Encoder Weights**: Xavier uniform initialization for stable gradients
2. **Encoder Bias**: Small negative bias (-0.1) to encourage sparsity
3. **Decoder Bias**: Zero initialization
4. **Weight Tying**: Decoder weights initialized as transpose of encoder weights

**Benefits**:

* **Prevents Dead Neurons**: Negative bias ensures some initial activation
* **Stable Training**: Xavier initialization prevents vanishing/exploding gradients
* **Parameter Efficiency**: Weight tying reduces parameter count

## Forward Pass Implementation

### Core Forward Method

```python
def forward(self, x: torch.Tensor) -> Tuple\[torch.Tensor, torch.Tensor, torch.Tensor]:
```

**Processing Pipeline**:

1. **Encoding**: Apply linear transformation + ReLU activation
2. **Decoding**: Reconstruct input from hidden representation
3. **Loss Computation**: Calculate reconstruction and sparsity losses

**Memory Optimization**:

* **In-place Operations**: Uses in-place ReLU where possible
* **Efficient Loss Calculation**: Combined loss computation
* **Single Forward Pass**: Returns all necessary values together

### Loss Function Design

```python
recon\_loss = F.mse\_loss(reconstruction, x)
sparsity\_loss = self.sparsity\_coeff \* torch.mean(torch.abs(hidden))
total\_loss = recon\_loss + sparsity\_loss
```

**Loss Components**:

* **Reconstruction Loss**: Mean squared error between input and reconstruction
* **Sparsity Loss**: L1 regularization on hidden activations
* **Combined Loss**: Weighted sum with configurable sparsity coefficient

## Feature Analysis and Statistics

### Comprehensive Feature Statistics

```python
def get\_feature\_stats(self, dataloader) -> Dict\[str, Any]:
```

**Statistical Metrics Computed**:

* **Total Features**: Number of hidden dimensions
* **Active Features**: Features with non-zero activations
* **Dead Features**: Features that never activate
* **Activation Frequency**: Percentage of samples where each feature activates
* **Mean Activations**: Average activation strength per feature
* **Maximum Activations**: Peak activation values per feature

**Memory-Efficient Processing**:

```python
# Process in batches with memory management
for batch\_idx, batch in enumerate(dataloader):
    # ... process batch ...
    
    # Clear GPU memory periodically
    if batch\_idx % 50 == 0 and device.type == "cuda":
        torch.cuda.empty\_cache()
```

### Dead Feature Detection and Handling

```python
def get\_dead\_features(self, dataloader, threshold: float = 1e-6) -> torch.Tensor:
```

**Dead Feature Criteria**:

* **Activation Threshold**: Features with activations below 1e-6
* **Frequency Threshold**: Features active in <1% of samples
* **Statistical Analysis**: Comprehensive activation pattern analysis

### Dead Feature Reinitialization

```python
def reinitialize\_dead\_features(self, dataloader, threshold: float = 1e-6):
```

**Reinitialization Strategy**:

1. **Detection**: Identify features with low activation frequency
2. **Weight Reset**: Xavier uniform reinitialization for dead features
3. **Bias Reset**: Restore negative bias for sparsity encouragement
4. **Decoder Update**: Synchronize decoder weights with encoder changes

**Benefits**:

* **Prevents Feature Collapse**: Maintains diverse feature representations
* **Improves Training Efficiency**: Utilizes full network capacity
* **Enhances Interpretability**: Ensures all features contribute meaningfully

## Model Persistence and Checkpointing

### Checkpoint Saving

```python
def save\_checkpoint(self, path: str, optimizer=None, epoch: int = 0, loss: float = 0.0):
```

**Saved Information**:

```python
checkpoint = {
    'epoch': epoch,
    'model\_state\_dict': self.state\_dict(),
    'loss': loss,
    'input\_dim': self.input\_dim,
    'hidden\_dim': self.hidden\_dim,
    'sparsity\_coeff': self.sparsity\_coeff,
    'optimizer\_state\_dict': optimizer.state\_dict()  # if provided
}
```

### Checkpoint Loading

```python
@classmethod
def load\_checkpoint(cls, path: str, device: torch.device = None):
```

**Loading Process**:

1. **Model Reconstruction**: Creates new instance with saved parameters
2. **State Loading**: Loads trained weights and biases
3. **Device Placement**: Moves model to specified device
4. **Return Metadata**: Returns both model and checkpoint information

## Utility Methods for Analysis

### Encoding and Decoding Operations

```python
def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Encode input to sparse features"""
    return F.relu(self.encoder(x))

def decode(self, hidden: torch.Tensor) -> torch.Tensor:
    """Decode hidden features back to input space"""
    return self.decoder(hidden)
```

**Use Cases**:

* **Feature Extraction**: Get sparse representations for analysis
* **Reconstruction Testing**: Evaluate decoder quality independently
* **Visualization**: Generate reconstructions for specific features

### Reconstruction Error Analysis

```python
def compute\_reconstruction\_error(self, dataloader) -> float:
```

**Error Computation**:

* **Dataset-Wide**: Averages reconstruction error across entire dataset
* **Memory Efficient**: Processes in batches with GPU memory management
* **Robust Handling**: Graceful recovery from out-of-memory conditions

**Applications**:

* **Model Quality Assessment**: Measures overall reconstruction fidelity
* **Training Monitoring**: Tracks reconstruction quality over time
* **Hyperparameter Tuning**: Compares different sparsity coefficients

## Memory Management Throughout

### Batch-Level Memory Optimization

**Strategies Applied**:

* **Periodic Cache Clearing**: GPU memory freed every 50 batches
* **Out-of-Memory Recovery**: Graceful handling of CUDA OOM errors
* **CPU Offloading**: Statistical computations moved to CPU when possible
* **Tensor Cleanup**: Explicit deletion of intermediate tensors

### Error Handling for Memory Constraints

```python
try:
    # Process batch
    batch\_data = batch\_data.to(device)
    features = self.encode(batch\_data)
    # ... processing ...
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        torch.cuda.empty\_cache()
        continue  # Skip this batch
    else:
        raise e
```

## Integration with Training Pipeline

### Training Loop Integration

**Key Integration Points**:

1. **Loss Computation**: Returns total loss for backpropagation
2. **Feature Monitoring**: Provides real-time sparsity statistics
3. **Dead Feature Management**: Automatic reinitialization during training
4. **Progress Tracking**: Detailed statistics for monitoring training progress

### Downstream Service Preparation

**Feature Activation Generation**:

* **Sparse Representations**: Encoded features for miStudioFind
* **Statistical Metadata**: Feature statistics for analysis tools
* **Quality Metrics**: Reconstruction error and sparsity levels
* **Integration Flags**: Readiness indicators for downstream services

## Performance Characteristics

### Computational Efficiency

* **Single Forward Pass**: Efficient computation of all required outputs
* **Memory-Optimized Operations**: Minimal GPU memory footprint
* **Batch Processing**: Vectorized operations for improved throughput
* **Adaptive Processing**: Adjusts to available memory constraints

### Scalability Features

* **Large Dataset Support**: Handles datasets too large for memory
* **Progressive Analysis**: Computes statistics incrementally
* **Resource Monitoring**: Tracks memory usage throughout processing
* **Graceful Degradation**: Continues operation with partial failures

This implementation provides a robust foundation for sparse autoencoder training with advanced memory management and comprehensive feature analysis capabilities, designed specifically for the interpretability pipeline in miStudio.



---

# api\_models.py - Pydantic API Models and Validation

## Purpose and Architecture

This module defines the complete API contract for the miStudioTrain service using Pydantic models. It provides comprehensive data validation, type checking, and automatic API documentation generation with advanced validation rules for training parameters.

## Request Models

### TrainingRequest - Primary Training Configuration

```python
class TrainingRequest(BaseModel):
```

**Core Training Parameters**:

* **model\_name**: HuggingFace model identifier with validation
* **corpus\_file**: Input corpus filename in samples/ directory
* **layer\_number**: Target transformer layer (0-48 range)
* **huggingface\_token**: Optional authentication token for private models

**SAE Configuration**:

* **hidden\_dim**: SAE hidden dimension (64-32768 range)
* **sparsity\_coeff**: L1 sparsity coefficient (0-1.0 range)

**Training Hyperparameters**:

* **learning\_rate**: Adam optimizer learning rate (default: 1e-4)
* **batch\_size**: Training batch size (1-128 range)
* **max\_epochs**: Maximum training epochs (default: 50)
* **min\_loss**: Early stopping threshold (default: 0.01)

**Hardware Configuration**:

* **gpu\_id**: Specific GPU selection (auto-select if None)
* **max\_sequence\_length**: Maximum input sequence length (default: 512)

### Comprehensive Validation System

#### Model Name Validation

```python
@validator('model\_name')
def validate\_model\_name(cls, v):
    """Validate model exists on HuggingFace Hub"""
    try:
        url = f"https://huggingface.co/api/models/{v}"
        response = requests.head(url, timeout=10)
        if response.status\_code == 404:
            raise ValueError(f"Model '{v}' not found on HuggingFace Hub")
    except requests.RequestException:
        # Network failure - allow and validate later
        pass
    return v
```

**Validation Features**:

* **Real-time Checking**: Verifies model existence on HuggingFace Hub
* **Network Resilience**: Graceful handling of network failures
* **Error Messages**: Clear feedback for invalid model names

#### Layer Number Validation

```python
@validator('layer\_number')
def validate\_layer\_number(cls, v, values):
    """Validate layer number is reasonable"""
    if v < 0:
        raise ValueError("Layer number must be non-negative")
    if v > 48:  # Most models don't have more than 48 layers
        raise ValueError("Layer number seems too high (max 48)")
    return v
```

#### SAE Architecture Validation

```python
@validator('hidden\_dim')
def validate\_hidden\_dim(cls, v):
    """Validate SAE hidden dimension"""
    if v < 64:
        raise ValueError("Hidden dimension too small (min 64)")
    if v > 32768:
        raise ValueError("Hidden dimension too large (max 32768)")
    return v
```

#### Training Parameter Validation

```python
@validator('batch\_size')
def validate\_batch\_size(cls, v):
    """Validate batch size"""
    if v < 1:
        raise ValueError("Batch size must be at least 1")
    if v > 128:
        raise ValueError("Batch size too large (max 128)")
    return v

@validator('sparsity\_coeff')
def validate\_sparsity\_coeff(cls, v):
    """Validate sparsity coefficient"""
    if v <= 0:
        raise ValueError("Sparsity coefficient must be positive")
    if v > 1.0:
        raise ValueError("Sparsity coefficient should be <= 1.0")
    return v
```

## Response Models

### ModelInfo - Model Metadata Container

```python
class ModelInfo(BaseModel):
    model\_name: str              # Original model name requested
    actual\_model\_loaded: str     # Actual model loaded (may differ)
    architecture: str            # Model architecture type
    total\_layers: int           # Total transformer layers
    selected\_layer: int         # Layer selected for extraction
    hidden\_size: int            # Transformer hidden dimension
    vocab\_size: int             # Tokenizer vocabulary size
    requires\_token: bool        # Whether model needs authentication
```

**Purpose**: Provides comprehensive model metadata for downstream services and API clients.

### TrainingStatus - Real-time Training Progress

```python
class TrainingStatus(BaseModel):
    job\_id: str                                    # Unique job identifier
    status: str                                    # 'queued', 'running', 'completed', 'failed'
    progress: float                                # 0.0 to 1.0 completion percentage
    current\_epoch: int                             # Current training epoch
    current\_loss: float                            # Current training loss
    estimated\_time\_remaining: Optional\[int]        # Seconds until completion
    message: str                                   # Human-readable status message
    model\_info: Optional\[ModelInfo] = None         # Model metadata
    last\_checkpoint: Optional\[str] = None          # Most recent checkpoint path
```

**Use Cases**:

* **Progress Monitoring**: Real-time training progress for UI components
* **Status Polling**: Periodic status checks from client applications
* **Error Reporting**: Detailed error messages for failed training jobs

### TrainingResult - Comprehensive Training Outcome

```python
class TrainingResult(BaseModel):
    job\_id: str                      # Job identifier
    status: str                      # Final job status
    model\_path: str                  # Path to saved SAE model
    activations\_path: str            # Path to feature activations
    metadata\_path: str               # Path to training metadata
    training\_stats: Dict\[str, Any]   # Detailed training statistics
    feature\_count: int               # Number of learned features
    model\_info: ModelInfo            # Complete model information
    ready\_for\_find\_service: bool     # Integration readiness flag
    checkpoints: Optional\[list] = None # Available checkpoint paths
```

**Integration Features**:

* **Service Readiness**: Flags indicating readiness for downstream services
* **Path Information**: Complete file paths for accessing training artifacts
* **Statistics Export**: Comprehensive training metrics for analysis

## Model Integration and Validation Flow

### Request Processing Pipeline

1. **Pydantic Validation**: Automatic field validation using decorators
2. **Type Conversion**: Automatic type coercion and validation
3. **Business Logic Validation**: Custom validators for domain-specific rules
4. **Error Aggregation**: Comprehensive error reporting for multiple validation failures

### Response Generation Pipeline

1. **Data Collection**: Gathering information from training service
2. **Model Population**: Creating Pydantic model instances
3. **Serialization**: Automatic JSON serialization with proper types
4. **API Documentation**: Automatic OpenAPI schema generation

## Advanced Validation Features

### Network-Based Validation

**HuggingFace Model Verification**:

* Real-time model existence checking
* Network timeout handling
* Graceful degradation for offline scenarios
* Clear error messages for invalid models

### Range and Constraint Validation

**Parameter Bounds**:

* Layer numbers: 0-48 (covers most transformer architectures)
* Hidden dimensions: 64-32768 (practical SAE sizes)
* Batch sizes: 1-128 (memory and performance constraints)
* Sparsity coefficients: 0-1.0 (mathematical constraints)

### Cross-Field Validation

**Context-Aware Validation**:

* Layer number validation considering model architecture
* Batch size recommendations based on model size
* Memory requirement validation based on parameters

## Error Handling and User Feedback

### Validation Error Messages

**User-Friendly Errors**:

```python
"Model 'invalid/model' not found on HuggingFace Hub"
"Layer number seems too high (max 48)"
"Hidden dimension too small (min 64)"
"Batch size too large (max 128)"
```

### Progressive Validation

**Multi-Stage Validation**:

1. **Syntax Validation**: Basic type and format checking
2. **Range Validation**: Parameter bounds verification
3. **Availability Validation**: External resource checking
4. **Compatibility Validation**: Cross-parameter consistency

## API Documentation Integration

### Automatic Schema Generation

**OpenAPI Benefits**:

* **Interactive Documentation**: Swagger UI with validation examples
* **Client Generation**: Automatic client library generation
* **Type Safety**: Strong typing for API consumers
* **Validation Preview**: Real-time validation feedback in documentation

### Field Documentation

**Comprehensive Descriptions**:

```python
model\_name: str = Field(description="HuggingFace model name (e.g., 'microsoft/Phi-4')")
layer\_number: int = Field(default=12, ge=0, le=48, description="Layer to extract activations from")
hidden\_dim: int = Field(default=1024, description="SAE hidden dimension")
```

## Type Safety and Development Benefits

### Static Type Checking

**Development Advantages**:

* **IDE Support**: Auto-completion and type hints
* **Error Prevention**: Catch type errors before runtime
* **Refactoring Safety**: Type-safe code modifications
* **Documentation**: Self-documenting code through types

### Runtime Validation

**Production Benefits**:

* **Data Integrity**: Ensures valid data throughout the system
* **Error Prevention**: Catches invalid requests before processing
* **Security**: Input sanitization and validation
* **Debugging**: Clear error messages for invalid data

This comprehensive API model system ensures data integrity, provides excellent developer experience, and maintains clear contracts between the API and its consumers while enabling automatic documentation generation and client library creation.



---

# settings.py - Unified Configuration Management System

## Purpose and Architecture

The configuration system provides a unified approach to managing application settings across different deployment environments. It implements an environment-first strategy where environment variables take precedence, with sensible defaults for development environments.

## Configuration Classes

### TrainConfig - Training-Specific Configuration

```python
@dataclass
class TrainConfig:
```

**Configuration Categories**:

#### Model and Training Parameters

```python
# Required fields for training requests
model\_name: str                    # HuggingFace model identifier
corpus\_file: str                   # Input corpus filename

# Model configuration
layer\_number: int = 12            # Target layer for activation extraction
huggingface\_token: Optional\[str] = None  # Authentication token

# SAE configuration
hidden\_dim: int = 1024            # SAE hidden dimension
sparsity\_coeff: float = 1e-3      # L1 sparsity coefficient

# Training configuration
learning\_rate: float = 1e-4       # Adam optimizer learning rate
batch\_size: int = 16              # Training batch size
max\_epochs: int = 50              # Maximum training epochs
min\_loss: float = 0.01            # Early stopping threshold

# Hardware configuration
gpu\_id: Optional\[int] = None      # Specific GPU selection
max\_sequence\_length: int = 512    # Maximum input sequence length
```

#### Environment-First Data Path Configuration

```python
# Data path configuration - unified approach
data\_path: str = os.getenv("DATA\_PATH", "/data")
```

**Key Principle**: Uses environment variables as the primary configuration source, with defaults for development convenience.

#### Service Metadata

```python
# Service metadata
service\_name: str = "miStudioTrain"
service\_version: str = "v1.2.0"
```

### Post-Initialization Directory Management

```python
def \_\_post\_init\_\_(self):
    """Ensure data path exists and create Path object for convenience"""
    self.data\_path\_obj = Path(self.data\_path)
    self.data\_path\_obj.mkdir(parents=True, exist\_ok=True)
```

**Benefits**:

* **Automatic Setup**: Creates required directories on initialization
* **Path Object Creation**: Provides convenient Path objects for file operations
* **Error Prevention**: Ensures directories exist before use

### Directory Property System

```python
@property
def models\_dir(self) -> Path:
    """Directory where trained SAE models are saved"""
    return self.data\_path\_obj / "models"

@property
def activations\_dir(self) -> Path:
    """Directory where feature activations are saved"""
    return self.data\_path\_obj / "activations"

@property
def samples\_dir(self) -> Path:
    """Directory where input corpus files are stored"""
    return self.data\_path\_obj / "samples"

@property
def cache\_dir(self) -> Path:
    """Directory for temporary/cache files"""
    return self.data\_path\_obj / "cache"

@property
def logs\_dir(self) -> Path:
    """Directory for service logs"""
    return self.data\_path\_obj / "logs" / "train"
```

**Directory Structure Created**:

```
/data/
â”œâ”€â”€ models/          # Trained SAE models
â”œâ”€â”€ activations/     # Feature activations for miStudioFind
â”œâ”€â”€ samples/         # Input corpus files
â”œâ”€â”€ cache/           # Temporary files
â””â”€â”€ logs/
    â””â”€â”€ train/       # Training service logs
```

## Global Service Configuration

### ServiceConfig - Environment-Driven Configuration

```python
class ServiceConfig:
    """Global service configuration - environment-first approach"""
```

#### Environment Variable Integration

```python
def \_\_init\_\_(self):
    # Primary data path - same pattern for all services
    self.data\_path = os.getenv("DATA\_PATH", "/data")
    self.data\_path\_obj = Path(self.data\_path)
    self.data\_path\_obj.mkdir(parents=True, exist\_ok=True)
    
    # API configuration
    self.api\_host = os.getenv("API\_HOST", "0.0.0.0")
    self.api\_port = int(os.getenv("API\_PORT", "8001"))
    self.log\_level = os.getenv("LOG\_LEVEL", "INFO")
    
    # Service metadata
    self.service\_name = "miStudioTrain"
    self.service\_version = "v1.2.0"
    
    # GPU configuration
    self.cuda\_memory\_fraction = float(os.getenv("CUDA\_MEMORY\_FRACTION", "0.9"))
    self.max\_concurrent\_jobs = int(os.getenv("MAX\_CONCURRENT\_JOBS", "2"))
```

**Environment Variables Supported**:

* `DATA\_PATH`: Base directory for all data storage
* `API\_HOST`: Service bind address (default: 0.0.0.0)
* `API\_PORT`: Service port (default: 8001)
* `LOG\_LEVEL`: Logging verbosity (default: INFO)
* `CUDA\_MEMORY\_FRACTION`: GPU memory allocation limit (default: 0.9)
* `MAX\_CONCURRENT\_JOBS`: Maximum simultaneous training jobs (default: 2)

### Automatic Directory Creation

```python
def \_ensure\_directories(self):
    """Ensure all required directories exist"""
    directories = \[
        self.data\_path\_obj / "models",
        self.data\_path\_obj / "activations", 
        self.data\_path\_obj / "samples",
        self.data\_path\_obj / "cache",
        self.data\_path\_obj / "logs" / "train"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist\_ok=True)
```

**Benefits**:

* **Zero-Configuration Setup**: Works out of the box
* **Docker-Friendly**: Handles container initialization
* **Development-Ready**: Creates all necessary directories
* **Production-Safe**: Idempotent directory creation

## Configuration Factory Pattern

### TrainConfig Factory

```python
def create\_train\_config(\*\*kwargs) -> TrainConfig:
    """Factory function to create TrainConfig with unified data path"""
    # Ensure data\_path uses the global config
    if 'data\_path' not in kwargs:
        kwargs\['data\_path'] = config.data\_path
    
    return TrainConfig(\*\*kwargs)
```

**Purpose**:

* **Consistency**: Ensures all TrainConfig instances use the same data path
* **Backwards Compatibility**: Maintains existing API while adding global configuration
* **Flexibility**: Allows override of global settings when needed

### Global Configuration Instance

```python
# Global configuration instance
config = ServiceConfig()
```

**Singleton Pattern Benefits**:

* **Single Source of Truth**: One configuration instance across the application
* **Lazy Loading**: Configuration loaded once on module import
* **Memory Efficiency**: Single configuration object shared across modules

## Configuration Integration Points

### Training Service Integration

**Usage in MiStudioTrainService**:

```python
def \_\_init\_\_(self, base\_data\_path: str = "/data"):
    self.base\_data\_path = Path(base\_data\_path)
    # Uses the same data path as global config
```

### API Endpoint Integration

**Environment Variable Access**:

* Health checks report configuration status
* File upload endpoints use configured directories
* Training jobs inherit global configuration settings

### Container and Deployment Integration

**Docker Environment Variables**:

```dockerfile
ENV DATA\_PATH=/app/data
ENV API\_PORT=8001
ENV LOG\_LEVEL=INFO
ENV CUDA\_MEMORY\_FRACTION=0.9
```

## Development vs Production Configuration

### Development Defaults

```python
# Development-friendly defaults
data\_path = "/data"              # Standard development path
api\_host = "0.0.0.0"            # Accept connections from anywhere
api\_port = 8001                 # Non-privileged port
log\_level = "INFO"              # Verbose logging
cuda\_memory\_fraction = 0.9      # Conservative GPU usage
max\_concurrent\_jobs = 2         # Prevent resource exhaustion
```

### Production Overrides

**Common Production Settings**:

```bash
export DATA\_PATH="/mnt/mistudio/data"
export API\_PORT=8001
export LOG\_LEVEL=WARNING
export CUDA\_MEMORY\_FRACTION=0.95
export MAX\_CONCURRENT\_JOBS=4
```

## Configuration Validation and Error Handling

### Path Validation

**Automatic Validation**:

* Directory creation on startup
* Write permission verification
* Disk space awareness (through logging)

### Type Safety

**Environment Variable Type Conversion**:

```python
self.api\_port = int(os.getenv("API\_PORT", "8001"))
self.cuda\_memory\_fraction = float(os.getenv("CUDA\_MEMORY\_FRACTION", "0.9"))
self.max\_concurrent\_jobs = int(os.getenv("MAX\_CONCURRENT\_JOBS", "2"))
```

**Error Prevention**:

* Type conversion with defaults
* Range validation for critical parameters
* Graceful fallbacks for invalid values

## Legacy Support and Migration

### Backwards Compatibility

**Maintaining Existing APIs**:

* TrainConfig class remains unchanged for existing code
* Factory function provides smooth migration path
* Global configuration is additive, not breaking

### Migration Strategy

**Gradual Adoption**:

1. **Phase 1**: Global configuration coexists with existing patterns
2. **Phase 2**: New code uses global configuration
3. **Phase 3**: Legacy code migrated to use factory functions

This configuration system provides a robust foundation for managing application settings across different deployment scenarios while maintaining backwards compatibility and enabling easy configuration management in containerized environments.



---

# logging\_config.py - Centralized Logging Configuration

## Purpose and Design

The logging configuration module provides a centralized, consistent logging setup for the entire miStudioTrain service. It implements a standardized logging format and manages log levels for both application code and third-party dependencies.

## Core Logging Function

### setup\_logging Function

```python
def setup\_logging(level: str = "INFO"):
    """Configure logging for the application"""
```

**Purpose**: Establishes application-wide logging configuration with consistent formatting and appropriate log levels for different components.

## Logging Configuration Components

### Basic Configuration Setup

```python
logging.basicConfig(
    level=getattr(logging, level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
```

**Configuration Elements**:

* **Level**: Configurable log level (INFO by default)
* **Format**: Structured log message format with timestamp, logger name, level, and message
* **Date Format**: ISO-8601 style timestamp format for consistency

### Log Message Format Analysis

**Format String**: `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`

**Example Output**:

```
2025-08-02 21:04:54 - main - INFO - miStudioTrain API starting up with memory optimizations...
2025-08-02 21:04:55 - core.gpu\_manager - INFO - Selected GPU 0 for training
2025-08-02 21:04:56 - core.training\_service - WARNING - Reduced batch size to 1 for large model
```

**Format Benefits**:

* **Timestamp**: Precise timing for debugging and monitoring
* **Logger Name**: Clear identification of log source (module/class)
* **Level**: Easy filtering by severity
* **Message**: Human-readable description of the event

## Third-Party Library Log Management

### Suppressing Verbose Third-Party Logs

```python
# Set specific loggers
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
```

**Rationale**:

* **transformers**: HuggingFace library produces verbose model loading logs
* **torch**: PyTorch generates numerous internal operation logs
* **urllib3**: HTTP client library logs every request/response

**Benefits**:

* **Noise Reduction**: Focuses logs on application-specific events
* **Performance**: Reduces I/O overhead from excessive logging
* **Clarity**: Makes application logs easier to read and debug

### Log Level Hierarchy

**Application Logging Strategy**:

* **Application Code**: Uses configured level (default INFO)
* **Critical Dependencies**: Reduced to WARNING level
* **Infrastructure Logs**: Minimal logging to reduce noise

## Integration with Service Components

### Usage Throughout the Codebase

**Logger Creation Pattern**:

```python
# In each module
import logging
logger = logging.getLogger(\_\_name\_\_)

# Usage examples
logger.info("Starting SAE training job...")
logger.warning("GPU memory insufficient, falling back to CPU")
logger.error("Training failed: model not found")
```

### Service Startup Integration

**In main.py**:

```python
from utils.logging\_config import setup\_logging
setup\_logging()
```

**Initialization Order**:

1. Logging configuration established first
2. All subsequent imports use configured logging
3. Third-party library log levels properly set

## Configuration Flexibility

### Environment-Driven Log Levels

**Integration with Settings**:

```python
# From settings.py
self.log\_level = os.getenv("LOG\_LEVEL", "INFO")

# Usage in main.py
setup\_logging(config.log\_level)
```

**Supported Log Levels**:

* **DEBUG**: Detailed diagnostic information
* **INFO**: General information about application flow
* **WARNING**: Warning messages about potential issues
* **ERROR**: Error messages for handled exceptions
* **CRITICAL**: Critical errors that may cause application failure

### Production vs Development Logging

**Development Settings**:

```bash
export LOG\_LEVEL=DEBUG  # Verbose logging for development
```

**Production Settings**:

```bash
export LOG\_LEVEL=WARNING  # Reduced logging for performance
```

## Logging Patterns Throughout the Service

### Training Pipeline Logging

**Training Service Examples**:

```python
logger.info(f"Created training job {job\_id} for model {request.model\_name}")
logger.info(f"Job {job\_id}: Processing {len(texts)} text chunks")
logger.warning(f"Job {job\_id}: Early stopping at epoch {epoch+1}")
logger.error(f"Job {job\_id}: Training failed: {str(e)}")
```

### GPU Management Logging

**GPU Manager Examples**:

```python
logger.info(f"Selected GPU {best\_gpu} for training")
logger.warning(f"GPU {i} insufficient for large models ({memory\_available:.1f}GB < {min\_memory\_gb}GB)")
logger.info(f"Cleared GPU cache, memory allocated: {allocated:.1f}GB")
```

### Model Loading Logging

**Activation Extractor Examples**:

```python
logger.info(f"Loading model: {self.model\_name}")
logger.info("Loaded model with 4-bit quantization")
logger.warning(f"Layer {old\_layer} not found, using layer {self.layer\_number}")
logger.error(f"Failed to load {self.model\_name}: {e}")
```

## Performance Considerations

### Logging Overhead Management

**Strategies Applied**:

* **Level-Based Filtering**: Only processes logs at or above configured level
* **Third-Party Suppression**: Reduces I/O from verbose libraries
* **Structured Messages**: Efficient string formatting

### Memory and I/O Impact

**Optimization Features**:

* **No File Logging**: Uses console output for container environments
* **Minimal Formatting**: Simple format string reduces processing overhead
* **Lazy Evaluation**: Log messages only formatted when needed

## Container and Deployment Integration

### Docker Container Logging

**Container-Friendly Features**:

* **Console Output**: All logs go to stdout/stderr for container log capture
* **Structured Format**: Easy parsing by log aggregation systems
* **No File Handles**: Avoids file system dependencies

### Log Aggregation Compatibility

**Integration Points**:

* **Kubernetes**: Logs captured by kubectl logs
* **Docker Compose**: Aggregated through docker-compose logs
* **Cloud Platforms**: Compatible with CloudWatch, Stackdriver, etc.

## Debugging and Troubleshooting Support

### Debug-Level Logging

**When LOG\_LEVEL=DEBUG**:

```python
logger.debug(f"Processing batch {batch\_idx} with {len(batch\_texts)} texts")
logger.debug(f"GPU memory before processing: {allocated:.1f}GB")
logger.debug(f"Feature activation shape: {features.shape}")
```

### Error Context Logging

**Error Handling with Context**:

```python
try:
    # Training operation
except Exception as e:
    logger.error(f"Training failed at epoch {epoch}: {str(e)}", exc\_info=True)
    # exc\_info=True includes stack trace in DEBUG mode
```

## Service Monitoring Integration

### Health Check Logging

**Health Endpoint Logging**:

```python
logger.info("Health check completed successfully")
logger.warning("GPU not available for health check")
```

### Performance Monitoring

**Resource Usage Logging**:

```python
logger.info(f"Training completed in {duration:.1f}s")
logger.info(f"Memory usage peak: {peak\_memory:.1f}GB")
logger.info(f"Features extracted: {feature\_count}")
```

This logging configuration provides a solid foundation for debugging, monitoring, and maintaining the miStudioTrain service across different deployment environments while balancing verbosity with performance considerations.



---

# dev.sh - Development Startup Script

## Purpose and Functionality

The development script provides a convenient way to start the miStudioTrain service in development mode with proper Python path configuration and auto-reloading capabilities.

## Script Analysis

### Script Header and Documentation

```bash
#!/bin/bash
#
# Development startup script for the miStudioTrain service.
#
# This script sets the necessary PYTHONPATH and launches the FastAPI
# application using uvicorn on the designated port.
```

**Purpose Statement**: Clear documentation of script functionality and usage.

### Environment Setup

```bash
echo "Starting miStudioTrain service..."

# Add the 'src' directory to the PYTHONPATH to allow uvicorn to find the 'main' module
export PYTHONPATH=${PYTHONPATH}:$(pwd)/src
```

**Key Features**:

* **Status Message**: Provides user feedback about startup process
* **PYTHONPATH Configuration**: Adds src/ directory to Python module search path
* **Dynamic Path**: Uses `$(pwd)/src` to work from any directory the script is called from

### Service Launch Configuration

```bash
# Launch the Uvicorn server
# --host 0.0.0.0 makes the service accessible from outside the container/machine
# --port 8001 sets the listening port
# --reload enables auto-reloading for development
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

**Uvicorn Parameters Explained**:

#### Module Reference

* **src.main:app**: References the FastAPI app instance in src/main.py
* **Requires PYTHONPATH**: Depends on the PYTHONPATH setup above

#### Network Configuration

* **--host 0.0.0.0**: Binds to all network interfaces

  * Allows access from host machine when running in container
  * Enables remote access for development team
  * Works with Docker port mapping

#### Port Configuration

* **--port 8001**: Designated port for miStudioTrain service

  * Consistent with service architecture (differs from main.py default of 8000)
  * Avoids conflicts with other miStudio services
  * Matches production deployment configuration

#### Development Features

* **--reload**: Automatic server restart on code changes

  * Monitors Python files for modifications
  * Restarts server automatically when changes detected
  * Essential for development workflow efficiency

## Integration with Service Architecture

### Port Consistency

**Service Port Allocation**:

* miStudioTrain: 8001 (this service)
* miStudioFind: 8002
* miStudioExplain: 8003
* miStudioScore: 8004

**Note**: The script uses port 8001 while main.py defaults to 8000, indicating the script is the authoritative development configuration.

### Container Compatibility

**Docker Integration**:

* **0.0.0.0 binding**: Allows container port mapping
* **PYTHONPATH setup**: Works within container environment
* **Port consistency**: Matches docker-compose configurations

### Development Workflow Support

**Developer Experience**:

* **Single Command Startup**: Simple `./scripts/dev.sh` execution
* **Auto-reload**: Immediate feedback on code changes
* **Path Resolution**: Works from any directory location
* **Clear Output**: Status messages for debugging startup issues

## Usage Patterns

### Local Development

```bash
cd ~/app/miStudio/services/miStudioTrain
./scripts/dev.sh
```

### Container Development

```bash
# Inside container
cd /app/miStudioTrain
./scripts/dev.sh
```

### CI/CD Integration

**Pipeline Usage**:

* Development environment setup
* Integration testing launch
* Local service testing

## Error Handling and Robustness

### Path Resolution

**Robust Directory Handling**:

* Uses `$(pwd)` for current directory resolution
* Works regardless of script execution location
* Handles symlinks and relative paths properly

### Dependency Requirements

**Prerequisites**:

* **uvicorn**: Must be installed in Python environment
* **src/main.py**: Must contain FastAPI app instance
* **Python modules**: All dependencies in requirements.txt must be available

### Common Issues and Solutions

**Troubleshooting**:

* **Module not found**: Verify PYTHONPATH and src/ directory structure
* **Port conflicts**: Ensure port 8001 is available
* **Permission errors**: Verify script execute permissions (`chmod +x dev.sh`)

## Development vs Production Differences

### Development Configuration (dev.sh)

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

### Production Configuration (would be)

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8001 --workers 4
```

**Key Differences**:

* **Development**: Single worker with auto-reload
* **Production**: Multiple workers without reload for performance

## Script Maintenance and Evolution

### Version Control Integration

**Git Integration**:

* Executable permissions preserved in Git
* Consistent across team development environments
* Versioned with application code

### Future Enhancements

**Potential Improvements**:

* Environment variable validation
* Health check after startup
* Conditional configuration based on environment
* Integration with docker-compose for multi-service development

This development script provides a simple, reliable way to start the miStudioTrain service in development mode while ensuring proper module resolution and development-friendly features like auto-reload.



---

# requirements.txt - External Dependencies Analysis

## Dependency Categories and Purposes

The requirements.txt file organizes dependencies into logical categories, each serving specific roles in the miStudioTrain service architecture.

## Core ML/AI Dependencies

### PyTorch Ecosystem

```text
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
```

**Purpose**: Core deep learning framework
**Usage in Service**:

* **torch**: Sparse autoencoder implementation, GPU management, tensor operations
* **torchvision**: Not directly used but part of standard PyTorch ecosystem
* **torchaudio**: Not directly used but maintains ecosystem compatibility

**Version Requirements**: PyTorch 2.0+ ensures latest memory optimization features and CUDA compatibility.

### HuggingFace Ecosystem

```text
transformers>=4.30.0
datasets>=2.12.0
```

**Purpose**: Transformer model loading and data processing
**Usage in Service**:

* **transformers**: Model loading, tokenization, activation extraction (AutoModel, AutoTokenizer)
* **datasets**: Potential corpus processing (currently using manual text processing)

**Version Requirements**: Recent versions ensure compatibility with latest models like Phi-4.

## Memory Optimization Dependencies

### Quantization and Acceleration

```text
accelerate>=0.20.0
bitsandbytes>=0.41.0
```

**Purpose**: Advanced memory optimization for large models
**Usage in Service**:

* **accelerate**: Mixed precision training, device mapping, memory optimization
* **bitsandbytes**: 4-bit quantization for large models (BitsAndBytesConfig)

**Critical for**: Phi-4, Llama, and other large model support on consumer GPUs.

## API Framework Dependencies

### Web Framework

```text
fastapi>=0.100.0
uvicorn\[standard]>=0.20.0
```

**Purpose**: REST API implementation and ASGI server
**Usage in Service**:

* **fastapi**: All API endpoints, request/response models, automatic documentation
* **uvicorn**: Development and production ASGI server with WebSocket support

### Data Validation

```text
pydantic>=2.0.0
```

**Purpose**: API request/response validation and serialization
**Usage in Service**:

* **TrainingRequest, TrainingStatus, TrainingResult**: All API models
* **Field validation**: Parameter bounds checking, model existence validation

### File Operations

```text
aiofiles>=23.0.0
```

**Purpose**: Asynchronous file operations
**Usage in Service**:

* **File uploads**: Corpus file upload endpoint
* **Async I/O**: Non-blocking file operations in FastAPI endpoints

## Data Processing Dependencies

### Numerical Computing

```text
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

**Purpose**: Data manipulation and analysis
**Usage in Service**:

* **numpy**: Array operations, statistical computations in SAE
* **pandas**: Potential CSV corpus processing, data analysis
* **scikit-learn**: Statistical utilities, potential clustering/analysis features

**Version Requirements**: Recent versions ensure compatibility with PyTorch tensors and modern data formats.

## Development Tools

### Interactive Development

```text
jupyter>=1.0.0
jupyterlab>=4.0.0
```

**Purpose**: Interactive development and analysis
**Usage**:

* **Research and development**: Prototyping SAE architectures
* **Data exploration**: Analyzing training results and feature statistics
* **Workflow testing**: E2E pipeline development (as seen in workflow notebook)

### Code Quality

```text
pytest>=7.0.0
black>=23.0.0
flake8>=6.0.0
```

**Purpose**: Testing and code formatting
**Usage**:

* **pytest**: Unit testing framework for service components
* **black**: Code formatting for consistent style
* **flake8**: Code linting and style checking

## Logging and Monitoring

### Enhanced Output

```text
rich>=13.0.0
loguru>=0.7.0
```

**Purpose**: Improved logging and console output
**Usage**:

* **rich**: Enhanced console output, progress bars, formatting
* **loguru**: Alternative logging system (though service uses standard logging)

**Note**: Currently using standard Python logging, but rich/loguru available for enhanced output.

## GPU Monitoring

### Hardware Monitoring

```text
nvidia-ml-py3>=7.0.0
gpustat>=1.1.0
```

**Purpose**: GPU resource monitoring and management
**Usage in Service**:

* **nvidia-ml-py3**: Direct NVIDIA GPU monitoring (used in GPUManager)
* **gpustat**: Command-line GPU monitoring tools

**Integration**: GPUManager class uses these for memory monitoring and device selection.

## Experiment Tracking (Optional)

### ML Experiment Platforms

```text
wandb>=0.15.0
tensorboard>=2.13.0
```

**Purpose**: Training monitoring and experiment tracking
**Current Status**: Listed as optional, not currently integrated
**Potential Usage**:

* **wandb**: Cloud-based experiment tracking
* **tensorboard**: Local training visualization

## Container and Deployment

### Containerization

```text
docker>=6.0.0
kubernetes>=27.0.0
```

**Purpose**: Containerization and orchestration
**Usage**:

* **docker**: Python SDK for container operations
* **kubernetes**: Python client for K8s integration

**Note**: Primarily for deployment automation, not core service functionality.

## HTTP and Network

### File Upload Support

```text
python-multipart>=0.0.6
```

**Purpose**: Multipart form data handling
**Usage in Service**:

* **File upload endpoint**: Handles corpus file uploads via FastAPI

### HTTP Clients

```text
httpx>=0.24.0
requests>=2.31.0
```

**Purpose**: HTTP client operations
**Usage in Service**:

* **requests**: HuggingFace model validation, external API calls
* **httpx**: Async HTTP operations (available but not currently used)

## Dependency Management Strategy

### Version Pinning Strategy

**Approach**: Minimum version requirements with compatibility ranges
**Benefits**:

* **Forward Compatibility**: Allows bug fixes and minor improvements
* **Stability**: Ensures minimum required features are available
* **Security**: Enables security patch updates

### Critical Dependencies

**Must-Have for Core Functionality**:

1. **torch**: Core ML operations
2. **transformers**: Model loading
3. **fastapi**: API framework
4. **pydantic**: Data validation
5. **accelerate/bitsandbytes**: Memory optimization

### Optional Dependencies

**Enhancement Features**:

1. **wandb/tensorboard**: Experiment tracking
2. **rich/loguru**: Enhanced output
3. **jupyter**: Development tools
4. **docker/kubernetes**: Deployment tools

## Security and Maintenance

### Security Considerations

**Version Management**:

* Regular dependency updates for security patches
* Monitoring for CVE notifications
* Testing compatibility with newer versions

### Maintenance Strategy

**Update Process**:

1. **Regular Reviews**: Monthly dependency update checks
2. **Security Patches**: Immediate updates for critical security issues
3. **Compatibility Testing**: Validation with core functionality
4. **Gradual Updates**: Incremental version bumps with testing

This comprehensive dependency management ensures the miStudioTrain service has all necessary tools for ML training, API operations, memory optimization, and development workflows while maintaining stability and security.



---

# e2e\_workflow.ipynb - End-to-End Integration Workflow

## Purpose and Scope

The Jupyter notebook demonstrates the complete miStudio interpretability pipeline, showcasing how miStudioTrain integrates with other services in the ecosystem. It serves as both documentation and a functional test of the entire system.

## Workflow Overview

### Complete Pipeline Steps

1. **Health Checks**: Verify all miStudio services are operational
2. **SAE Training**: Train sparse autoencoder on Phi-4 layer 30
3. **Feature Analysis**: Analyze learned features with miStudioFind
4. **Explanation Generation**: Generate human-readable explanations with miStudioExplain
5. **Feature Scoring**: Score feature importance with miStudioScore
6. **Results Export**: Export and analyze complete results

## Service Integration Analysis

### miStudioTrain Integration Points

#### Service Configuration

```python
SERVICE\_PORTS = {
    'train': 8001,    # miStudioTrain (this service)
    'find': 8002,     # miStudioFind
    'explain': 8003,  # miStudioExplain
    'score': 8004     # miStudioScore
}
```

**Inter-Service Dependencies**:

* miStudioTrain provides training results for miStudioFind
* Training metadata flows through the entire pipeline
* Each service consumes outputs from the previous service

### Training Configuration for Production Workflow

```python
train\_request = {
    "model\_name": "microsoft/Phi-4",        # Large model example
    "corpus\_file": "webtext\_corpus.txt",    # Production corpus
    "layer\_number": 30,                     # Deep layer analysis
    "hidden\_dim": 1024,                     # Standard SAE size
    "sparsity\_coeff": 1e-3,                # Sparsity regularization
    "learning\_rate": 1e-4,                  # Conservative learning rate
    "batch\_size": 8,                        # Optimized for Phi-4
    "max\_epochs": 20,                       # Reasonable training time
    "min\_loss": 0.01,                       # Early stopping threshold
    "max\_sequence\_length": 512,             # Standard sequence length
    "gpu\_id": 0                            # Explicit GPU selection
}
```

**Configuration Rationale**:

* **Phi-4 Model**: Tests large model handling capabilities
* **Layer 30**: Deep layer for complex feature extraction
* **Conservative Batch Size**: Accommodates large model memory requirements
* **Production Corpus**: Real-world training data

## miStudioTrain Workflow Integration

### 1\. Training Job Creation

```python
response = requests.post(f"{SERVICE\_URLS\['train']}/api/v1/train", json=train\_request)
```

**Integration Features Demonstrated**:

* **Memory Validation**: Pre-training memory requirement checking
* **Model Optimization**: Automatic optimization application for Phi-4
* **Background Processing**: Asynchronous training job execution
* **Status Tracking**: Real-time progress monitoring

### 2\. Training Progress Monitoring

```python
def wait\_for\_job\_completion(service\_url: str, job\_id: str, service\_name: str, max\_wait\_minutes: int = 120):
```

**Monitoring Capabilities**:

* **Real-time Status**: Progress percentage and current epoch
* **Time Estimation**: Remaining training time calculation
* **Error Handling**: Graceful handling of training failures
* **Timeout Management**: Prevents indefinite waiting

### 3\. Result Retrieval and Validation

```python
result\_response = requests.get(f"{SERVICE\_URLS\['train']}/api/v1/train/{training\_job\_id}/result")
```

**Result Integration**:

* **Model Artifacts**: Paths to trained SAE models
* **Feature Data**: Activations prepared for downstream services
* **Metadata**: Comprehensive training statistics and model information
* **Service Readiness**: Flags indicating readiness for next pipeline stage

## Data Flow Through Pipeline

### Training Output â†’ Find Input

**miStudioTrain Produces**:

```python
# Saved by training service
{
    "feature\_activations": feature\_activations,    # Sparse features
    "original\_activations": original\_activations,  # Original activations
    "texts": texts,                                # Source texts
    "feature\_count": hidden\_dim,                   # Number of features
    "model\_info": model\_info.model\_dump(),        # Model metadata
}
```

**miStudioFind Consumes**:

```python
find\_request = {
    "source\_job\_id": training\_job\_id,              # References training job
    "top\_k": 20,                                   # Top activations to analyze
    "coherence\_threshold": 0.5,                    # Feature quality threshold
    "include\_statistics": True                     # Request detailed stats
}
```

### Integration Metadata Flow

**Training Metadata Propagation**:

* **Model Information**: Architecture, dimensions, layer selection
* **Training Configuration**: SAE parameters, optimization settings
* **Performance Metrics**: Loss convergence, sparsity levels, feature statistics
* **Integration Flags**: Service readiness indicators

## Error Handling and Resilience

### Service Health Validation

```python
def check\_service\_health(service\_name: str, url: str) -> bool:
```

**Health Check Features**:

* **Connectivity Testing**: Verify service accessibility
* **Status Validation**: Confirm service operational status
* **Dependency Verification**: Ensure all required services are available
* **Early Failure Detection**: Prevent pipeline start with missing services

### Training Failure Recovery

**Failure Scenarios Handled**:

* **Memory Exhaustion**: GPU out-of-memory conditions
* **Model Loading Failures**: Invalid model names or authentication issues
* **Training Convergence Issues**: Non-converging loss or poor feature quality
* **Resource Conflicts**: GPU availability or disk space issues

### Pipeline Continuity

**Graceful Degradation**:

* **Partial Results**: Continue pipeline with available data
* **Service Substitution**: Alternative approaches when services unavailable
* **Manual Intervention Points**: Clear failure indicators for human review

## Production Readiness Validation

### Memory Optimization Verification

**Workflow Tests**:

* **Large Model Handling**: Phi-4 training with memory optimizations
* **Resource Monitoring**: GPU memory usage tracking throughout training
* **Optimization Application**: Verification of automatic model-specific optimizations

### Service Integration Testing

**End-to-End Validation**:

* **Data Format Compatibility**: Ensure training outputs match downstream expectations
* **Metadata Consistency**: Verify metadata propagation through pipeline
* **Performance Characteristics**: Monitor training time and resource usage

### Result Quality Assessment

**Quality Metrics**:

* **Feature Diversity**: Analysis of learned feature variety
* **Reconstruction Quality**: Validation of SAE reconstruction fidelity
* **Interpretability Readiness**: Preparation of features for explanation generation

## Development and Debugging Support

### Interactive Development

**Notebook Benefits**:

* **Step-by-Step Execution**: Individual pipeline stage testing
* **Parameter Experimentation**: Easy configuration modification
* **Result Visualization**: Immediate feedback on training outcomes
* **Error Diagnosis**: Clear error reporting and troubleshooting

### Configuration Testing

**Parameter Validation**:

* **Model Compatibility**: Testing different model types and sizes
* **Resource Requirements**: Validation of memory and compute needs
* **Training Convergence**: Experimentation with hyperparameters

This end-to-end workflow serves as both a comprehensive integration test and a practical example of how miStudioTrain fits into the larger miStudio interpretability ecosystem, demonstrating production-ready capabilities and robust error handling.



---

# External Dependencies and Integration Summary

## External Dependencies Analysis

### Third-Party Python Libraries

#### Core ML Framework Dependencies

1. **PyTorch Ecosystem**

   * `torch>=2.0.0` - Core tensor operations, GPU management, neural network primitives
   * `torchvision>=0.15.0` - Computer vision utilities (ecosystem compatibility)
   * `torchaudio>=2.0.0` - Audio processing utilities (ecosystem compatibility)

2. **HuggingFace Ecosystem**

   * `transformers>=4.30.0` - Model loading, tokenization, pre-trained models
   * `datasets>=2.12.0` - Dataset loading and processing utilities

3. **Memory Optimization**

   * `accelerate>=0.20.0` - Mixed precision training, device mapping
   * `bitsandbytes>=0.41.0` - 4-bit quantization for large models

#### Web Framework Dependencies

1. **FastAPI Stack**

   * `fastapi>=0.100.0` - REST API framework, automatic documentation
   * `uvicorn\[standard]>=0.20.0` - ASGI server for production deployment
   * `pydantic>=2.0.0` - Data validation and serialization
   * `aiofiles>=23.0.0` - Asynchronous file operations

2. **HTTP and Network**

   * `requests>=2.31.0` - HTTP client for external API calls
   * `httpx>=0.24.0` - Async HTTP client
   * `python-multipart>=0.0.6` - Multipart form data handling

#### Data Processing Dependencies

1. **Scientific Computing**

   * `numpy>=1.24.0` - Numerical operations, array processing
   * `pandas>=2.0.0` - Data manipulation and analysis
   * `scikit-learn>=1.3.0` - Machine learning utilities

#### Development and Monitoring

1. **Development Tools**

   * `jupyter>=1.0.0` - Interactive development environment
   * `jupyterlab>=4.0.0` - Enhanced Jupyter interface
   * `pytest>=7.0.0` - Testing framework
   * `black>=23.0.0` - Code formatting
   * `flake8>=6.0.0` - Code linting

2. **Monitoring and Logging**

   * `rich>=13.0.0` - Enhanced console output
   * `loguru>=0.7.0` - Advanced logging (available but not used)
   * `nvidia-ml-py3>=7.0.0` - GPU monitoring
   * `gpustat>=1.1.0` - GPU statistics

3. **Experiment Tracking** (Optional)

   * `wandb>=0.15.0` - Cloud experiment tracking
   * `tensorboard>=2.13.0` - Training visualization

4. **Deployment** (Optional)

   * `docker>=6.0.0` - Container management SDK
   * `kubernetes>=27.0.0` - Kubernetes client library

### External Services and APIs

#### HuggingFace Hub Integration

**Purpose**: Model loading and validation
**Usage**:

* Model existence validation via HuggingFace API
* Model and tokenizer downloading
* Authentication for private models

**API Endpoints Used**:

* `https://huggingface.co/api/models/{model\_name}` - Model existence verification

#### File System Dependencies

**Data Storage Requirements**:

* `/data/models/` - Trained SAE model storage
* `/data/activations/` - Feature activations for downstream services
* `/data/samples/` - Input corpus file storage
* `/data/cache/` - Temporary file storage
* `/data/logs/train/` - Service logging

#### GPU Hardware Dependencies

**CUDA Requirements**:

* NVIDIA GPU with CUDA support
* CUDA drivers compatible with PyTorch 2.0+
* Minimum 4GB GPU memory for small models
* 8GB+ GPU memory recommended for large models (Phi-4, Llama)

### Inter-Service Dependencies

#### Downstream Service Integration

1. **miStudioFind** (Port 8002)

   * **Consumes**: Feature activations, training metadata
   * **Expects**: Specific file formats and directory structure
   * **Data Flow**: Training results â†’ Feature analysis

2. **miStudioExplain** (Port 8003)

   * **Indirectly Consumes**: Training metadata via miStudioFind
   * **Data Flow**: Training â†’ Find â†’ Explain



---

# miStudioTrain Complete Architecture Summary

## Service Overview

**miStudioTrain** is a sophisticated microservice designed for training Sparse Autoencoders (SAEs) on transformer model activations. It serves as the foundation of the miStudio interpretability pipeline, providing memory-optimized training capabilities for large language models including Microsoft Phi-4, Meta Llama, and other transformer architectures.

## Core Functionality

### Primary Capabilities

1. **Dynamic Model Loading**: Loads any HuggingFace transformer model with automatic optimization
2. **Activation Extraction**: Extracts internal activations from specific transformer layers
3. **SAE Training**: Trains sparse autoencoders with L1 regularization for interpretable features
4. **Memory Optimization**: Advanced memory management for large models on consumer GPUs
5. **Integration Preparation**: Prepares outputs for downstream miStudio services

### Key Features

* **Memory-Optimized Training**: 4-bit quantization, mixed precision, gradient checkpointing
* **Large Model Support**: Specialized handling for Phi-4, Llama, Mistral models
* **Asynchronous Processing**: Background job execution with real-time progress tracking
* **Comprehensive Validation**: Pre-training validation of models, memory, and parameters
* **Service Integration**: Seamless data flow to miStudioFind, miStudioExplain, miStudioScore

## Technical Architecture

### Application Structure

```
miStudioTrain/
â”œâ”€â”€ src/main.py                    # FastAPI application entry point
â”œâ”€â”€ src/core/
â”‚   â”œâ”€â”€ training\_service.py        # Main training orchestration
â”‚   â”œâ”€â”€ gpu\_manager.py            # GPU resource management
â”‚   â””â”€â”€ activation\_extractor.py   # Model activation extraction
â”œâ”€â”€ src/models/
â”‚   â”œâ”€â”€ api\_models.py             # Pydantic API models
â”‚   â””â”€â”€ sae.py                    # Sparse autoencoder implementation
â”œâ”€â”€ src/config/settings.py        # Configuration management
â”œâ”€â”€ src/utils/logging\_config.py   # Logging setup
â”œâ”€â”€ requirements.txt              # Dependencies
â””â”€â”€ scripts/dev.sh               # Development startup
```

### Data Flow Architecture

1. **Input**: Text corpus + Model specification + Training parameters
2. **Processing**: Model loading â†’ Activation extraction â†’ SAE training
3. **Output**: Trained SAE + Feature activations + Metadata
4. **Integration**: Prepared data for downstream services

### API Architecture

**REST API Endpoints**:

* **Training Control**: `/api/v1/train` (POST), `/api/v1/train/{job\_id}/status` (GET)
* **Model Validation**: `/api/v1/validate-model` (POST), `/api/v1/check-memory/{model\_name}` (GET)
* **File Management**: `/api/v1/upload` (POST), `/api/v1/files` (GET/DELETE)
* **System Health**: `/health` (GET), `/gpu/status` (GET)

## Core Components Deep Dive

### 1\. MiStudioTrainService (training\_service.py)

**Purpose**: Central orchestrator for all training operations
**Key Responsibilities**:

* Job lifecycle management (creation, execution, monitoring)
* Training pipeline orchestration (setup â†’ training â†’ storage)
* Memory optimization strategy implementation
* Integration preparation for downstream services

**Advanced Features**:

* Model-specific optimization recommendations
* Real-time progress tracking with time estimation
* Comprehensive error handling and recovery
* Memory-efficient batch processing

### 2\. GPUManager (gpu\_manager.py)

**Purpose**: Intelligent GPU resource management
**Key Responsibilities**:

* Automatic GPU selection based on memory requirements
* Model-specific optimization recommendations
* Real-time memory monitoring and management
* Hardware compatibility assessment

**Optimization Features**:

* Large model detection and automatic parameter adjustment
* Memory requirement validation before training
* Progressive memory cleanup during training
* Hardware-specific optimization recommendations

### 3\. EnhancedActivationExtractor (activation\_extractor.py)

**Purpose**: Dynamic model loading and activation extraction
**Key Responsibilities**:

* HuggingFace model loading with quantization support
* Layer-specific activation extraction via forward hooks
* Memory-optimized processing of large text corpora
* Model metadata extraction for downstream services

**Advanced Capabilities**:

* Architecture-agnostic layer detection
* 4-bit quantization for large models
* Out-of-memory recovery mechanisms
* Mixed precision processing

### 4\. SparseAutoencoder (sae.py)

**Purpose**: Memory-optimized sparse autoencoder implementation
**Key Responsibilities**:

* Sparse representation learning with L1 regularization
* Dead feature detection and reinitialization
* Comprehensive feature statistics computation
* Model checkpointing and persistence

**Technical Features**:

* Weight initialization strategies for sparsity
* Memory-efficient batch processing
* Progressive feature analysis
* Integration-ready output generation

### 5\. API Models (api\_models.py)

**Purpose**: Comprehensive API contract definition
**Key Responsibilities**:

* Request/response validation with Pydantic
* Real-time model existence validation
* Parameter bounds checking and constraint validation
* Automatic API documentation generation

**Validation Features**:

* HuggingFace Hub model existence verification
* Cross-parameter consistency checking
* Memory requirement pre-validation
* User-friendly error messaging

## Memory Optimization Strategy

### Multi-Level Optimization

1. **Model Level**: 4-bit quantization, mixed precision loading
2. **Training Level**: Gradient checkpointing, batch size optimization
3. **Processing Level**: Progressive memory cleanup, CPU offloading
4. **System Level**: CUDA memory configuration, cache management

### Large Model Support

**Optimization Triggers**:

* Model name detection (Phi-4, Llama, Mistral)
* Memory requirement assessment
* Hardware capability evaluation

**Applied Optimizations**:

* Reduced batch sizes (8 â†’ 1 for large models)
* Shorter sequence lengths (512 â†’ 256)
* 4-bit quantization with BitsAndBytesConfig
* Mixed precision training with automatic scaling

## Integration with miStudio Ecosystem

### Service Mesh Integration

**Port Allocation**:

* miStudioTrain: 8001
* miStudioFind: 8002
* miStudioExplain: 8003
* miStudioScore: 8004

### Data Flow Through Pipeline

1. **Training â†’ Find**: Feature activations + model metadata
2. **Find â†’ Explain**: Interpreted features + statistics
3. **Explain â†’ Score**: Explanations + quality metrics
4. **Score â†’ Analysis**: Scored features + importance rankings

### Output Preparation

**For miStudioFind**:

* Feature activations in PyTorch tensor format
* Original activations for comparison
* Text-to-activation mappings
* Comprehensive model metadata

**Integration Metadata**:

* Service readiness flags
* Version compatibility information
* File path specifications
* Processing statistics

## Configuration and Environment Management

### Environment-First Configuration

**Primary Configuration Sources**:

1. Environment variables (highest priority)
2. Configuration files (fallback)
3. Defaults (development convenience)

**Key Environment Variables**:

* `DATA\_PATH`: Base data directory
* `API\_PORT`: Service port (8001)
* `LOG\_LEVEL`: Logging verbosity
* `CUDA\_MEMORY\_FRACTION`: GPU memory limit
* `MAX\_CONCURRENT\_JOBS`: Concurrent training limit

### Directory Structure Management

**Automatic Creation**:

```
/data/
â”œâ”€â”€ models/          # Trained SAE models
â”œâ”€â”€ activations/     # Feature activations
â”œâ”€â”€ samples/         # Input corpus files
â”œâ”€â”€ cache/           # Temporary files
â””â”€â”€ logs/train/      # Service logs
```

## Development and Deployment

### Development Environment

**Features**:

* Auto-reload development server via dev.sh
* Interactive development with Jupyter notebooks
* Comprehensive logging for debugging
* Local GPU development support

### Production Deployment

**Capabilities**:

* Container-ready with Docker support
* Kubernetes orchestration compatibility
* Horizontal scaling support (with proper resource management)
* Health checking and monitoring integration

### Testing and Validation

**E2E Workflow Testing**:

* Complete pipeline validation via Jupyter notebook
* Integration testing with all miStudio services
* Memory optimization validation
* Production scenario simulation

## Performance Characteristics

### Scalability

* **Concurrent Jobs**: Configurable limit based on resources
* **Memory Efficiency**: Optimized for large models on consumer hardware
* **Processing Speed**: Optimized batch processing with memory management
* **Resource Utilization**: Intelligent GPU selection and memory management

### Reliability

* **Error Recovery**: Graceful handling of memory errors and failures
* **Service Resilience**: Proper cleanup on failures
* **Data Integrity**: Comprehensive validation and consistency checking
* **Monitoring**: Real-time resource and progress monitoring

## Security and Best Practices

### Security Features

* **Input Validation**: Comprehensive parameter and file validation
* **Authentication**: HuggingFace token support for private models
* **Data Isolation**: Proper file system permissions and isolation
* **Network Security**: Internal service communication without external exposure

### Best Practices Implementation

* **Clean Architecture**: Clear separation of concerns
* **Type Safety**: Comprehensive type hints and Pydantic validation
* **Error Handling**: Graceful error recovery and user feedback
* **Documentation**: Automatic API documentation generation
* **Testing**: Comprehensive end-to-end workflow validation

This architecture provides a robust, scalable, and maintainable foundation for sparse autoencoder training within the miStudio interpretability ecosystem, with particular strengths in memory optimization, large model support, and seamless service integration.

