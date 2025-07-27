# Module-Specific Updates for Memory Optimization

## Files to Modify

### 1. `core/activation_extractor.py` - PRIMARY UPDATES NEEDED

This is the main file causing the memory issues. Here are the specific functions to modify:

#### A. Update `_load_model()` method:
```python
def _load_model(self):
    """Load the specified model with memory optimizations"""
    try:
        logger.info(f"Loading model: {self.model_name}")

        # Prepare authentication
        use_auth_token = self.huggingface_token if self.huggingface_token else None

        # Load tokenizer first
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=use_auth_token,
            trust_remote_code=True,
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # ===== MEMORY OPTIMIZATION ADDITIONS =====
        # Load model with memory optimizations
        logger.info("Loading model with memory optimizations...")
        
        # Use quantization for large models
        from transformers import BitsAndBytesConfig
        
        if "phi-2" in self.model_name.lower() or "phi-4" in self.model_name.lower():
            # Use 4-bit quantization for large models
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                use_auth_token=use_auth_token,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            # Regular loading for smaller models
            self.model = AutoModel.from_pretrained(
                self.model_name,
                use_auth_token=use_auth_token,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map=None,
            ).to(self.device)

        self.model.eval()
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for memory efficiency")
        
        # Clear cache after model loading
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        # ===== END MEMORY OPTIMIZATIONS =====
        
        self.actual_model_name = self.model_name
        self._extract_model_info()
        self._register_hook()

        logger.info(f"✅ Successfully loaded {self.model_name}")

    except Exception as e:
        logger.error(f"Failed to load {self.model_name}: {e}")
        raise RuntimeError(f"Could not load model {self.model_name}: {str(e)}")
```

#### B. Update `extract_activations()` method:
```python
def extract_activations(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
    """Extract activations from texts with memory management"""
    self.activations = []
    all_activations = []

    # Reduce batch size for large models
    if "phi" in self.model_name.lower():
        batch_size = min(batch_size, 1)  # Force batch size 1 for phi models
    
    logger.info(f"Extracting activations from {len(texts)} texts with batch_size={batch_size}...")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        try:
            # ===== MEMORY MANAGEMENT ADDITIONS =====
            # Clear GPU cache before each batch
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Monitor memory usage
            if self.device.type == "cuda" and i % 100 == 0:
                allocated = torch.cuda.memory_allocated(self.device) / 1e9
                reserved = torch.cuda.memory_reserved(self.device) / 1e9
                logger.info(f"Batch {i//batch_size}: GPU memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
            # ===== END MEMORY MANAGEMENT =====

            # Tokenize batch with reduced sequence length for large models
            max_length = self.max_sequence_length
            if "phi" in self.model_name.lower():
                max_length = min(max_length, 256)  # Reduce sequence length for phi models
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            # Forward pass to trigger hooks with mixed precision
            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        _ = self.model(**inputs)
                else:
                    _ = self.model(**inputs)

            # Collect activations from this batch
            if self.activations:
                batch_activations = self.activations[-1]  # Most recent activation
                # Average pool over sequence length to get fixed-size representation
                pooled = torch.mean(batch_activations, dim=1)  # [batch_size, hidden_dim]
                # Move to CPU immediately to free GPU memory
                all_activations.append(pooled.cpu())
                
                # Clear activations list to free GPU memory
                self.activations = []

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"CUDA OOM in batch {i//batch_size}. Clearing cache and continuing...")
                # Clear cache and continue with next batch
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                self.activations = []
                continue
            else:
                logger.error(f"Error processing batch {i//batch_size}: {e}")
                continue

    if not all_activations:
        raise RuntimeError("No activations extracted successfully")

    result = torch.cat(all_activations, dim=0)
    logger.info(f"Extracted activations shape: {result.shape}")
    return result
```

### 2. `core/gpu_manager.py` - MINOR UPDATES

Add memory management utilities:

```python
@staticmethod
def clear_gpu_cache():
    """Clear GPU cache and log memory stats"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            logger.info(f"GPU {i} - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")

@staticmethod
def get_memory_info(device_id: int = 0) -> dict:
    """Get detailed memory information for a GPU"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    props = torch.cuda.get_device_properties(device_id)
    allocated = torch.cuda.memory_allocated(device_id) / 1e9
    reserved = torch.cuda.memory_reserved(device_id) / 1e9
    total = props.total_memory / 1e9
    
    return {
        "total_gb": total,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": total - allocated,
        "utilization_pct": (allocated / total) * 100
    }
```

### 3. `core/training_service.py` - MODERATE UPDATES

#### A. Update the `run_training_job()` method around line 200:

```python
# Add memory clearing before training starts
if gpu_id >= 0:
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    # Clear any residual memory
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    logger.warning("Using CPU - this will be very slow for large models")
```

#### B. Update SAE training loop (around line 300):

```python
# Training loop with memory management
for epoch in range(config.max_epochs):
    epoch_losses = []
    epoch_sparsity = []
    epoch_recon_errors = []

    for batch_idx, (batch_data,) in enumerate(dataloader):
        batch_data = batch_data.to(device)

        try:
            optimizer.zero_grad()
            
            # Use mixed precision for memory efficiency
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    reconstruction, hidden, total_loss = sae(batch_data)
                    recon_error = F.mse_loss(reconstruction, batch_data)
                    sparsity_level = torch.mean(torch.sum(hidden > 0, dim=1).float()) / hidden_dim
            else:
                reconstruction, hidden, total_loss = sae(batch_data)
                recon_error = F.mse_loss(reconstruction, batch_data)
                sparsity_level = torch.mean(torch.sum(hidden > 0, dim=1).float()) / hidden_dim

            total_loss.backward()
            optimizer.step()

            epoch_losses.append(total_loss.item())
            epoch_sparsity.append(sparsity_level.item())
            epoch_recon_errors.append(recon_error.item())
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"Error in training epoch {epoch}: {e}")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
```

### 4. `main.py` - MINOR UPDATES

Add memory management at startup:

```python
# Add after other imports
import os

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add before app initialization
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    logger.info("Cleared GPU cache at startup")
```

### 5. `requirements.txt` - ADD DEPENDENCIES

Add these lines to support quantization:

```
bitsandbytes>=0.41.0
accelerate>=0.20.0
```

## Priority Order for Updates

1. **CRITICAL**: `core/activation_extractor.py` - This is causing the immediate OOM issues
2. **HIGH**: `core/training_service.py` - Prevents training failures after activation extraction
3. **MEDIUM**: `core/gpu_manager.py` - Provides better monitoring and management
4. **LOW**: `main.py` - Environmental optimizations

## Testing Strategy

1. Update `activation_extractor.py` first
2. Test with a small model like `microsoft/DialoGPT-small`
3. Gradually test with larger models
4. Monitor memory usage with the new GPU manager utilities

These specific module updates will address the memory issues you're experiencing while maintaining the existing architecture of your miStudioTrain service.