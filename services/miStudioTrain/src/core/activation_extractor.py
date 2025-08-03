# =============================================================================
# core/activation_extractor.py - Memory Optimized Version with Enhanced Cleanup
# =============================================================================

import torch
import logging
import gc
from typing import Optional, List
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from models.api_models import ModelInfo

logger = logging.getLogger(__name__)


class EnhancedActivationExtractor:
    """Enhanced activation extractor with comprehensive memory management"""

    def __init__(
        self,
        model_name: str,
        layer_number: int,
        device: torch.device,
        huggingface_token: Optional[str] = None,
        max_sequence_length: int = 512,
    ):
        self.model_name = model_name
        self.layer_number = layer_number
        self.device = device
        self.huggingface_token = huggingface_token
        self.max_sequence_length = max_sequence_length

        self.activations = []
        self.hook = None
        self.model_info = None
        self.model = None
        self.tokenizer = None

        # Track cleanup state
        self._is_cleaned_up = False

        # Load the specified model
        self._load_model()

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
            
            # Check if this is a large model that needs quantization
            is_large_model = any(name in self.model_name.lower() for name in ["phi-2", "phi-4", "llama", "mistral"])
            
            if is_large_model and self.device.type == "cuda":
                # Use 4-bit quantization for large models
                try:
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
                    logger.info("Loaded model with 4-bit quantization")
                    
                except Exception as e:
                    logger.warning(f"Quantization failed, falling back to regular loading: {e}")
                    # Fallback to regular loading
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        use_auth_token=use_auth_token,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map=None,
                    ).to(self.device)
            else:
                # Regular loading for smaller models or CPU
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
                allocated = torch.cuda.memory_allocated(self.device) / 1e9
                logger.info(f"Model loaded, GPU memory allocated: {allocated:.1f}GB")
            
            # ===== END MEMORY OPTIMIZATIONS =====
            
            self.actual_model_name = self.model_name

            # Extract model information
            self._extract_model_info()

            # Register hook for activation extraction
            self._register_hook()

            logger.info(f"✅ Successfully loaded {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            # Clean up any partially loaded resources
            self._cleanup_partial_load()
            raise RuntimeError(f"Could not load model {self.model_name}: {str(e)}")

    def _cleanup_partial_load(self):
        """Clean up resources if model loading fails"""
        try:
            if self.model is not None:
                if self.device.type == "cuda":
                    self.model.cpu()
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                
            logger.debug("Cleaned up partial model loading resources")
        except Exception as e:
            logger.warning(f"Error during partial cleanup: {e}")

    def _extract_model_info(self):
        """Extract information about the loaded model"""
        try:
            config = self.model.config

            # Determine architecture
            architecture = "unknown"
            if hasattr(config, "model_type"):
                architecture = config.model_type
            elif "phi" in self.model_name.lower():
                architecture = "phi"
            elif "gpt" in self.model_name.lower():
                architecture = "gpt"

            # Get model dimensions
            hidden_size = getattr(
                config,
                "hidden_size",
                getattr(config, "d_model", getattr(config, "n_embd", "unknown")),
            )

            vocab_size = getattr(config, "vocab_size", len(self.tokenizer))

            # Count layers
            total_layers = self._count_layers()

            self.model_info = ModelInfo(
                model_name=self.model_name,
                actual_model_loaded=self.actual_model_name,
                architecture=architecture,
                total_layers=total_layers,
                selected_layer=min(self.layer_number, total_layers - 1),
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                requires_token=self.huggingface_token is not None,
            )

            logger.info(f"Model info: {architecture} architecture, {total_layers} layers, {hidden_size} hidden size")

        except Exception as e:
            logger.warning(f"Could not extract model info: {e}")
            self.model_info = ModelInfo(
                model_name=self.model_name,
                actual_model_loaded=self.actual_model_name,
                architecture="unknown",
                total_layers=0,
                selected_layer=self.layer_number,
                hidden_size=0,
                vocab_size=0,
                requires_token=self.huggingface_token is not None,
            )

    def _count_layers(self) -> int:
        """Count the number of transformer layers"""
        try:
            # Try different common layer attributes
            layer_attrs = [
                "layers",           # PHI-4 style
                "h",               # GPT style
                "layer",           # BERT style
                "blocks",          # Some other models
                "transformer.h",   # GPT-2 style
                "encoder.layer"    # BERT encoder
            ]

            for attr_path in layer_attrs:
                try:
                    obj = self.model
                    for attr in attr_path.split("."):
                        obj = getattr(obj, attr)

                    if hasattr(obj, "__len__"):
                        return len(obj)
                except AttributeError:
                    continue

            # If no standard layers found, count manually
            layer_count = 0
            for name, module in self.model.named_modules():
                if any(layer_type in name.lower() for layer_type in ["layer", "block", "h."]):
                    layer_count += 1

            return layer_count

        except Exception as e:
            logger.warning(f"Could not count layers: {e}")
            return 12  # Default assumption

    def _register_hook(self):
        """Register forward hook to extract activations"""
        def hook_fn(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]  # Usually the hidden states
            elif hasattr(output, "last_hidden_state"):
                hidden_states = output.last_hidden_state
            else:
                hidden_states = output

            # Store activation but immediately move to CPU to save GPU memory
            self.activations.append(hidden_states.detach().cpu())

        try:
            # Find layers based on model architecture
            layers = None

            # PHI-4 style
            if hasattr(self.model, "layers"):
                layers = self.model.layers
            # GPT style
            elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                layers = self.model.transformer.h
            # Direct h attribute
            elif hasattr(self.model, "h"):
                layers = self.model.h
            # BERT style
            elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
                layers = self.model.encoder.layer
            else:
                raise ValueError(f"Cannot find layers in {self.model_name}")

            # Adjust layer number if out of bounds
            if self.layer_number >= len(layers):
                old_layer = self.layer_number
                self.layer_number = len(layers) - 1
                logger.warning(f"Layer {old_layer} not found, using layer {self.layer_number}")

                # Update model info
                if self.model_info:
                    self.model_info.selected_layer = self.layer_number

            # Register hook
            self.hook = layers[self.layer_number].register_forward_hook(hook_fn)
            logger.info(f"Registered hook on layer {self.layer_number}/{len(layers)}")

        except Exception as e:
            logger.error(f"Failed to register hook: {e}")
            raise ValueError(f"Cannot register hook on layer {self.layer_number}: {str(e)}")

    def extract_activations(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        """Extract activations from texts with enhanced memory management"""
        if self._is_cleaned_up:
            raise RuntimeError("Cannot extract activations after cleanup has been called")
            
        self.activations = []
        all_activations = []

        # Reduce batch size for large models
        if "phi" in self.model_name.lower() or "llama" in self.model_name.lower():
            batch_size = min(batch_size, 1)  # Force batch size 1 for large models
            logger.info(f"Reduced batch size to {batch_size} for large model")
        
        logger.info(f"Extracting activations from {len(texts)} texts with batch_size={batch_size}...")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                # ===== MEMORY MANAGEMENT ADDITIONS =====
                # Clear GPU cache before each batch
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                
                # Monitor memory usage every 10 batches
                if self.device.type == "cuda" and i % (10 * batch_size) == 0:
                    allocated = torch.cuda.memory_allocated(self.device) / 1e9
                    reserved = torch.cuda.memory_reserved(self.device) / 1e9
                    logger.info(f"Batch {i//batch_size}: GPU memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
                # ===== END MEMORY MANAGEMENT =====

                # Tokenize batch with reduced sequence length for large models
                max_length = self.max_sequence_length
                if "phi" in self.model_name.lower() or "llama" in self.model_name.lower():
                    max_length = min(max_length, 256)  # Reduce sequence length for large models
                
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
                    batch_activations = self.activations[-1]  # Most recent activation (already on CPU)
                    # Average pool over sequence length to get fixed-size representation
                    pooled = torch.mean(batch_activations, dim=1)  # [batch_size, hidden_dim]
                    all_activations.append(pooled)
                    
                    # Clear activations list to free memory
                    self.activations = []

                # Clear input tensors from GPU
                del inputs
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

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
            except Exception as e:
                logger.error(f"Unexpected error in batch {i//batch_size}: {e}")
                continue

        if not all_activations:
            raise RuntimeError("No activations extracted successfully")

        result = torch.cat(all_activations, dim=0)
        logger.info(f"Extracted activations shape: {result.shape}")
        
        # Final memory cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return result

    def cleanup(self):
        """Enhanced cleanup method for comprehensive resource management"""
        if self._is_cleaned_up:
            logger.debug("Cleanup already performed, skipping")
            return
            
        logger.info("Starting comprehensive activation extractor cleanup...")
        
        try:
            # Step 1: Remove forward hooks
            if self.hook:
                self.hook.remove()
                self.hook = None
                logger.debug("Removed forward hooks")
            
            # Step 2: Clear activation cache
            if self.activations:
                del self.activations
                self.activations = []
                logger.debug("Cleared activation cache")
            
            # Step 3: Move model to CPU and delete
            if self.model is not None:
                try:
                    if self.device.type == "cuda":
                        logger.debug(f"Moving model from {self.device} to CPU")
                        self.model.cpu()
                    del self.model
                    self.model = None
                    logger.debug("Model deleted")
                except Exception as e:
                    logger.warning(f"Error cleaning up model: {e}")
            
            # Step 4: Clean up tokenizer
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                logger.debug("Tokenizer deleted")
            
            # Step 5: Python garbage collection
            collected = gc.collect()
            logger.debug(f"Garbage collection freed {collected} objects")
            
            # Step 6: Final GPU cache clear
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("GPU cache cleared and synchronized")
            
            # Step 7: Mark as cleaned up
            self._is_cleaned_up = True
            
            logger.info("Activation extractor cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during activation extractor cleanup: {e}")
            # Still mark as cleaned up to prevent repeated attempts
            self._is_cleaned_up = True

    def cleanup_gpu_resources(self):
        """Alias for cleanup method for consistency with other classes"""
        self.cleanup()

    def get_model_info(self) -> ModelInfo:
        """Get model information"""
        if self._is_cleaned_up:
            logger.warning("Model info requested after cleanup - returning cached info")
        return self.model_info

    def __del__(self):
        """Destructor to ensure cleanup when object is garbage collected"""
        try:
            if not self._is_cleaned_up:
                self.cleanup()
        except Exception:
            # Ignore errors during destruction
            pass