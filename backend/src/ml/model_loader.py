"""
Model loading and quantization utilities.

This module provides functions for loading language models from HuggingFace,
applying quantization, extracting architecture configuration, and estimating
memory requirements.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)

from ..models.model import QuantizationFormat

logger = logging.getLogger(__name__)


# Note: Architecture validation has been replaced with dynamic layer discovery.
# Any transformer model with standard attention + MLP structure is now supported.
# See layer_discovery.py for the dynamic introspection logic.


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class OutOfMemoryError(Exception):
    """Raised when loading fails due to insufficient memory."""
    pass


# validate_architecture() has been removed.
# Architecture validation now happens dynamically during hook registration.
# This allows any transformer model with standard structure to be used.


def extract_architecture_config(config: AutoConfig) -> Dict[str, Any]:
    """
    Extract relevant architecture configuration from HuggingFace config.

    Args:
        config: HuggingFace model configuration

    Returns:
        Dictionary containing architecture details
    """
    arch_config = {
        "model_type": config.model_type,
    }

    # Common fields across architectures
    common_fields = [
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "intermediate_size",
        "max_position_embeddings",
        "vocab_size",
        "num_key_value_heads",  # For GQA/MQA
        "hidden_act",
        "initializer_range",
        "layer_norm_eps",
        "use_cache",
        "tie_word_embeddings",
        "rope_theta",  # For RoPE embeddings
    ]

    for field in common_fields:
        if hasattr(config, field):
            arch_config[field] = getattr(config, field)

    return arch_config


def get_quantization_config(quant_format: QuantizationFormat) -> Optional[BitsAndBytesConfig]:
    """
    Get BitsAndBytes quantization configuration for the specified format.

    Args:
        quant_format: Quantization format enum value

    Returns:
        BitsAndBytesConfig for bitsandbytes quantization, or None for FP32/FP16
    """
    if quant_format == QuantizationFormat.FP32:
        return None  # Load in full precision

    elif quant_format == QuantizationFormat.FP16:
        return None  # Will use torch_dtype=torch.float16

    elif quant_format == QuantizationFormat.Q8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

    elif quant_format == QuantizationFormat.Q4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    elif quant_format == QuantizationFormat.Q2:
        # Q2 is experimental - use 4-bit with aggressive settings
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4",  # More aggressive than nf4
        )

    else:
        raise ValueError(f"Unknown quantization format: {quant_format}")


def estimate_model_memory(
    params_count: int,
    quant_format: QuantizationFormat,
) -> int:
    """
    Estimate memory requirements for a model in bytes.

    Args:
        params_count: Number of model parameters
        quant_format: Quantization format

    Returns:
        Estimated memory requirement in bytes
    """
    # Base memory per parameter based on quantization
    bytes_per_param = {
        QuantizationFormat.FP32: 4,
        QuantizationFormat.FP16: 2,
        QuantizationFormat.Q8: 1,
        QuantizationFormat.Q4: 0.5,
        QuantizationFormat.Q2: 0.25,
    }

    base_memory = params_count * bytes_per_param[quant_format]

    # Add overhead for activations, gradients, optimizer states (roughly 20%)
    overhead = base_memory * 0.2

    return int(base_memory + overhead)


def get_fallback_format(quant_format: QuantizationFormat) -> Optional[QuantizationFormat]:
    """
    Get the next less aggressive quantization format for fallback.

    Args:
        quant_format: Current quantization format

    Returns:
        Next fallback format, or None if no fallback available
    """
    fallback_chain = {
        QuantizationFormat.Q2: QuantizationFormat.Q4,
        QuantizationFormat.Q4: QuantizationFormat.Q8,
        QuantizationFormat.Q8: QuantizationFormat.FP16,
        QuantizationFormat.FP16: QuantizationFormat.FP32,
        QuantizationFormat.FP32: None,
    }

    return fallback_chain.get(quant_format)


def estimate_parameter_count(config) -> Optional[int]:
    """Parameter count from the config alone, before any weight is downloaded.

    Standard decoder arithmetic: embeddings + per-layer attention and MLP. Only
    an ESTIMATE — MoE, tied embeddings and multimodal towers all shift it — so
    the caller treats a shortfall as advisory sizing, not a measurement, and
    `preflight_gpu_capacity` skips entirely when this returns None.

    Reads through a sub-config when the top level has no `hidden_size`: unified
    and multimodal configs keep the text fields nested, which is the same shape
    that broke `vocab_size` (MIS-E2E-083's sibling, reported the same day).
    """
    def _get(name: str) -> Optional[int]:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
        for attr in dir(config):
            if attr.startswith("_"):
                continue
            try:
                child = getattr(config, attr)
            except Exception:
                continue
            nested = getattr(child, name, None)
            if isinstance(nested, int) and nested > 0:
                return nested
        return None

    hidden = _get("hidden_size")
    layers = _get("num_hidden_layers")
    vocab = _get("vocab_size")
    if not (hidden and layers and vocab):
        return None

    intermediate = _get("intermediate_size") or 4 * hidden
    # Attention: q,k,v,o ≈ 4·h². MLP: gate/up/down ≈ 3·h·i (2 for non-gated,
    # rounded up deliberately — under-estimating defeats the point).
    per_layer = 4 * hidden * hidden + 3 * hidden * intermediate
    embeddings = 2 * vocab * hidden          # input + output, untied worst case
    return int(layers * per_layer + embeddings)


def load_model_from_hf(
    repo_id: str,
    quant_format: QuantizationFormat = QuantizationFormat.FP16,
    cache_dir: Optional[Path] = None,
    device_map: str = "auto",
    trust_remote_code: bool = False,
    hf_token: Optional[str] = None,
    auto_fallback: bool = True,
    local_files_only: bool = False,
    attn_implementation: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, AutoConfig, Dict[str, Any]]:
    """
    Load a language model from HuggingFace Hub with specified quantization.

    Args:
        repo_id: HuggingFace model repository ID (e.g., "meta-llama/Llama-2-7b-hf")
        quant_format: Quantization format to apply
        cache_dir: Directory to cache downloaded models
        device_map: Device mapping strategy ("auto", "cpu", "cuda:0", etc.)
        trust_remote_code: Whether to trust remote code execution
        hf_token: HuggingFace API token for gated models
        auto_fallback: Automatically fallback to less aggressive quantization on OOM
        local_files_only: If True, only use locally cached files (no network calls).
        attn_implementation: Force an attention backend, e.g. "eager". Left as
            None the model keeps transformers' own choice, which is SDPA
            wherever supported — so every existing caller's `from_pretrained`
            call is byte-identical to before. Pass "eager" only when the caller
            needs attention PROBABILITIES: SDPA and flash kernels never
            materialise them (`sdpa_attention_forward` returns
            `(attn_output, None)`), and eager is materially slower.
            Use this when the model is already downloaded to avoid HF API validation
            for gated models.

    Returns:
        Tuple of (model, tokenizer, config, metadata dict)

    Raises:
        ModelLoadError: If model loading fails
        OutOfMemoryError: If loading fails due to OOM and auto_fallback is False
        ValueError: If architecture is unsupported
    """
    logger.info(f"Loading model {repo_id} with quantization {quant_format.value} (local_files_only={local_files_only})")

    try:
        # Load configuration first to validate architecture
        config = AutoConfig.from_pretrained(
            repo_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=trust_remote_code,
            token=hf_token,
            local_files_only=local_files_only,
        )

        # Note: Architecture validation removed. Dynamic layer discovery handles this.
        # Any transformer model with standard attention + MLP blocks is now supported.
        logger.info(f"Model architecture: {config.model_type}")

        # Extract architecture configuration
        arch_config = extract_architecture_config(config)

        # WILL THIS FIT? ASK BEFORE SPENDING THE MINUTES (live 2026-08-23).
        #
        # An extraction on gemma-4-12B-it ran 2m47s and died with "CUDA out of
        # memory. Tried to allocate 120.00 MiB. GPU 0 has a total capacity of
        # 23.56 GiB of which 113.06 MiB is free" — the weights had taken the
        # card and the first forward pass had nowhere to go. A 12B model at FP16
        # is ~24 GB of weights on a 23.56 GB card; it was never going to fit,
        # and nothing said so.
        #
        # The check lives HERE rather than at the ten call sites, because a
        # guard added to one caller and not its siblings is this codebase's most
        # repeated defect. The config is already loaded above, so the parameter
        # count is available before a single weight is fetched.
        from ..services.resource_config import preflight_gpu_capacity

        preflight_gpu_capacity(
            params_count=estimate_parameter_count(config),
            quantization=quant_format.value,
            model_name=repo_id,
        )

        # Get quantization configuration
        quantization_config = get_quantization_config(quant_format)

        # Determine torch dtype
        if quant_format == QuantizationFormat.FP16:
            torch_dtype = torch.float16
        elif quant_format in (QuantizationFormat.Q8, QuantizationFormat.Q4, QuantizationFormat.Q2):
            torch_dtype = torch.float16  # bitsandbytes uses fp16 for compute
        else:
            torch_dtype = torch.float32

        # Load model
        try:
            load_kwargs: Dict[str, Any] = dict(
                config=config,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                cache_dir=str(cache_dir) if cache_dir else None,
                trust_remote_code=trust_remote_code,
                token=hf_token,
                local_files_only=local_files_only,
            )
            # INSERTED ONLY WHEN ASKED FOR, so "a default caller's call is
            # unchanged" is a property a test can assert rather than a claim.
            if attn_implementation is not None:
                load_kwargs["attn_implementation"] = attn_implementation
            model = AutoModelForCausalLM.from_pretrained(repo_id, **load_kwargs)

            logger.info(f"Successfully loaded model with {quant_format.value} quantization")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                if auto_fallback:
                    fallback_format = get_fallback_format(quant_format)
                    if fallback_format:
                        logger.warning(
                            f"Out of memory with {quant_format.value}. "
                            f"Falling back to {fallback_format.value}"
                        )
                        # Recursive call with fallback format
                        return load_model_from_hf(
                            repo_id=repo_id,
                            quant_format=fallback_format,
                            cache_dir=cache_dir,
                            device_map=device_map,
                            trust_remote_code=trust_remote_code,
                            hf_token=hf_token,
                            auto_fallback=auto_fallback,
                            local_files_only=local_files_only,
                            # CARRIED THROUGH THE FALLBACK. Without this an OOM
                            # retry silently drops back to SDPA, and the caller
                            # that asked for eager gets a model that can never
                            # produce attention weights — with no error, because
                            # the fallback itself succeeded.
                            attn_implementation=attn_implementation,
                        )
                    else:
                        raise OutOfMemoryError(
                            f"Out of memory loading {repo_id} even with FP32. "
                            "Model may be too large for available hardware."
                        )
                else:
                    raise OutOfMemoryError(
                        f"Out of memory loading {repo_id} with {quant_format.value}. "
                        f"Try a more aggressive quantization format."
                    )
            else:
                raise

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=trust_remote_code,
            token=hf_token,
            local_files_only=local_files_only,
        )

        # Calculate metadata
        params_count = sum(p.numel() for p in model.parameters())
        memory_required = estimate_model_memory(params_count, quant_format)

        metadata = {
            "repo_id": repo_id,
            "quantization": quant_format.value,
            "params_count": params_count,
            "memory_required_bytes": memory_required,
            "architecture": config.model_type,
            "architecture_config": arch_config,
            # THE RESOLVED backend, not the requested one. A caller that needs
            # attention probabilities can assert on this and refuse in seconds,
            # rather than discovering minutes into a GPU job that its request
            # was ignored or lost in an OOM fallback.
            "attn_implementation": getattr(
                model.config, "_attn_implementation", None),
        }

        return model, tokenizer, config, metadata

    except Exception as e:
        if isinstance(e, (OutOfMemoryError, ValueError)):
            raise
        logger.error(f"Failed to load model {repo_id}: {e}")
        raise ModelLoadError(f"Failed to load model {repo_id}: {e}")
