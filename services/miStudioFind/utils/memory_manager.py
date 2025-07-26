# utils/memory_manager.py
"""
Memory management utilities for miStudioFind service.

This module provides efficient memory usage for large-scale feature analysis,
including batch processing and streaming analysis capabilities.
"""

import logging
import torch
import gc
import psutil
from typing import List, Any, Iterator, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages memory optimization for large-scale feature analysis."""

    def __init__(self, max_memory_gb: float = 8.0):
        """
        Initialize MemoryManager.

        Args:
            max_memory_gb: Maximum memory usage in GB
        """
        self.max_memory_gb = max_memory_gb
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_memory_usage(self) -> dict:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory usage information
        """
        # System memory
        memory = psutil.virtual_memory()
        system_memory = {
            "total_gb": memory.total / 1e9,
            "available_gb": memory.available / 1e9,
            "used_gb": memory.used / 1e9,
            "percentage": memory.percent,
        }

        # GPU memory if available
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9

                gpu_memory[f"gpu_{i}"] = {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "total_gb": total,
                    "free_gb": total - allocated,
                    "utilization_pct": (allocated / total) * 100,
                }

        return {
            "system_memory": system_memory,
            "gpu_memory": gpu_memory,
            "max_allowed_gb": self.max_memory_gb,
        }

    def check_memory_available(self, required_gb: float) -> bool:
        """
        Check if sufficient memory is available.

        Args:
            required_gb: Required memory in GB

        Returns:
            True if sufficient memory is available
        """
        memory_info = self.get_memory_usage()
        available_gb = memory_info["system_memory"]["available_gb"]

        return available_gb >= required_gb

    def optimize_tensor_loading(
        self, tensor_path: str, device: str = "cpu"
    ) -> torch.Tensor:
        """
        Load tensor with memory optimization.

        Args:
            tensor_path: Path to tensor file
            device: Target device for tensor

        Returns:
            Loaded tensor
        """
        self.logger.debug(f"Loading tensor from {tensor_path} to {device}")

        # Load to CPU first to avoid GPU memory spikes
        tensor = torch.load(tensor_path, map_location="cpu")

        # Clear any cached memory
        self.clear_memory_efficiently()

        # Move to target device if needed
        if device != "cpu" and torch.cuda.is_available():
            tensor = tensor.to(device)

        return tensor

    def manage_batch_processing(
        self, data: List[Any], batch_size: int, memory_threshold_gb: float = None
    ) -> Iterator[List[Any]]:
        """
        Manage batch processing with dynamic batch sizing based on memory.

        Args:
            data: List of data items to process
            batch_size: Initial batch size
            memory_threshold_gb: Memory threshold for batch size adjustment

        Yields:
            Batches of data items
        """
        memory_threshold_gb = memory_threshold_gb or (self.max_memory_gb * 0.8)
        current_batch_size = batch_size

        for i in range(0, len(data), current_batch_size):
            batch = data[i : i + current_batch_size]

            # Check memory before yielding batch
            memory_info = self.get_memory_usage()
            used_memory = memory_info["system_memory"]["used_gb"]

            if used_memory > memory_threshold_gb and current_batch_size > 1:
                # Reduce batch size if memory usage is high
                current_batch_size = max(1, current_batch_size // 2)
                self.logger.warning(
                    f"Reduced batch size to {current_batch_size} due to memory pressure"
                )

                # Re-batch current data
                batch = data[i : i + current_batch_size]

            yield batch

            # Clear memory after each batch
            self.clear_memory_efficiently()

    def implement_streaming_analysis(
        self, data_source: Iterator[Any], process_func: callable, buffer_size: int = 100
    ) -> Iterator[Any]:
        """
        Implement streaming analysis for memory-efficient processing.

        Args:
            data_source: Iterator providing data items
            process_func: Function to process each data item
            buffer_size: Size of processing buffer

        Yields:
            Processed results
        """
        buffer = []

        for item in data_source:
            buffer.append(item)

            if len(buffer) >= buffer_size:
                # Process buffer
                for buffered_item in buffer:
                    try:
                        result = process_func(buffered_item)
                        yield result
                    except Exception as e:
                        self.logger.error(f"Error processing item: {e}")
                        continue

                # Clear buffer and memory
                buffer.clear()
                self.clear_memory_efficiently()

        # Process remaining items in buffer
        for buffered_item in buffer:
            try:
                result = process_func(buffered_item)
                yield result
            except Exception as e:
                self.logger.error(f"Error processing final item: {e}")
                continue

    def clear_memory_efficiently(self) -> None:
        """Clear memory caches efficiently."""
        # Clear Python garbage collection
        gc.collect()

        # Clear PyTorch GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """
        Context manager for memory monitoring during operations.

        Args:
            operation_name: Name of operation for logging
        """
        # Log memory before operation
        before_memory = self.get_memory_usage()
        self.logger.debug(
            f"Starting {operation_name} - Memory: {before_memory['system_memory']['used_gb']:.1f}GB"
        )

        try:
            yield
        finally:
            # Clear memory and log after operation
            self.clear_memory_efficiently()
            after_memory = self.get_memory_usage()
            self.logger.debug(
                f"Completed {operation_name} - Memory: {after_memory['system_memory']['used_gb']:.1f}GB"
            )

    def estimate_tensor_memory(
        self, shape: tuple, dtype: torch.dtype = torch.float32
    ) -> float:
        """
        Estimate memory requirements for a tensor.

        Args:
            shape: Tensor shape
            dtype: Tensor data type

        Returns:
            Estimated memory in GB
        """
        # Calculate number of elements
        num_elements = 1
        for dim in shape:
            num_elements *= dim

        # Calculate bytes per element based on dtype
        if dtype == torch.float32:
            bytes_per_element = 4
        elif dtype == torch.float16:
            bytes_per_element = 2
        elif dtype == torch.int64:
            bytes_per_element = 8
        elif dtype == torch.int32:
            bytes_per_element = 4
        else:
            bytes_per_element = 4  # Default assumption

        total_bytes = num_elements * bytes_per_element
        return total_bytes / 1e9  # Convert to GB
