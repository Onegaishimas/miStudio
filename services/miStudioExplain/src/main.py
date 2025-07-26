"""
Main entry point for miStudioExplain Service
"""

import asyncio
import logging
from pathlib import Path

from utils.logging import setup_logging
from utils.config import ConfigManager
from api.explain_service import ExplainService
from api.endpoints import app

# Import all core modules
from core.input_manager import InputManager
from core.feature_prioritizer import FeaturePrioritizer
from core.context_builder import ContextBuilder
from core.explanation_generator import ExplanationGenerator
from core.quality_validator import QualityValidator
from core.result_manager import ResultManager

# Import infrastructure modules
from infrastructure.ollama_manager import OllamaManager
from infrastructure.gpu_scheduler import GPUScheduler
from infrastructure.model_cache import ModelCache


async def initialize_service():
    """Initialize the miStudioExplain service"""
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Setup logging
    logger = setup_logging(
        log_level=config.log_level,
        log_file="./logs/mistudio_explain.log"
    )
    
    logger.info("🚀 Initializing miStudioExplain service...")
    
    # Create data directories
    Path(config.data_path).mkdir(parents=True, exist_ok=True)
    Path(config.cache_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize infrastructure components
    logger.info("🔧 Initializing infrastructure components...")
    
    gpu_scheduler = GPUScheduler()
    ollama_manager = OllamaManager(namespace=config.ollama.namespace)
    model_cache = ModelCache(ollama_manager, gpu_scheduler)
    
    # Initialize Ollama connection
    await ollama_manager.initialize()
    
    # Initialize core components
    logger.info("🧠 Initializing core processing components...")
    
    input_manager = InputManager(f"{config.data_path}/input")
    feature_prioritizer = FeaturePrioritizer(config.processing.default_quality_threshold)
    context_builder = ContextBuilder()
    explanation_generator = ExplanationGenerator(ollama_manager, gpu_scheduler)
    quality_validator = QualityValidator()
    result_manager = ResultManager(f"{config.data_path}/output")
    
    # Initialize main service
    explain_service = ExplainService(
        input_manager=input_manager,
        feature_prioritizer=feature_prioritizer,
        context_builder=context_builder,
        explanation_generator=explanation_generator,
        quality_validator=quality_validator,
        result_manager=result_manager
    )
    
    logger.info("✅ miStudioExplain service initialized successfully!")
    
    return explain_service, config


def main():
    """Main entry point"""
    import uvicorn
    
    # Run initialization
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        explain_service, config = loop.run_until_complete(initialize_service())
        
        # Store service instance for API access
        app.state.explain_service = explain_service
        app.state.config = config
        
        # Start the FastAPI server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8002,
            log_level=config.log_level.lower()
        )
        
    except Exception as e:
        logging.error(f"❌ Failed to start service: {e}")
        raise
    finally:
        loop.close()


if __name__ == "__main__":
    main()

