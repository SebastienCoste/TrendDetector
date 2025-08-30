#!/usr/bin/env python3
"""
Trending Content Detection System - Main Application
"""

import asyncio
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.core.config import AppConfig
from src.core.logging import setup_logging
from src.core.model_manager import ModelManager, initialize_model_manager
from src.core.gpu_utils import initialize_gpu
from src.api.v2 import inference, update, stats
from src.api.middleware import LoggingMiddleware

# Global configuration
config: AppConfig = None
model_manager: ModelManager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global config, model_manager

    # Startup
    logging.info("Starting Trending Content Detection System...")

    try:
        # Load configuration
        config_path = Path(f"{get_to_root}config/config.yaml")
        if config_path.exists():
            config = AppConfig.from_yaml(str(config_path))
        else:
            config = AppConfig()
            config.to_yaml(str(config_path))
            logging.info(f"Created default config at {config_path}")

        # Setup logging
        setup_logging(config.logging_config)

        # Initialize GPU
        gpu_manager = initialize_gpu(config.gpu_config)
        logging.info(f"GPU enabled: {gpu_manager.is_gpu_enabled}")

        # Initialize model manager
        model_manager = initialize_model_manager(config)

        # Try to load existing model
        try:
            model_manager.load_latest_model(config.server_config.preload_model)
            logging.info(f"Loaded existing model {config.server_config.preload_model}")
        except Exception as e:
            logging.warning(f"No existing model found: {e}")
            logging.info("Starting with fresh model")

        logging.info("System initialization complete")

    except Exception as e:
        logging.error(f"Startup failed: {e}")
        sys.exit(1)

    yield

    # Shutdown
    logging.info("Shutting down Trending Content Detection System...")

    try:
        # Save current model state
        if model_manager and model_manager.is_model_loaded(config.server_config.preload_model):
            model_manager.save_model(config.server_config.preload_model)
            logging.info(f"Model {config.server_config.preload_model} state saved")
    except Exception as e:
        logging.error(f"Error during shutdown: {e}")

    logging.info("Shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Trending Content Detection System",
    description="Real-time content trend prediction with online learning",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Include API routes
app.include_router(inference.router, prefix="/v2", tags=["inference"])
app.include_router(update.router, prefix="/v2", tags=["update"])
app.include_router(stats.router, prefix="/v2", tags=["statistics"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Trending Content Detection System",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        gpu_info = None
        if model_manager:
            from .core.gpu_utils import get_gpu_manager
            try:
                gpu_manager = get_gpu_manager()
                if gpu_manager.is_gpu_enabled:
                    gpu_info = gpu_manager.get_memory_info()
            except:
                pass

        return {
            "status": "healthy",
            "model_loaded": model_manager.is_model_loaded(config.server_config.preload_model) if model_manager else False,
            "gpu_enabled": gpu_info is not None,
            "version": "1.0.0"
        }
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

get_to_root = "../"

if __name__ == "__main__":
    # Run with uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )