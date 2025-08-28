import logging
import logging.handlers
from pathlib import Path
from .config import LoggingConfig

def setup_logging(config: LoggingConfig):
    """Setup comprehensive logging configuration"""
    
    # Create logs directory
    log_path = Path(config.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.level.upper()))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if config.max_file_size.endswith('MB'):
        max_bytes = int(config.max_file_size[:-2]) * 1024 * 1024
    else:
        max_bytes = 10 * 1024 * 1024  # Default 10MB
    
    file_handler = logging.handlers.RotatingFileHandler(
        config.file_path,
        maxBytes=max_bytes,
        backupCount=config.backup_count
    )
    file_handler.setLevel(getattr(logging, config.level.upper()))
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    
    logging.info("Logging system initialized")