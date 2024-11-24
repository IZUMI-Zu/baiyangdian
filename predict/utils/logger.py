import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "predict", 
                log_level: int = logging.INFO,
                log_dir: str = "logs") -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Root logger name (default: 'predict')
        log_level: Logging level (default: INFO)
        log_dir: Directory to store log files (default: 'logs')
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # If logger is already configured, return it
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(log_level)
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler
    today = datetime.now().strftime('%Y-%m-%d')
    file_handler = logging.FileHandler(
        filename=log_path / f"{today}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 