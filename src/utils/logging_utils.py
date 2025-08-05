import logging
import sys, os

def setup_logging(level: str = "INFO", log_file = None) -> None:
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format
        )


