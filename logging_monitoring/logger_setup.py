# logging_monitoring/logger_setup.py

import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logger(logger_name: str, log_level: str = "INFO", log_file: str = "brokr.log"):
    """
    Configures a logger to output messages to both the console and a rotating file.

    Args:
        logger_name (str): The name for the logger (typically your module or application name).
        log_level (str): The logging level ('DEBUG', 'INFO', etc.).
        log_file (str): Filename for the log file.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Formatter for log messages.
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler: rotates at midnight, keeps 7 backup files.
    file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent log message propagation to the root logger.
    logger.propagate = False
    return logger

# Standalone test of logger setup.
if __name__ == "__main__":
    logger = setup_logger("BROKR")
    logger.debug("This is a debug message.")
    logger.info("Logger setup is complete.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
