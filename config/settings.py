# config/settings.py

import os
import json
import logging

# Define default configuration values
DEFAULT_CONFIG = {
    "api_base_url": "https://api.example.com",  # Default exchange API URL
    "log_level": "INFO",                        # Default logging level
    "exchanges": ["binance", "kraken"],         # List of exchanges to monitor
    "trading_parameters": {
        "max_trade_amount": 1000,
        "min_trade_amount": 10
    },
    "security": {
        "api_key": None,
        "api_secret": None
    }
}


class Settings:
    """
    Encapsulates the application configuration which can be loaded from a file,
    environment variables, or fall back to default values.
    """
    
    def __init__(self, config_data):
        self.api_base_url = config_data.get("api_base_url", DEFAULT_CONFIG["api_base_url"])
        self.log_level = config_data.get("log_level", DEFAULT_CONFIG["log_level"])
        self.exchanges = config_data.get("exchanges", DEFAULT_CONFIG["exchanges"])
        self.trading_parameters = config_data.get(
            "trading_parameters", DEFAULT_CONFIG["trading_parameters"]
        )
        self.security = config_data.get("security", DEFAULT_CONFIG["security"])

    @classmethod
    def load(cls, config_path="config.json"):
        """
        Loads configuration data from a JSON file, and then applies overrides from environment variables.
        
        Args:
            config_path (str): Path to the configuration file (if available).
        
        Returns:
            Settings: An instance of Settings with the loaded configuration.
        """
        config_data = DEFAULT_CONFIG.copy()
        
        # Load configuration from a file if it exists
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                    config_data.update(file_config)
            except Exception as e:
                logging.error(f"Failed to load configuration file '{config_path}': {e}")
        
        # Override specific settings with environment variables (if provided)
        api_base_url = os.getenv("API_BASE_URL")
        if api_base_url:
            config_data["api_base_url"] = api_base_url
        
        log_level = os.getenv("LOG_LEVEL")
        if log_level:
            config_data["log_level"] = log_level
        
        return cls(config_data)


def setup_logging(log_level: str):
    """
    Configures the logging module with the specified log level and format.
    
    Args:
        log_level (str): The logging level (e.g., 'INFO', 'DEBUG').
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Logging configured at {log_level.upper()} level.")


def initialize_app():
    """
    Bootstraps the application by loading the configuration and setting up logging.
    
    Returns:
        Settings: The loaded configuration settings.
    """
    settings = Settings.load()
    setup_logging(settings.log_level)
    return settings
