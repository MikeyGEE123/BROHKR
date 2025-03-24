import json
import os
import logging
from datetime import datetime

class ConfigManager:
    """Manages application configuration and settings"""
    
    def __init__(self, config_file="config.json"):
        """Initialize the configuration manager"""
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                # Return default configuration
                return self.get_default_config()
        
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            return self.get_default_config()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            return True
        
        except Exception as e:
            logging.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get_default_config(self):
        """Get default configuration"""
        return {
            "general": {
                "theme": "Light",
                "notifications": True,
                "log_level": "INFO",
                "auto_save": True
            },
            "trading": {
                "default_exchange": "OKX",
                "paper_trading": True,
                "initial_balance": 10000,
                "position_size": 0.1,
                "stop_loss": 0.05,
                "take_profit": 0.1
            },
            "api_keys": {
                "okx": {
                    "api_key": "",
                    "api_secret": "",
                    "passphrase": ""
                },
                "binance": {
                    "api_key": "",
                    "api_secret": ""
                },
                "kucoin": {
                    "api_key": "",
                    "api_secret": "",
                    "passphrase": ""
                }
            },
            "strategies": {},
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_setting(self, section, key, default=None):
        """Get a specific setting"""
        try:
            return self.config.get(section, {}).get(key, default)
        
        except Exception as e:
            logging.error(f"Error getting setting {section}.{key}: {str(e)}")
            return default
    
    def set_setting(self, section, key, value):
        """Set a specific setting"""
        try:
            if section not in self.config:
                self.config[section] = {}
            
            self.config[section][key] = value
            self.config["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return True
        
        except Exception as e:
            logging.error(f"Error setting {section}.{key}: {str(e)}")
            return False
    
    def reset_to_default(self):
        """Reset configuration to default"""
        try:
            self.config = self.get_default_config()
            return self.save_config()
        
        except Exception as e:
            logging.error(f"Error resetting configuration: {str(e)}")
            return False