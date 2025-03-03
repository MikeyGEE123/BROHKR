# config/review_config.py

from config.settings import Settings

def review_configuration():
    """
    Loads the configuration and prints out the current settings.
    """
    settings = Settings.load()
    
    print("------ Current Configuration ------")
    print(f"API Base URL       : {settings.api_base_url}")
    print(f"Log Level          : {settings.log_level}")
    print(f"Exchanges          : {settings.exchanges}")
    print(f"Trading Parameters : {settings.trading_parameters}")
    print(f"Security Settings  : {settings.security}")
    print("-----------------------------------")

if __name__ == "__main__":
    review_configuration()
