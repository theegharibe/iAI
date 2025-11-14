"""Configuration manager."""

import json
from pathlib import Path


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.default_config = {
            'model': 'phi3:mini',
            'temperature': 0.7,
            'context_length': 2048,  # Reduced for better performance
            'auto_start_ollama': True,
            'ollama_path': '',  # Optional explicit path to ollama executable
            'theme': 'dark'
        }
        self.config = self.load()
    
    def load(self) -> dict:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    config = self.default_config.copy()
                    config.update(loaded)
                    return config
            except:
                pass
        return self.default_config.copy()
    
    def save(self) -> bool:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            return True
        except:
            return False
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value."""
        self.config[key] = value