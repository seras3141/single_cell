"""Configuration management for single cell analysis pipeline."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


class Config:
    """Configuration manager for the single cell analysis pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            # Look for config in multiple locations
            possible_paths = [
                Path(__file__).parent.parent.parent / "config" / "config.yaml",
                Path("config/config.yaml"),
                Path("config.yaml")
            ]
            
            config_path = None
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
                    
            if config_path is None:
                raise FileNotFoundError("No configuration file found. Please create config/config.yaml")
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'segmentation.cellpose.model_type')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'segmentation.cellpose.model_type')
            value: New value
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Output path. If None, overwrites original file.
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    @property
    def paths(self) -> Dict[str, str]:
        """Get paths configuration."""
        return self.get('paths', {})
    
    @property
    def segmentation(self) -> Dict[str, Any]:
        """Get segmentation configuration."""
        return self.get('segmentation', {})
    
    @property
    def tracking(self) -> Dict[str, Any]:
        """Get tracking configuration."""
        return self.get('tracking', {})
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('training', {})
    
    def expand_path(self, path: str) -> Path:
        """Expand relative paths based on data_root."""
        path = Path(path)
        if not path.is_absolute():
            data_root = self.get('paths.data_root', 'data/')
            path = Path(data_root) / path
        return path


# Global configuration instance
_config = None

def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config

def set_config(config_path: str) -> None:
    """Set global configuration from file."""
    global _config
    _config = Config(config_path)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config = Config(config_path)
    return config._config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
