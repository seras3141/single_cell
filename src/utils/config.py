"""Configuration management for single cell analysis pipeline using OmegaConf."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os
import logging

from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ValidationError

from .config_schemas import (
    PipelineConfig, SegmentationConfig, TrainingConfig, TrackingConfig,
    FilterConfig, PostprocessingConfig, validate_pipeline_config
)

logger = logging.getLogger(__name__)


class ConfigManager:
    """OmegaConf-based configuration manager for the single cell analysis pipeline."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = self._find_config_path(config_path)
        self._config = self._load_config()
    
    def _find_config_path(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Find configuration file in standard locations."""
        if config_path is not None:
            path = Path(config_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Look for config in multiple locations
        possible_paths = [
            Path(__file__).parent.parent.parent / "config" / "config.yaml",
            Path("config/config.yaml"),
            Path("config.yaml")
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        raise FileNotFoundError("No configuration file found. Please create config/config.yaml")
    
    def _load_config(self) -> DictConfig:
        """Load configuration with schema validation."""
        try:
            # Load YAML configuration
            config = OmegaConf.load(self.config_path)
            
            # Create structured config from schema
            schema = OmegaConf.structured(PipelineConfig)
            
            # Merge loaded config with schema (validates and fills defaults)
            merged_config = OmegaConf.merge(schema, config)
            
            # Validate the configuration
            validate_pipeline_config(OmegaConf.to_object(merged_config))
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return merged_config
            
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'segmentation.cellpose.model_type')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self._config, key, default=default)
        except Exception:
            return default
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'segmentation.cellpose.model_type')
            value: New value
        """
        OmegaConf.update(self._config, key, value)
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Output path. If None, overwrites original file.
        """
        save_path = Path(path) if path else self.config_path
        OmegaConf.save(self._config, save_path)
        logger.info(f"Saved configuration to {save_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(self._config, resolve=True) # type: ignore
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return OmegaConf.to_yaml(self._config)
    
    def get_structured_config(self, section: str) -> Any:
        """Get a structured configuration section as a dataclass object."""
        section_config = self.get(section)
        if section_config is None:
            return None
        
        # Map section names to dataclass types
        section_map = {
            'segmentation': SegmentationConfig,
            'training': TrainingConfig,
            'tracking': TrackingConfig,
            'filtering': FilterConfig,
            'postprocessing': PostprocessingConfig,
        }
        
        if section in section_map:
            # Convert to object using the structured config approach
            structured_conf = OmegaConf.structured(section_map[section])
            merged = OmegaConf.merge(structured_conf, section_config)
            return OmegaConf.to_object(merged)
        
        return section_config
    
    @property
    def config(self) -> DictConfig:
        """Get the full configuration as DictConfig."""
        return self._config
    
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
    
    @property
    def filtering(self) -> Dict[str, Any]:
        """Get filtering configuration."""
        return self.get('filtering', {})
    
    @property
    def postprocessing(self) -> Dict[str, Any]:
        """Get postprocessing configuration."""
        return self.get('postprocessing', {})
        
    def merge_with_overrides(self, overrides: Dict[str, Any]) -> 'ConfigManager':
        """Create a new ConfigManager with overrides applied."""
        # Make a copy of the current config
        merged_config = OmegaConf.create(self._config)
        
        # Apply overrides using dot notation
        for key, value in overrides.items():
            OmegaConf.update(merged_config, key, value)
        
        # Create new instance with merged config
        new_manager = ConfigManager.__new__(ConfigManager)
        new_manager.config_path = self.config_path
        new_manager._config = merged_config
        
        return new_manager
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigManager':
        """Create ConfigManager from dictionary."""
        config = OmegaConf.create(config_dict)
        schema = OmegaConf.structured(PipelineConfig)
        merged_config = OmegaConf.merge(schema, config)
        
        instance = cls.__new__(cls)
        instance.config_path = None
        instance._config = merged_config
        
        return instance
    
    @classmethod
    def from_cli_args(cls, args_dict: Dict[str, Any], base_config_path: Optional[str] = None) -> 'ConfigManager':
        """Create ConfigManager from CLI arguments with optional base config."""
        if base_config_path:
            base_manager = cls(base_config_path)
            return base_manager.merge_with_overrides(args_dict)
        else:
            return cls.from_dict(args_dict)


# Maintain backward compatibility with old Config class
class Config(ConfigManager):
    """Backward compatibility wrapper for ConfigManager."""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        # Store config as dict for backward compatibility
        self._config_dict = self.to_dict()
    
    def _load_config(self) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        # Call parent method and convert to dict
        super()._load_config()
        return self.to_dict()


# Global configuration instance
_config_manager: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def set_config(config_path: Union[str, Path]) -> None:
    """Set global configuration manager from file."""
    global _config_manager
    _config_manager = ConfigManager(config_path)

def load_config(config_path: Union[Path, str]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_manager = ConfigManager(config_path)
    return config_manager.to_dict()

def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dict to OmegaConf and save
    omega_config = OmegaConf.create(config)
    OmegaConf.save(omega_config, output_path)

def create_config_from_overrides(base_config_path: Optional[Union[str, Path]] = None, 
                                **overrides) -> ConfigManager:
    """
    Create configuration with overrides.
    
    Args:
        base_config_path: Base configuration file path
        **overrides: Configuration overrides as keyword arguments
        
    Returns:
        ConfigManager instance with overrides applied
    """
    if base_config_path:
        return ConfigManager.from_cli_args(overrides, str(base_config_path))
    else:
        return ConfigManager.from_dict(overrides)

def merge_configs(*config_paths: Union[str, Path]) -> ConfigManager:
    """
    Merge multiple configuration files.
    
    Args:
        *config_paths: Paths to configuration files to merge
        
    Returns:
        ConfigManager with merged configuration
    """
    if not config_paths:
        return ConfigManager()
    
    configs = [OmegaConf.load(path) for path in config_paths]
    merged = OmegaConf.merge(*configs)
    
    manager = ConfigManager.__new__(ConfigManager)
    manager.config_path = Path(config_paths[0])
    manager._config = merged
    
    return manager
