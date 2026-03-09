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
            validate_pipeline_config(OmegaConf.to_object(merged_config)) # type: ignore
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return merged_config # type: ignore
            
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
    manager._config = merged # type: ignore
    
    return manager

def get_paths_from_config(config: ConfigManager) -> Dict[str, Path]:
    """
    Get paths configuration from a config dictionary.
    
    Args:
        config: Configuration dictionary. If None, uses global config.
        
    Returns:
        Dictionary of paths
    """
    config_dict = config.to_dict() 

    paths_config = config_dict.get('paths', {})
    output_dir = paths_config.get('output_dir', 'output')

    # split_folder = config_dict.get("preprocessing", {}).get("split_folder", "split_data")
    # split_dir = os.path.join(output_dir, split_folder)

    out_3d_folder = config_dict.get("preprocessing", {}).get("out_3d_folder", "3d_images")

    results_folder = config_dict["segmentation"]["inference"]["results_folder"]
    model_type = config_dict["segmentation"]["cellpose"]["model_type"]
    dataset_name = config_dict["segmentation"]["inference"]["dataset_name"]

    image_dir = os.path.join(output_dir, out_3d_folder)
    blur_dir = os.path.join(output_dir, "blur_heatmaps")

    mask_dir = os.path.join(output_dir, results_folder, model_type, dataset_name)
    mask_dir = os.path.join(mask_dir, "masks_3d")
    track_dir = os.path.join(mask_dir, "tracking")
    final_dir = os.path.join(track_dir, "final")

    config_out = {
        "output_dir": Path(output_dir),
        "3d_data_dir": Path(image_dir),
        "blur_heatmap_dir": Path(blur_dir),
        "inference_dir": Path(mask_dir),
        "postprocessed_dir": Path(final_dir)
    }

    return config_out


def get_config_manager(cli_args : dict, legacy_args_function : Optional[callable] = None) -> ConfigManager: # type: ignore
    """
    Get a ConfigManager instance with the specified configuration file and legacy overrides.
    """

    def parse_logging_config(cli_args: dict) -> Dict[str, Any]:
        """Parse logging configuration from CLI arguments."""
        legacy_args = {}

        # Logging options
        if "log_level" in cli_args:
            legacy_args["logging.level"] = cli_args["log_level"].upper()
        if "log_file" in cli_args:
            legacy_args["logging.filename"] = cli_args["log_file"]
        if 'logging_level' in cli_args:
            legacy_args['logging.level'] = cli_args['logging_level'].upper()
        if 'logging_file' in cli_args:
            legacy_args['logging.filename'] = cli_args['logging_file']

        return legacy_args

    logger = logging.getLogger(__name__)
    # Remove None values from cli_args
    cli_args = {k: v for k, v in cli_args.items() if v is not None}

    # 1: Load base config (from YAML or default)
    if "config" in cli_args:
        config_manager = ConfigManager(cli_args["config"])
        logger.info(f"Loaded configuration from {cli_args['config']}")
    else:
        config_manager = ConfigManager()  # Use defaults
        logger.info("Using default configuration")

    # 2: Apply dotlist overrides from CLI
    if "override" in cli_args:
        overrides = OmegaConf.from_dotlist(cli_args["override"])
        override_dict = OmegaConf.to_container(overrides)
        logger.info(f"Applying CLI overrides: {cli_args['override']}")
        config_manager = config_manager.merge_with_overrides(override_dict) #type: ignore

    # 3: Apply legacy overrides and Merge config and CLI args
    if legacy_args_function is None or not callable(legacy_args_function):
        legacy_args_function = parse_logging_config

    legacy_overrides = legacy_args_function(cli_args) # type: ignore
    if legacy_overrides:
        config_manager = config_manager.merge_with_overrides(legacy_overrides)
        logger.info(f"Applied legacy CLI overrides: {list(legacy_overrides.keys())}")

    return config_manager
