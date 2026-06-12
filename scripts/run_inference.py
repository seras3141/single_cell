#!/usr/bin/env python3
"""
Run inference on test dataset using trained Cellpose models.

This script provides a command-line interface for running cell segmentation
inference on test datasets with organized output structure.

#TODO : Update usage
Usage:
    python scripts/run_inference.py --input-dir data/test --output-dir results --model-name cyto3
    
    python scripts/run_inference.py --config config/inference_config.yaml
"""

import os
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.distributed as dist

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.inference.inference_pipeline import InferencePipeline

from src.utils.logging_utils import setup_logging
from src.utils.config import get_config_manager

def get_inference_args():
    parser = argparse.ArgumentParser(description="Run inference on cell segmentation test dataset")

    def optional_arg(*args, **kwargs):
        kwargs['default'] = argparse.SUPPRESS  # Don't set a default so we can detect user input
        return parser.add_argument(*args, **kwargs)

    # Data args
    optional_arg("--input-dir", "-i", type=str, help="Directory containing test images")
    optional_arg("--output-dir", "-o", type=str, help="Base output directory for results")
    optional_arg("--dataset-name", type=str, help="Dataset name (default: test)")
    optional_arg("--file-pattern", type=str, help="File pattern (default: *_BF.tif)")
    optional_arg("--process-z-stacks", action="store_true", help="Process images as Z-stacks")
    optional_arg("--no-overlays", action="store_true", help="Skip overlay visualizations")
    optional_arg("--no-metadata", action="store_true", help="Skip saving prediction metadata")
    # Cellpose args
    # TODO : model_type argument is not used in v4.0.1+. Ignoring this argument...
    optional_arg("--model-name", "-m", type=str, help="Model name/type to use (default: cyto3)")
    optional_arg("--model-path", type=str, help="Path to custom model weights")
    optional_arg("--gpu", dest="gpu", action="store_true", help="Use GPU for inference")
    optional_arg("--no-gpu", dest="gpu", action="store_false", help="Disable GPU usage")
    optional_arg("--flow-threshold", type=float, help="Flow error threshold")
    optional_arg("--cellprob-threshold", type=float, help="Cell probability threshold")
    optional_arg("--min-size", type=int, help="Minimum cell size")
    optional_arg("--diameter", type=float, help="Expected cell diameter")
    optional_arg("--config", type=str, help="Path to YAML config file")
    optional_arg("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    optional_arg("--override", "-O", action="append", help="Config overrides in dot notation")
    optional_arg("--overwrite", action="store_true", help="Overwrite existing output files")
    
    return parser.parse_args()

def get_legacy_args(cli_args):
    # Apply legacy CLI argument overrides for backward compatibility
    legacy_overrides = {}
    # Legacy arguments for input/output directories
    if 'input_dir' in cli_args:
        legacy_overrides['paths.input_dir'] = cli_args['input_dir']
    if 'output_dir' in cli_args:
        legacy_overrides['paths.output_dir'] = cli_args['output_dir']
    # Legacy arguments for inference parameters
    if 'dataset_name' in cli_args:
        legacy_overrides['segmentation.inference.dataset_name'] = cli_args['dataset_name']
    if 'file_pattern' in cli_args:
        legacy_overrides['segmentation.inference.file_pattern'] = cli_args['file_pattern']
    if 'no_overlays' in cli_args:
        legacy_overrides['segmentation.inference.save_overlays'] = not cli_args['no_overlays']
    if 'no_metadata' in cli_args:
        legacy_overrides['segmentation.inference.save_metadata'] = not cli_args['no_metadata']
    if 'process_z_stacks' in cli_args:
        legacy_overrides['segmentation.inference.process_z_stacks'] = cli_args['process_z_stacks']
    # Legacy arguments for Cellpose parameters
    if 'gpu' in cli_args:
        legacy_overrides['segmentation.cellpose.gpu'] = cli_args['gpu']
    if 'flow_threshold' in cli_args:
        legacy_overrides['segmentation.cellpose.flow_threshold'] = cli_args['flow_threshold']
    if 'cellprob_threshold' in cli_args:
        legacy_overrides['segmentation.cellpose.cellprob_threshold'] = cli_args['cellprob_threshold']
    if 'min_size' in cli_args:
        legacy_overrides['segmentation.cellpose.min_size'] = cli_args['min_size']
    if 'diameter' in cli_args:
        legacy_overrides['segmentation.cellpose.diameter'] = cli_args['diameter']
    if 'model_name' in cli_args:
        legacy_overrides['segmentation.cellpose.model_type'] = cli_args['model_name']
    if 'overwrite' in cli_args:
        legacy_overrides['segmentation.inference.overwrite'] = cli_args['overwrite']

    return legacy_overrides


def run_inference_from_config(config : Dict[str, Any], input_dir: Optional[str] = None, dataset_name : Optional[str] = None):
    """
    Run inference using a configuration file and command-line arguments.
    Merges config file with CLI args, with CLI args taking precedence.
    """

    pipeline = InferencePipeline.from_config(config=config)
    validation = pipeline.validate_setup()
    if not validation['overall']:
        raise ValueError("Pipeline validation failed")

    inference_config = config.get('segmentation', {}).get('inference', {})

    if input_dir is None:
        input_dir = config.get('paths', {}).get('input_dir', 'data/test')
    if dataset_name is None:
        dataset_name = inference_config.get('dataset_name', 'test')

    model_info = pipeline.get_model_info()
    logging.info(f"Model information: {model_info}")

    results = pipeline.run_inference(
        input_dir=Path(input_dir) / dataset_name, # type: ignore
        file_pattern=inference_config.get('file_pattern', '*_BF.tif'),
        process_z_stacks=inference_config.get('process_z_stacks', False),
        save_overlays=inference_config.get('save_overlays', False),
        save_metadata=inference_config.get('save_metadata', False)
    )

    print("\n" + "="*50)
    print("INFERENCE COMPLETED")
    print("="*50)
    print(f"Total files processed: {len(results['processed_files'])}")
    print(f"Total files skipped (already exist): {len(results.get('skipped_files', []))}")
    print(f"Total cells detected: {results['total_cells']}")
    print(f"Failed files: {len(results['failed_files'])}")
    # print(f"Results saved to: {output_manager.output_dir}")
    print(f"Summary report: {results['summary_path']}")
    if results['failed_files']:
        print("\nFailed files:")
        for failed in results['failed_files']:
            print(f"  - {failed['file']}: {failed['error']}")
    logging.info("Inference completed successfully")


def run_inference_from_config_dist(config : Dict[str, Any], input_dir: Optional[str] = None, dataset_name : Optional[str] = None):
    """
    Run inference using a configuration file and command-line arguments.
    Distributed inference version.
    Merges config file with CLI args, with CLI args taking precedence.
    """

    local_rank = int(os.environ['LOCAL_RANK'])
    torch_device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(torch_device)
    dist.init_process_group(backend='nccl', init_method='env://', device_id=torch_device)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    pipeline = InferencePipeline.from_config(config=config, device=torch_device)
    validation = pipeline.validate_setup()
    if not validation['overall']:
        raise ValueError("Pipeline validation failed")

    inference_config = config.get('segmentation', {}).get('inference', {})

    if input_dir is None:
        input_dir = config.get('paths', {}).get('input_dir', 'data/test')
    if dataset_name is None:
        dataset_name = inference_config.get('dataset_name', 'test')

    model_info = pipeline.get_model_info()
    logging.info(f"Model information: {model_info}")

    input_files = pipeline.get_file_list(Path(input_dir) / dataset_name, inference_config.get('file_pattern', '*_BF.tif')) # type: ignore
    N = len(input_files)

    # Distribute files among processes
    indices = list(range(rank, N, world_size))
    input_files = [input_files[i] for i in indices if i < len(input_files)]

    results = pipeline.run_inference(
        input_files=input_files, # type: ignore
        process_z_stacks=inference_config.get('process_z_stacks', False),
        save_overlays=inference_config.get('save_overlays', False),
        save_metadata=inference_config.get('save_metadata', False),
        combine_z_stacks=False  # Disable combining in each process
    )

    dist.barrier()
    if rank == 0:
        pipeline._combine_2d_to_3d()  # Combine 2D to 3D once in the main process
        # optional: aggregate logs, or create summary
        print("All done.")

    dist.destroy_process_group()



    print("\n" + "="*50)
    print("INFERENCE COMPLETED")
    print("="*50)
    print(f"Total files processed: {len(results['processed_files'])}")
    print(f"Total cells detected: {results['total_cells']}")
    print(f"Failed files: {len(results['failed_files'])}")
    # print(f"Results saved to: {output_manager.output_dir}")
    print(f"Summary report: {results['summary_path']}")
    if results['failed_files']:
        print("\nFailed files:")
        for failed in results['failed_files']:
            print(f"  - {failed['file']}: {failed['error']}")
    logging.info("Inference completed successfully")


def main():
    """Main inference function with OmegaConf configuration support."""
    args = get_inference_args()
    cli_args = vars(args)

    # Set up logging
    setup_logging(cli_args.get("log_level", "INFO"))

    config_manager = get_config_manager(cli_args=cli_args, legacy_args_function=get_legacy_args)

    # Get final config as dict for backward compatibility
    merged_config = config_manager.to_dict()
    logging.info("Final merged configuration:")
    logging.info(merged_config)

    try:
        if merged_config.get('distributed', {}).get('enabled', False):
            logging.info("Starting distributed inference...")
            run_inference_from_config_dist(merged_config)
        else:
            logging.info("Starting inference...")
            run_inference_from_config(merged_config)

    except Exception as e:
        logging.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
