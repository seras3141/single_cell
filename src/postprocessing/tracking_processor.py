"""
Unified postprocessing pipeline for 3D cell tracking with optional blur-based filtering.

This module provides a single class for both batch and single-file processing,
with optional saving of intermediate results and TIFF-only output.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm
import json
import time
from joblib import Parallel, delayed

from .cell_tracking import CellTracker3D
from .blur_filtering import BlurFilter, blur_intensity_metric

from src.utils.file_utils import BlurFileHandler
from src.utils.blur_measure import get_or_compute_blur_heatmap
from src.utils.config_schemas import PostprocessingConfig, TrackingConfig, FilterConfig

logger = logging.getLogger(__name__)


class CellTrackingPipeline:
    def __init__(self, config: Optional[PostprocessingConfig] = None):
        self.config = config or PostprocessingConfig()
        self.tracker = CellTracker3D(self.config.tracking)
        self.blur_filter = BlurFilter(self.config.filtering) if self.config.enable_blur_filtering else None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _get_or_compute_blur_heatmap(self, image_path: Path, blur_cache_dir: Optional[Path]) -> Optional[np.ndarray]:
        """
        Get or compute the blur heatmap for a given image.
        """
        if not self.config.enable_blur_filtering:
            return None
        
        if blur_cache_dir is not None:
            blur_file_handler = BlurFileHandler()
            blur_file_name = blur_file_handler.rename_image(str(image_path), self.config.filtering.blur_map_suffix)
            blur_path = blur_cache_dir / blur_file_name

        sharpness_image = get_or_compute_blur_heatmap(
            image_path,
            blur_path=blur_path if blur_cache_dir else None,
            patch_size=self.config.filtering.patch_size,
            stride_size=self.config.filtering.stride_size,
            normalize=True
        )
        
        return sharpness_image

    def _get_candidate_image_for_segmentation(self, segmentation_path: Path, image_dir: Path, segmentation_suffix: str, image_suffix: str) -> Optional[Path]:
        """
        Find the corresponding image file for a given segmentation file.
        Assumes image files have the same base name as segmentation files with a different suffix.
        Args:
            segmentation_path: Path to the segmentation file
            image_dir: Directory containing image files
        """

        # TODO : Implement a file handler to rename files based on patterns
        # Replace suffix
        image_name = segmentation_path.name.replace(segmentation_suffix, image_suffix)
        image_path = image_dir / image_name
        
        if image_path.exists():
            return image_path
        self.logger.warning(f"No corresponding image found for {segmentation_path.name}")
        return None
    
    def _check(self):
        """
        Check if the pipeline is properly configured.
        Raises an error if any required configuration is missing.
        """

        if not self.config.tracking:
            raise ValueError("Tracking configuration is not set.")
        
        if self.config.enable_blur_filtering and not self.blur_filter:
            raise ValueError("Blur filtering is enabled but no filter configuration provided.")

    def process_single_file(self, segmentation_path: Union[str, Path], image_path: Union[str, Path],
                           output_dir: Optional[Union[str, Path]] = None, output_manager = None,
                           blur_cache_dir: Optional[Union[str, Path]] = None,
                           filename_prefix: Optional[str] = None) -> Dict[str, Any]:
        """ Process a single segmentation file with its corresponding image file.
        Args:
            segmentation_path: Path to the segmentation file (TIFF format)
            image_path: Path to the corresponding image file (TIFF format)
            output_dir: Directory to save the output results
            blur_cache_dir: Optional directory to read or cache blur heatmaps
            filename_prefix: Optional prefix for output filenames
        Returns:
            A dictionary containing results and paths of processed files.
        """
        # Start overall timing
        start_time = time.time()
        timing_results = {}
        
        segmentation_path = Path(segmentation_path)
        image_path = Path(image_path)

        if output_manager is None:
            assert output_dir is not None, "output_dir must be provided if output_manager is not set"
            output_manager = TrackingOutputManager(output_dir)

        if filename_prefix is None:
            filename_prefix = segmentation_path.stem
        self.logger.info(f"Processing {segmentation_path.name}")

        # Time: Loading segmentation
        step_start = time.time()
        segmentation_stack = tiff.imread(str(segmentation_path)).astype(np.uint16)
        if not segmentation_stack.ndim == 3:
            raise ValueError(f"Segmentation file {segmentation_path} is not a 3D stack.")
        timing_results['load_segmentation'] = time.time() - step_start
        self.logger.info(f"Load segmentation: {timing_results['load_segmentation']:.2f}s")

        # Time: Blur heatmap computation
        step_start = time.time()
        blur_heatmap = self._get_or_compute_blur_heatmap(image_path, Path(blur_cache_dir) if blur_cache_dir else None)
        timing_results['blur_heatmap'] = time.time() - step_start
        self.logger.info(f"Blur heatmap: {timing_results['blur_heatmap']:.2f}s")

        current_stack = segmentation_stack.copy()
        results = {'input_segmentation': str(segmentation_path), 'input_image': str(image_path)}

        # Processing order
        self._check()

        # 1. If blur filtering is enabled, apply it before (default) or after tracking based on config
        if self.config.enable_blur_filtering and self.config.filter_before_tracking:
            step_start = time.time()
            current_stack, quality_stats = self.blur_filter.filter_3d_stack(current_stack, blur_heatmap) # type: ignore
            timing_results['blur_filtering_before'] = time.time() - step_start
            self.logger.info(f"Blur filtering (before tracking): {timing_results['blur_filtering_before']:.2f}s")
            results['blur_filtering'] = {'output_shape': current_stack.shape} # type: ignore
            
            if self.config.save_intermediate_results:
                step_start = time.time()
                results['blur_filtered_path'] = output_manager.save_blur_filtered(current_stack, filename_prefix)
                timing_results['save_blur_filtered'] = time.time() - step_start
                self.logger.info(f"Save blur filtered: {timing_results['save_blur_filtered']:.2f}s")

        # 2. Track cells in the current stack
        step_start = time.time()
        tracked_stack = self.tracker.track_cells(current_stack)
        timing_results['tracking'] = time.time() - step_start
        self.logger.info(f"Cell tracking: {timing_results['tracking']:.2f}s")
        results['tracking'] = {'output_shape': tracked_stack.shape} # type: ignore
        
        if self.config.save_intermediate_results:
            step_start = time.time()
            results['tracked_path'] = output_manager.save_tracked(tracked_stack, filename_prefix)
            timing_results['save_tracked'] = time.time() - step_start
            self.logger.info(f"Save tracked: {timing_results['save_tracked']:.2f}s")

        # 3. Optionally filter after tracking
        if self.config.enable_blur_filtering and not self.config.filter_before_tracking:
            step_start = time.time()
            tracked_stack, quality_stats = self.blur_filter.filter_3d_stack(tracked_stack, blur_heatmap) # type: ignore
            timing_results['blur_filtering_after'] = time.time() - step_start
            self.logger.info(f"Blur filtering (after tracking): {timing_results['blur_filtering_after']:.2f}s")
            results['blur_filtering'] = {'output_shape': tracked_stack.shape} # type: ignore
            
            if self.config.save_intermediate_results:
                step_start = time.time()
                results['tracked_blur_filtered_path'] = output_manager.save_tracked_blur_filtered(tracked_stack, filename_prefix)
                timing_results['save_tracked_blur_filtered'] = time.time() - step_start
                self.logger.info(f"Save tracked+blur filtered: {timing_results['save_tracked_blur_filtered']:.2f}s")

        # 4. Save final result
        step_start = time.time()
        final_output_path = output_manager.save_final(tracked_stack, filename_prefix)
        timing_results['save_final'] = time.time() - step_start
        self.logger.info(f"Save final: {timing_results['save_final']:.2f}s")
        results['final_output'] = final_output_path

        # Optional : Convert final output to 2D if required
        if self.config.convert_to_2d:
            step_start = time.time()
            from src.utils.conversion import split_3d_to_2d
            split_3d_to_2d(final_output_path, output_manager.subdirs['final_2d'], suffix="masks")
            timing_results['convert_to_2d'] = time.time() - step_start
            self.logger.info(f"Convert to 2D: {timing_results['convert_to_2d']:.2f}s")

        # Total time
        total_time = time.time() - start_time
        timing_results['total'] = total_time
        results['timing'] = timing_results
        
        self.logger.info(f"Total processing time: {total_time:.2f}s")
        self.logger.info(f"Timing breakdown: {json.dumps(timing_results, indent=2)}")

        return results


    def process_single_file_opt(
            self, 
            segmentation_path: Union[str, Path], 
            image_path: Union[str, Path],
            segmentation_suffix: Optional[str] = "_masks.tif",
            output_dir: Optional[Union[str, Path]] = None,
            output_manager = None,
            blur_cache_dir: Optional[Union[str, Path]] = None,
            filename_prefix: Optional[str] = None) -> Dict[str, Any]:
        """ Process a single segmentation file with its corresponding image file.
        Args:
            segmentation_path: Path to the segmentation file (TIFF format)
            image_path: Path to the corresponding image file (TIFF format)
            output_dir: Directory to save the output results
            blur_cache_dir: Optional directory to read or cache blur heatmaps
            filename_prefix: Optional prefix for output filenames
        Returns:
            A dictionary containing results and paths of processed files.
        """
        # Start overall timing
        start_time = time.time()
        timing_results = {}
        
        segmentation_path = Path(segmentation_path)
        image_path = Path(image_path)

        if output_manager is None:
            assert output_dir is not None, "output_dir must be provided if output_manager is not set"
            output_manager = TrackingOutputManager(output_dir)

        if filename_prefix is None:
            filename_prefix = segmentation_path.stem
        self.logger.info(f"Processing {segmentation_path.name}")

        # Time: Loading segmentation
        t0 = time.time()
        segmentation_stack = tiff.imread(str(segmentation_path)).astype(np.uint16)
        if not segmentation_stack.ndim == 3:
            raise ValueError(f"Segmentation file {segmentation_path} is not a 3D stack.")
        timing_results['load_segmentation'] = time.time() - t0
        self.logger.info(f"Load segmentation: {timing_results['load_segmentation']:.2f}s")

        # Time: Blur heatmap computation
        t0 = time.time()
        blur_heatmap = self._get_or_compute_blur_heatmap(image_path, Path(blur_cache_dir) if blur_cache_dir else None)
        timing_results['blur_heatmap'] = time.time() - t0
        self.logger.info(f"Blur heatmap: {timing_results['blur_heatmap']:.2f}s")

        current_stack = segmentation_stack.copy()
        results = {'input_segmentation': str(segmentation_path), 'input_image': str(image_path)}

        # Processing order
        self._check()

        t0 = time.time()
        current_stack, quality_stats = self.blur_filter.filter_3d_stack_fast(current_stack, blur_heatmap) # type: ignore
        timing_results['blur_filtering_before'] = time.time() - t0
        self.logger.info(f"Blur filtering (before tracking): {timing_results['blur_filtering_before']:.2f}s")
        results['blur_filtering'] = {'output_shape': current_stack.shape} # type: ignore
        
        if self.config.save_intermediate_results:
            t0 = time.time()
            results['blur_filtered_path'] = output_manager.save_blur_filtered(current_stack, filename_prefix)
            timing_results['save_blur_filtered'] = time.time() - t0
            self.logger.info(f"Save blur filtered: {timing_results['save_blur_filtered']:.2f}s")

        # 2. Track cells in the current stack
        t0 = time.time()
        tracked_stack = self.tracker.track_cells(current_stack)
        timing_results['tracking'] = time.time() - t0
        self.logger.info(f"Cell tracking: {timing_results['tracking']:.2f}s")
        results['tracking'] = {'output_shape': tracked_stack.shape} # type: ignore
        
        if self.config.save_intermediate_results:
            t0 = time.time()
            results['tracked_path'] = output_manager.save_tracked(tracked_stack, filename_prefix)
            timing_results['save_tracked'] = time.time() - t0
            self.logger.info(f"Save tracked: {timing_results['save_tracked']:.2f}s")

        # 4. Save final result
        t0 = time.time()
        final_output_path = output_manager.save_final(tracked_stack, filename_prefix)
        timing_results['save_final'] = time.time() - t0
        self.logger.info(f"Save final: {timing_results['save_final']:.2f}s")
        results['final_output'] = final_output_path

        # Optional : Convert final output to 2D if required
        if self.config.convert_to_2d:
            t0 = time.time()
            from src.utils.conversion import split_3d_to_2d
            split_3d_to_2d(final_output_path, output_manager.subdirs['final_2d'], suffix=segmentation_suffix)
            timing_results['convert_to_2d'] = time.time() - t0
            self.logger.info(f"Convert to 2D: {timing_results['convert_to_2d']:.2f}s")

        # Total time
        total_time = time.time() - start_time
        timing_results['total'] = total_time
        results['timing'] = timing_results
        
        self.logger.info(f"Total processing time: {total_time:.2f}s")
        self.logger.info(f"Timing breakdown: {json.dumps(timing_results, indent=2)}")

        return results

    def _process_single_wrapper(
        self,
        seg_file: Path,
        image_dir: Path,
        mask_suffix: str,
        image_suffix: str,
        blur_cache_dir: Optional[Path],
        output_manager: "TrackingOutputManager",
    ) -> Optional[Dict[str, Any]]:
        """Error-handling wrapper around process_single_file_opt for parallel execution."""
        image_file = self._get_candidate_image_for_segmentation(seg_file, image_dir, mask_suffix, image_suffix)
        if image_file is None:
            return None
        try:
            return self.process_single_file_opt(
                seg_file, image_file,
                segmentation_suffix=mask_suffix,
                blur_cache_dir=blur_cache_dir,
                output_manager=output_manager,
            )
        except Exception as e:
            self.logger.error(f"Failed to process {seg_file.name}: {e}")
            return {'input_segmentation': str(seg_file), 'error': str(e)}

    def process_batch(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        output_dir: Union[str, Path],
        blur_cache_dir: Optional[Union[str, Path]] = None,
        image_pattern: Optional[str] = None,
        mask_pattern: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """ Process a batch of segmentation files with their corresponding image files.
        Args:
            image_dir: Directory containing image files (TIFF format)
            mask_dir: Directory containing segmentation files (TIFF format)
            output_dir: Directory to save the output results
            blur_cache_dir: Optional directory to read or cache blur heatmaps
        Returns:
            A list of dictionaries containing results and paths of processed files.
        """
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)

        output_manager = TrackingOutputManager(output_dir)

        image_pattern = image_pattern or self.config.image_pattern
        mask_pattern = mask_pattern or self.config.mask_pattern

        image_suffix = image_pattern.replace("*", "")
        mask_suffix = mask_pattern.replace("*", "")

        seg_files = list(mask_dir.glob(mask_pattern))
        if not seg_files:
            self.logger.warning(f"No segmentation files found in {mask_dir} with suffix {mask_suffix}")
            return []

        # Patched up code for slurm parallel execution 
        # TODO : will be refactored in the future when we have a more robust file handling system in place
        if n_jobs is None:
            n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        self.logger.info(f"Processing {len(seg_files)} files with {n_jobs} workers")

        blur_cache_path = Path(blur_cache_dir) if blur_cache_dir else None

        raw_results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(self._process_single_wrapper)(
                seg_file, image_dir, mask_suffix, image_suffix,
                blur_cache_path, output_manager,
            )
            for seg_file in tqdm(seg_files, desc="Processing files")
        )

        results = [r for r in raw_results if r is not None]

        # Save batch summary as JSON
        output_manager.save_batch_summary(results)
        return results

class TrackingOutputManager:
    """Handles saving of intermediate and final outputs for cell tracking pipeline."""
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.subdirs = {
            'blur_filtered': self.output_dir / 'blur_filtered',
            'tracked': self.output_dir / 'tracked',
            'tracked_blur_filtered': self.output_dir / 'tracked_blur_filtered',
            'final': self.output_dir / 'final',
            'final_2d': self.output_dir / 'final_2d',
        }
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)

    def save_blur_filtered(self, arr: np.ndarray, filename_prefix: str):
        path = self.subdirs['blur_filtered'] / f"{filename_prefix}.tif"
        tiff.imwrite(str(path), arr)
        return str(path)

    def save_tracked(self, arr: np.ndarray, filename_prefix: str):
        path = self.subdirs['tracked'] / f"{filename_prefix}.tif"
        tiff.imwrite(str(path), arr)
        return str(path)

    def save_tracked_blur_filtered(self, arr: np.ndarray, filename_prefix: str):
        path = self.subdirs['tracked_blur_filtered'] / f"{filename_prefix}.tif"
        tiff.imwrite(str(path), arr)
        return str(path)

    def save_final(self, arr: np.ndarray, filename_prefix: str):
        path = self.subdirs['final'] / f"{filename_prefix}.tif"
        tiff.imwrite(str(path), arr)
        return str(path)

    def save_batch_summary(self, results: list, filename: str = "batch_summary.json"):
        summary_path = self.output_dir / filename
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return str(summary_path)
