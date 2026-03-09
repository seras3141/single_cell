"""
Performance optimization utilities for the visualization pipeline.

This module provides utilities for improving performance when working with
large datasets and complex visualizations.
"""

import logging
import time
import functools
import gc
from typing import Any, Callable, Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage during pipeline operations."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.start_memory = 0
    
    def start(self) -> None:
        """Start monitoring memory usage."""
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        logger.debug(f"Memory monitoring started. Initial memory: {self.start_memory:.1f} MB")
    
    def check(self) -> float:
        """Check current memory usage and update peak."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        return current_memory
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        current = self.check()
        return {
            'start_memory_mb': self.start_memory,
            'current_memory_mb': current,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': current - self.start_memory
        }


@contextmanager
def memory_profiler():
    """Context manager for monitoring memory usage."""
    monitor = MemoryMonitor()
    monitor.start()
    
    try:
        yield monitor
    finally:
        stats = monitor.get_stats()
        logger.info(f"Memory usage - Peak: {stats['peak_memory_mb']:.1f} MB, "
                   f"Increase: {stats['memory_increase_mb']:.1f} MB")


def profile_performance(func: Callable) -> Callable:
    """Decorator to profile function performance."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        with memory_profiler() as monitor:
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(f"{func.__name__} completed in {execution_time:.2f}s")
                
                # Add performance metadata if result is a dict
                if isinstance(result, dict):
                    result['_performance'] = {
                        'execution_time_s': execution_time,
                        'memory_stats': monitor.get_stats()
                    }
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
                raise
    
    return wrapper


class DataOptimizer:
    """Optimize data for faster processing."""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
        """
        Optimize DataFrame dtypes to reduce memory usage.
        
        Args:
            df: DataFrame to optimize
            aggressive: Use more aggressive optimization (may lose precision)
            
        Returns:
            Optimized DataFrame
        """
        logger.info(f"Optimizing dtypes for DataFrame with {len(df)} rows, {len(df.columns)} columns")
        
        original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type != 'object':
                # Optimize numeric types
                if pd.api.types.is_integer_dtype(col_type):
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
                elif pd.api.types.is_float_dtype(col_type):
                    downcast_type = 'float' if aggressive else None
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast=downcast_type)
            else:
                # Convert object types to category if beneficial
                num_unique_values = optimized_df[col].nunique()
                num_total_values = len(optimized_df[col])
                
                if num_unique_values / num_total_values < 0.5:  # Less than 50% unique
                    optimized_df[col] = optimized_df[col].astype('category')
        
        new_memory = optimized_df.memory_usage(deep=True).sum() / 1024**2  # MB
        reduction_percent = (original_memory - new_memory) / original_memory * 100
        
        logger.info(f"Memory optimization: {original_memory:.1f}MB -> {new_memory:.1f}MB "
                   f"({reduction_percent:.1f}% reduction)")
        
        return optimized_df
    
    @staticmethod
    def sample_data(df: pd.DataFrame, sample_size: Optional[int] = None, 
                   sample_fraction: Optional[float] = None, 
                   stratify_column: Optional[str] = None) -> pd.DataFrame:
        """
        Sample data for faster processing during development/testing.
        
        Args:
            df: DataFrame to sample
            sample_size: Absolute number of samples
            sample_fraction: Fraction of data to sample
            stratify_column: Column to stratify sampling by
            
        Returns:
            Sampled DataFrame
        """
        if sample_size is None and sample_fraction is None:
            return df
        
        if sample_size and len(df) <= sample_size:
            logger.info("Dataset smaller than requested sample size, returning full dataset")
            return df
        
        if sample_fraction and sample_fraction >= 1.0:
            logger.info("Sample fraction >= 1.0, returning full dataset")
            return df
        
        logger.info(f"Sampling data from {len(df)} rows")
        
        if stratify_column and stratify_column in df.columns:
            # Stratified sampling
            sampled_dfs = []
            for group_val, group_df in df.groupby(stratify_column):
                if sample_size:
                    group_sample_size = int(sample_size * len(group_df) / len(df))
                    group_sample_size = min(group_sample_size, len(group_df))
                    group_sample = group_df.sample(n=group_sample_size, random_state=42)
                else:
                    group_sample = group_df.sample(frac=sample_fraction, random_state=42)
                sampled_dfs.append(group_sample)
            
            sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        else:
            # Simple random sampling
            if sample_size:
                sampled_df = df.sample(n=sample_size, random_state=42)
            else:
                sampled_df = df.sample(frac=sample_fraction, random_state=42)
        
        logger.info(f"Sampled {len(sampled_df)} rows ({len(sampled_df)/len(df)*100:.1f}%)")
        return sampled_df
    
    @staticmethod
    def chunk_data(df: pd.DataFrame, chunk_size: int = 10000) -> list:
        """
        Split DataFrame into chunks for batch processing.
        
        Args:
            df: DataFrame to chunk
            chunk_size: Size of each chunk
            
        Returns:
            List of DataFrame chunks
        """
        logger.info(f"Splitting {len(df)} rows into chunks of size {chunk_size}")
        
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks


class ParallelProcessor:
    """Handle parallel processing of visualization tasks."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers or min(4, psutil.cpu_count())
        logger.info(f"ParallelProcessor initialized with {self.max_workers} workers")
    
    def process_batch(self, tasks: list, task_func: Callable, 
                     progress_callback: Optional[Callable] = None) -> list:
        """
        Process a batch of tasks in parallel.
        
        Args:
            tasks: List of task arguments
            task_func: Function to apply to each task
            progress_callback: Optional callback for progress reporting
            
        Returns:
            List of results
        """
        logger.info(f"Processing {len(tasks)} tasks in parallel")
        
        results = []
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(task_func, task): i 
                for i, task in enumerate(tasks)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task_index = future_to_task[future]
                
                try:
                    result = future.result()
                    results.append((task_index, result))
                    completed_count += 1
                    
                    if progress_callback:
                        progress_callback(completed_count, len(tasks))
                    
                except Exception as e:
                    logger.error(f"Task {task_index} failed: {e}")
                    results.append((task_index, None))
        
        # Sort results by original task index
        results.sort(key=lambda x: x[0])
        final_results = [result for _, result in results]
        
        logger.info(f"Completed {len(final_results)} tasks")
        return final_results


class CacheManager:
    """Manage caching of intermediate results."""
    
    def __init__(self, cache_dir: Union[str, Path] = "cache"):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        logger.info(f"CacheManager initialized with directory: {self.cache_dir}")
    
    def get_cache_path(self, key: str) -> Path:
        """Get path for cache file."""
        # Create safe filename from key
        safe_key = "".join(c for c in key if c.isalnum() or c in ('-', '_')).rstrip()
        return self.cache_dir / f"{safe_key}.pkl"
    
    def get(self, key: str) -> Any:
        """
        Get cached result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        cache_path = self.get_cache_path(key)
        
        if cache_path.exists():
            try:
                import pickle
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                logger.debug(f"Cache hit for key: {key}")
                return result
            except Exception as e:
                logger.warning(f"Failed to load cache for key {key}: {e}")
                return None
        
        logger.debug(f"Cache miss for key: {key}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Cache a result.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        cache_path = self.get_cache_path(key)
        
        try:
            import pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logger.debug(f"Cached result for key: {key}")
        except Exception as e:
            logger.warning(f"Failed to cache result for key {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cache cleared")
    
    def get_size(self) -> Dict[str, Any]:
        """Get cache size information."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'num_files': len(cache_files),
            'total_size_mb': total_size / 1024**2,
            'cache_dir': str(self.cache_dir)
        }


# Utility functions for memory cleanup
def cleanup_memory():
    """Force garbage collection and memory cleanup."""
    gc.collect()
    logger.debug("Memory cleanup performed")


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024**2,
        'vms_mb': memory_info.vms / 1024**2,
        'percent': process.memory_percent()
    }


# Context manager for automatic cleanup
@contextmanager
def auto_cleanup():
    """Context manager that automatically cleans up memory."""
    try:
        yield
    finally:
        cleanup_memory()
