"""
Dimensionality Reduction Utilities for Single-Cell Analysis

This module provides standardized dimensionality reduction functionality for
single-cell feature analysis. Supports PCA, t-SNE, and UMAP with optimized
parameters for visualization and analysis.

Features:
- PCA with explained variance tracking
- t-SNE with adaptive parameters  
- UMAP with balanced speed/quality parameters
- Automatic handling of NaN values and data preprocessing
- Consistent interface across all reduction methods
- Memory-efficient processing for large datasets
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

from src.utils.feature_list import get_all_feature_names
import logging


class DimensionalityReducer:
    """
    Unified dimensionality reduction class for single-cell feature analysis.
    
    Provides consistent interface and optimized parameters for PCA, t-SNE, and UMAP.
    Handles data preprocessing, NaN values, and provides detailed reduction metadata.
    """

    FEATURE_DICT = get_all_feature_names()
    
    def __init__(self):
        """Initialize the dimensionality reducer."""
        # Store reduction results for later analysis
        self.last_reducer = None
        self.last_method = None
        self.explained_variance_ratio_ = None
        
    def reduce_dimensionality(self, 
                            data: Union[np.ndarray, pd.DataFrame], 
                            method: str = "pca", 
                            n_components: int = 2,
                            feature_columns: Optional[List[str]] = None,
                            **kwargs) -> np.ndarray:
        """
        Perform dimensionality reduction on feature data.
        
        Args:
            data: Input data (numpy array or pandas DataFrame)
            method: Reduction method ('pca', 'tsne', 'umap')
            n_components: Number of components to reduce to
            feature_columns: Column names if data is DataFrame (auto-detected if None)
            **kwargs: Additional parameters for the reduction method
            
        Returns:
            Tuple of (reduced_features, fitted_reducer_object)
            
        Raises:
            ValueError: If unknown reduction method or invalid parameters
        """
        # Handle DataFrame input
        if isinstance(data, pd.DataFrame):
            if feature_columns is None:
                feature_columns = self._get_feature_columns(data)
            X = data[feature_columns].values
        else:
            X = data
        
        # Data validation and preprocessing
        X = self._preprocess_features(X)
        
        # Store method for reference
        self.last_method = method.lower()
        
        # Apply reduction method
        if self.last_method == "pca":
            X_reduced, reducer = self._apply_pca(X, n_components, **kwargs)
        elif self.last_method == "tsne":
            X_reduced, reducer = self._apply_tsne(X, n_components, **kwargs)
        elif self.last_method == "umap":
            X_reduced, reducer = self._apply_umap(X, n_components, **kwargs)
        else:
            raise ValueError(f"Unknown reduction method: {method}. Use 'pca', 'tsne', or 'umap'")
        
        # Store reducer for later use
        self.last_reducer = reducer
        
        #return reduced features only
        return X_reduced
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Automatically identify feature columns (exclude metadata columns).
        """
        exclude_cols = [
            'instance_id', 'filename', 'image_filename', 'image_path', 'mask_filename',
            'sample_id', 'z_stack', 'sample_z_id', 'processing_timestamp',
            'feature_extraction_version', 'dataset_name'
        ]
        
        # Include common morphological, intensity, and texture features
        include_patterns = self.FEATURE_DICT['morphology'] + \
            self.FEATURE_DICT['intensity'] + self.FEATURE_DICT['texture']
        
        # include_patterns = [
        #     'area', 'perimeter', 'elongation', 'compactness', 'circularity',
        #     'feret_diameter', 'radius_of_gyration', 'major_axis', 'minor_axis',
        #     'mean_intensity', 'std_intensity', 'cv_intensity', 'total_intensity',
        #     'centroid_x', 'centroid_y', 'center_of_mass_x', 'center_of_mass_y', 
        #     'mass_displacement', 'gabor_mean', 'gabor_std', 'skewness', 
        #     'kurtosis', 'entropy'
        # ]
        
        # Get columns that match include patterns and are not in exclude list
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols and df[col].dtype in ['int64', 'float64']:
                # Either exact match
                if col in include_patterns:
                    feature_cols.append(col)

        logging.debug(f"Identified {len(feature_cols)} feature columns for reduction.")
        return feature_cols
    
    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess features for dimensionality reduction.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Preprocessed feature matrix
        """
        # Handle NaN values
        if np.any(np.isnan(X)):
            warnings.warn("NaN values detected in features. Replacing with zeros.")
            X = np.nan_to_num(X, nan=0.0)
        
        # Handle infinite values
        if np.any(np.isinf(X)):
            warnings.warn("Infinite values detected in features. Replacing with finite values.")
            X = np.nan_to_num(X, posinf=np.finfo(X.dtype).max, neginf=np.finfo(X.dtype).min) # type: ignore
        
        # Basic validation
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 samples for dimensionality reduction")
        
        if X.shape[1] == 0:
            raise ValueError("No features found for dimensionality reduction")
        
        return X
    
    def _apply_pca(self, X: np.ndarray, n_components: int, **kwargs) -> Tuple[np.ndarray, PCA]:
        """Apply PCA with optimized parameters."""
        # Default PCA parameters
        default_params = {
            'random_state': 42
        }
        default_params.update(kwargs)
        
        # Ensure n_components doesn't exceed data dimensions
        max_components = min(X.shape[0], X.shape[1])
        n_components = min(n_components, max_components)
        
        reducer = PCA(n_components=n_components, **default_params) # type: ignore
        X_reduced = reducer.fit_transform(X)
        
        # Store explained variance for reporting
        self.explained_variance_ratio_ = reducer.explained_variance_ratio_
        
        return X_reduced, reducer
    
    def _apply_tsne(self, X: np.ndarray, n_components: int, **kwargs) -> Tuple[np.ndarray, TSNE]:
        """Apply t-SNE with optimized parameters."""
        # For t-SNE, first reduce with PCA if we have too many features
        if X.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            X = pca.fit_transform(X)
        
        # Default t-SNE parameters optimized for visualization
        default_params = {
            'random_state': 42,
            'perplexity': min(30, max(5, X.shape[0] // 4)),  # Adaptive perplexity
            'max_iter': 1000,
            'learning_rate': 'auto',
            'n_iter_without_progress': 300
        }
        default_params.update(kwargs)
        
        # Ensure perplexity is valid for dataset size
        max_perplexity = (X.shape[0] - 1) // 3
        if default_params['perplexity'] >= max_perplexity:
            default_params['perplexity'] = max(1, max_perplexity - 1)
            warnings.warn(f"Adjusted t-SNE perplexity to {default_params['perplexity']} based on dataset size")
        
        reducer = TSNE(n_components=n_components, **default_params)
        X_reduced = reducer.fit_transform(X)
        
        return X_reduced, reducer
    
    def _apply_umap(self, X: np.ndarray, n_components: int, **kwargs) -> Tuple[np.ndarray, umap.UMAP]:
        """Apply UMAP with optimized parameters."""
        # Default UMAP parameters balanced for speed and quality
        default_params = {
            'random_state': 42,
            'n_neighbors': min(15, max(2, X.shape[0] // 10)),  # Adaptive neighbors
            'min_dist': 0.1,
            'metric': 'euclidean',
            'n_jobs': 1  # Avoid parallelization issues
        }
        default_params.update(kwargs)
        
        # Ensure n_neighbors is valid for dataset size
        if default_params['n_neighbors'] >= X.shape[0]:
            default_params['n_neighbors'] = max(2, X.shape[0] - 1)
            warnings.warn(f"Adjusted UMAP n_neighbors to {default_params['n_neighbors']} based on dataset size")
        
        reducer = umap.UMAP(n_components=n_components, **default_params)
        X_reduced = reducer.fit_transform(X)
        
        return X_reduced, reducer # type: ignore
    
    def get_reduction_info(self) -> Dict[str, Any]:
        """
        Get information about the last dimensionality reduction performed.
        
        Returns:
            Dictionary with reduction metadata
        """
        info = {
            'method': self.last_method,
            'reducer': self.last_reducer,
            'explained_variance_ratio': self.explained_variance_ratio_
        }
        
        # Add method-specific information
        if self.last_method == "pca" and self.explained_variance_ratio_ is not None:
            info['total_variance_explained'] = np.sum(self.explained_variance_ratio_)
        
        return info
    
    def transform_new_data(self, X_new: Union[np.ndarray, pd.DataFrame],
                          feature_columns: Optional[List[str]] = None) -> Optional[np.ndarray]:
        """
        Transform new data using the last fitted reducer.
        
        Args:
            X_new: New data to transform
            feature_columns: Column names if X_new is DataFrame
            
        Returns:
            Transformed data
            
        Raises:
            ValueError: If no reducer has been fitted
        """
        if self.last_reducer is None:
            raise ValueError("No reducer fitted. Call reduce_dimensionality first.")
        
        # Handle DataFrame input
        if isinstance(X_new, pd.DataFrame):
            if feature_columns is None:
                feature_columns = self._get_feature_columns(X_new)
            X_new = X_new[feature_columns].values
        
        # Preprocess
        X_new = self._preprocess_features(X_new)
        
        # Transform based on method
        if self.last_method in ["pca"]:
            return self.last_reducer.transform(X_new) # type: ignore
        else:
            # t-SNE and UMAP don't support transform for new data
            warnings.warn(f"{self.last_method} doesn't support transforming new data. "
                         f"Refit the reducer with all data.")
            return None


# Convenience functions for backwards compatibility
def reduce_dimensionality(data: Union[np.ndarray, pd.DataFrame], 
                         method: str = "pca", 
                         n_components: int = 2,
                         feature_columns: Optional[List[str]] = None,
                         **kwargs) -> np.ndarray:
    """
    Convenience function for dimensionality reduction.
    
    Args:
        data: Input data (numpy array or pandas DataFrame)
        method: Reduction method ('pca', 'tsne', 'umap')
        n_components: Number of components to reduce to
        feature_columns: Column names if data is DataFrame
        **kwargs: Additional parameters for the reduction method
        
    Returns:
        Reduced feature array
    """
    reducer = DimensionalityReducer()
    X_reduced = reducer.reduce_dimensionality(
        data, method, n_components, feature_columns, **kwargs
    )
    return X_reduced


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Convenience function to get feature columns from DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of feature column names
    """
    reducer = DimensionalityReducer()
    return reducer._get_feature_columns(df)


# Example usage and testing
def main():
    """Example usage of the dimensionality reduction utilities."""
    print("Dimensionality Reduction Utilities Demo")
    print("=" * 40)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples, n_features = 500, 20
    
    # Generate correlated features
    X = np.random.randn(n_samples, n_features)
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)  # Correlation
    X[:, 2] = 0.5 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * np.random.randn(n_samples)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['sample_id'] = np.random.choice(['A', 'B', 'C'], n_samples)
    df['instance_id'] = range(n_samples)
    
    print(f"Created synthetic dataset: {df.shape}")
    
    # Initialize reducer
    reducer = DimensionalityReducer()
    
    # Test different methods
    methods = ['pca', 'tsne', 'umap']
    results = {}
    
    for method in methods:
        print(f"\nTesting {method.upper()}...")
        try:
            X_reduced = reducer.reduce_dimensionality(
                df, method=method, n_components=3
            )
            results[method] = X_reduced
            
            print(f"  Input shape: {X.shape}")
            print(f"  Output shape: {X_reduced.shape}")
            
            # Get reduction info
            info = reducer.get_reduction_info()
            if info['explained_variance_ratio'] is not None:
                total_var = np.sum(info['explained_variance_ratio'])
                print(f"  Explained variance: {total_var:.3f}")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test convenience function
    print(f"\nTesting convenience function...")
    X_pca = reduce_dimensionality(df, method='pca', n_components=2)
    print(f"  PCA result shape: {X_pca.shape}")
    
    print(f"\nDemo complete!")


if __name__ == "__main__":
    main()
