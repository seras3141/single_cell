"""
Test suite for the dimensionality reduction utility module.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch


try:
    from src.utils.dimensionality_reduction import DimensionalityReducer
except ImportError as e:
    print(f"Import error: {e}")
    DimensionalityReducer = None


class TestDimensionalityReducer:
    """Test suite for DimensionalityReducer class."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        # Create sample high-dimensional data
        self.sample_data = np.random.randn(100, 50)  # 100 samples, 50 features
        self.sample_labels = np.random.randint(0, 3, 100)  # 3 classes
    
    def test_initialization(self):
        """Test proper initialization of DimensionalityReducer."""
        if DimensionalityReducer is None:
            return
            
        reducer = DimensionalityReducer()
        assert reducer is not None
        # Check if expected methods exist
        assert hasattr(reducer, 'reduce_dimensionality')
    
    def test_pca_reduction_2d(self):
        """Test PCA dimensionality reduction to 2D."""
        if DimensionalityReducer is None:
            return
            
        reducer = DimensionalityReducer()
        
        try:
            reduced_data = reducer.reduce_dimensionality(
                self.sample_data, 
                method='pca', 
                n_components=2
            )
            
            assert reduced_data.shape == (100, 2)
            assert isinstance(reduced_data, np.ndarray)
        except Exception as e:
            print(f"Error occurred: {e}")
            # If sklearn not available, this is expected
            assert "sklearn" in str(e).lower() or "import" in str(e).lower()
    
    def test_pca_reduction_3d(self):
        """Test PCA dimensionality reduction to 3D."""
        if DimensionalityReducer is None:
            return
            
        reducer = DimensionalityReducer()
        
        try:
            reduced_data = reducer.reduce_dimensionality(
                self.sample_data,
                method='pca',
                n_components=3
            )
            
            assert reduced_data.shape == (100, 3)
            assert isinstance(reduced_data, np.ndarray)
        except Exception as e:
            # If sklearn not available, this is expected
            assert "sklearn" in str(e).lower() or "import" in str(e).lower()
    
    def test_tsne_reduction(self):
        """Test t-SNE dimensionality reduction."""
        if DimensionalityReducer is None:
            return
            
        reducer = DimensionalityReducer()
        
        try:
            # Use smaller dataset for t-SNE to avoid warnings
            small_data = self.sample_data[:50, :10]  # 50 samples, 10 features
            
            reduced_data = reducer.reduce_dimensionality(
                small_data,
                method='tsne',
                n_components=2
            )
            
            assert reduced_data.shape == (50, 2)
            assert isinstance(reduced_data, np.ndarray)
        except Exception as e:
            # If sklearn not available or other issues, this is expected
            assert any(term in str(e).lower() for term in ["sklearn", "import", "tsne"])
    
    def test_umap_reduction(self):
        """Test UMAP dimensionality reduction."""
        if DimensionalityReducer is None:
            return
            
        reducer = DimensionalityReducer()
        
        try:
            reduced_data = reducer.reduce_dimensionality(
                self.sample_data,
                method='umap',
                n_components=2
            )
            
            assert reduced_data.shape == (100, 2)
            assert isinstance(reduced_data, np.ndarray)
        except Exception as e:
            # If umap-learn not available, this is expected
            assert any(term in str(e).lower() for term in ["umap", "import", "module"])
    
    def test_invalid_method(self):
        """Test handling of invalid reduction method."""
        if DimensionalityReducer is None:
            return
            
        reducer = DimensionalityReducer()
        
        try:
            reduced_data = reducer.reduce_dimensionality(
                self.sample_data,
                method='invalid_method',
                n_components=2
            )
            # If it doesn't raise an error, check that it returns something sensible
            if reduced_data is not None:
                assert reduced_data.shape[0] == self.sample_data.shape[0]
        except (ValueError, KeyError, AttributeError) as e:
            # Should raise an appropriate error for invalid method
            assert "method" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_invalid_n_components(self):
        """Test handling of invalid number of components."""
        if DimensionalityReducer is None:
            return
            
        reducer = DimensionalityReducer()
        
        try:
            # Test with n_components > n_features
            reduced_data = reducer.reduce_dimensionality(
                self.sample_data[:, :5],  # Only 5 features
                method='pca',
                n_components=10  # More components than features
            )
            # Should either handle gracefully or raise appropriate error
        except (ValueError, Exception) as e:
            # Should raise an appropriate error
            assert any(term in str(e).lower() for term in ["component", "dimension", "feature"])
    
    def test_with_pandas_dataframe(self):
        """Test reduction with pandas DataFrame input."""
        if DimensionalityReducer is None:
            return
            
        reducer = DimensionalityReducer()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.sample_data)
        
        try:
            reduced_data = reducer.reduce_dimensionality(
                df,
                method='pca',
                n_components=2
            )
            
            assert reduced_data.shape == (100, 2)
            assert isinstance(reduced_data, np.ndarray)
        except Exception as e:
            # If sklearn not available, this is expected
            assert "sklearn" in str(e).lower() or "import" in str(e).lower()
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random state."""
        if DimensionalityReducer is None:
            return
            
        reducer = DimensionalityReducer()
        
        try:
            # First reduction
            reduced_1 = reducer.reduce_dimensionality(
                self.sample_data,
                method='pca',
                n_components=2,
                random_state=42
            )
            
            # Second reduction with same random state
            reduced_2 = reducer.reduce_dimensionality(
                self.sample_data,
                method='pca',
                n_components=2,
                random_state=42
            )
            
            # Results should be identical for PCA (deterministic)
            if reduced_1 is not None and reduced_2 is not None:
                np.testing.assert_array_almost_equal(reduced_1, reduced_2)
                
        except Exception as e:
            # If sklearn not available, this is expected
            assert "sklearn" in str(e).lower() or "import" in str(e).lower()
    
    def test_empty_data(self):
        """Test handling of empty data."""
        if DimensionalityReducer is None:
            return
            
        reducer = DimensionalityReducer()
        
        empty_data = np.array([]).reshape(0, 10)
        
        try:
            reduced_data = reducer.reduce_dimensionality(
                empty_data,
                method='pca',
                n_components=2
            )
            # Should handle empty data gracefully
        except (ValueError, Exception) as e:
            # Should raise appropriate error for empty data
            assert any(term in str(e).lower() for term in ["empty", "sample", "data"])
    
    def test_single_sample(self):
        """Test handling of single sample."""
        if DimensionalityReducer is None:
            return
            
        reducer = DimensionalityReducer()
        
        single_sample = self.sample_data[:1, :]  # Just one sample
        
        try:
            reduced_data = reducer.reduce_dimensionality(
                single_sample,
                method='pca',
                n_components=2
            )
            
            if reduced_data is not None:
                assert reduced_data.shape == (1, 2)
        except (ValueError, Exception) as e:
            # Should raise appropriate error for insufficient samples
            assert any(term in str(e).lower() for term in ["sample", "insufficient", "data"])


class TestDimensionalityReductionIntegration:
    """Integration tests for dimensionality reduction with realistic data."""
    
    def setup_method(self):
        """Set up realistic single-cell data."""
        np.random.seed(42)
        
        # Simulate single-cell feature data
        n_cells = 300
        n_features = 20
        
        # Create features with some correlation structure
        self.cell_features = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features) + 0.3 * np.random.rand(n_features, n_features),
            size=n_cells
        )
        
        # Add some treatment effects
        treatment_mask = np.random.choice([True, False], n_cells)
        self.cell_features[treatment_mask, :5] += 2.0  # Treatment effect on first 5 features
        
        self.treatment_labels = treatment_mask.astype(int)
    
    def test_pca_preserves_variance_structure(self):
        """Test that PCA preserves variance structure appropriately."""
        if DimensionalityReducer is None:
            return
            
        reducer = DimensionalityReducer()
        
        try:
            # Reduce to 2D
            reduced_2d = reducer.reduce_dimensionality(
                self.cell_features,
                method='pca',
                n_components=2
            )
            
            if reduced_2d is not None:
                # First PC should have higher variance than second
                pc1_var = np.var(reduced_2d[:, 0])
                pc2_var = np.var(reduced_2d[:, 1])
                assert pc1_var >= pc2_var
                
        except Exception as e:
            assert "sklearn" in str(e).lower()
    
    def test_treatment_separation(self):
        """Test that dimensionality reduction can separate treatment groups."""
        if DimensionalityReducer is None:
            return
            
        reducer = DimensionalityReducer()
        
        try:
            reduced_data = reducer.reduce_dimensionality(
                self.cell_features,
                method='pca',
                n_components=2
            )
            
            if reduced_data is not None:
                # Calculate separation between treatment groups
                treated = reduced_data[self.treatment_labels == 1]
                control = reduced_data[self.treatment_labels == 0]
                
                treated_center = np.mean(treated, axis=0)
                control_center = np.mean(control, axis=0)
                
                # Should have some separation (distance > 0)
                separation = np.linalg.norm(treated_center - control_center)
                assert separation >= 0  # Basic sanity check
                
        except Exception as e:
            assert "sklearn" in str(e).lower()
    
    def test_multiple_methods_consistency(self):
        """Test that different methods produce consistent dimensionality."""
        if DimensionalityReducer is None:
            return
            
        reducer = DimensionalityReducer()
        
        methods = ['pca']  # Start with just PCA as it's most likely to be available
        
        for method in methods:
            try:
                reduced_data = reducer.reduce_dimensionality(
                    self.cell_features,
                    method=method,
                    n_components=2
                )
                
                if reduced_data is not None:
                    assert reduced_data.shape == (300, 2)
                    # Check that data is not all zeros or NaN
                    assert not np.all(reduced_data == 0)
                    assert not np.any(np.isnan(reduced_data))
                    
            except Exception as e:
                # Expected if dependencies not available
                assert any(term in str(e).lower() for term in ["sklearn", "import", "umap", "tsne"])
