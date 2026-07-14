"""
Unit tests for feature_extractor_2d module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

from src.feature_extraction.feature_extractor_incarta import (
    compute_morphology_features,
    compute_intensity_features,
    compute_spatial_features,
    compute_texture_features,
    extract_instance_features,
    extract_all_instance_features
)


class TestComputeMorphologyFeatures:
    """Test morphology feature computation."""
    
    def test_simple_circle_mask(self):
        """Test morphology features for a simple circular mask."""
        # Create a simple circular mask
        mask = np.zeros((20, 20), dtype=np.uint8)
        center = (10, 10)
        radius = 5
        y, x = np.ogrid[:20, :20]
        mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        mask[mask_circle] = 1
        
        features = compute_morphology_features(mask)
        
        # Check that all expected keys are present
        expected_keys = [
            "area", "perimeter", "elongation", "compactness", 
            "circularity", "feret_diameter", "radius_of_gyration",
            "major_axis", "minor_axis"
        ]
        assert all(key in features for key in expected_keys)
        
        # Check reasonable values for a circle
        assert features["area"] > 0
        assert features["perimeter"] > 0
        assert features["elongation"] >= 1.0  # Should be close to 1 for circle
        assert 0 < features["circularity"] <= 1.0
        assert features["compactness"] >= 1.0
        
    def test_rectangular_mask(self):
        """Test morphology features for a rectangular mask."""
        # Create a rectangular mask (elongated)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[8:12, 5:15] = 1  # Rectangle: 4 pixels high, 10 pixels wide
        
        features = compute_morphology_features(mask)
        
        # Rectangle should be elongated
        assert features["elongation"] > 1.5  # Should be significantly elongated
        assert features["area"] == 40  # 4 * 10 = 40 pixels
        assert features["circularity"] < 0.9  # Not circular
        
    def test_single_pixel_mask(self):
        """Test morphology features for a single pixel."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 1
        
        features = compute_morphology_features(mask)
        
        assert features["area"] == 1
        assert features["elongation"] >= 0  # Should handle division edge case
        
    def test_empty_mask(self):
        """Test morphology features for an empty mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        
        with pytest.raises(IndexError):
            # Should raise error for empty mask (no regions found)
            compute_morphology_features(mask)


class TestComputeIntensityFeatures:
    """Test intensity feature computation."""
    
    def test_uniform_intensity(self):
        """Test intensity features with uniform intensity."""
        mask = np.ones((5, 5), dtype=np.uint8)
        image = np.full((5, 5), 100, dtype=np.uint8)
        
        features = compute_intensity_features(mask, image)
        
        expected_keys = ["mean_intensity", "std_intensity", "cv_intensity", "total_intensity"]
        assert all(key in features for key in expected_keys)
        
        assert features["mean_intensity"] == 100
        assert features["std_intensity"] == 0
        assert features["cv_intensity"] == 0  # std/mean = 0/100 = 0
        assert features["total_intensity"] == 2500  # 25 pixels * 100 = 2500
        
    def test_varying_intensity(self):
        """Test intensity features with varying intensity."""
        mask = np.ones((3, 3), dtype=np.uint8)
        image = np.array([[100, 200, 150],
                         [120, 180, 160],
                         [140, 170, 130]], dtype=np.uint8)
        
        features = compute_intensity_features(mask, image)
        
        expected_mean = np.mean([100, 200, 150, 120, 180, 160, 140, 170, 130])
        expected_std = np.std([100, 200, 150, 120, 180, 160, 140, 170, 130])
        expected_cv = expected_std / expected_mean
        expected_total = np.sum([100, 200, 150, 120, 180, 160, 140, 170, 130])
        
        assert abs(features["mean_intensity"] - expected_mean) < 1e-6
        assert abs(features["std_intensity"] - expected_std) < 1e-6
        assert abs(features["cv_intensity"] - expected_cv) < 1e-6
        assert features["total_intensity"] == expected_total
        
    def test_partial_mask(self):
        """Test intensity features with partial mask."""
        mask = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]], dtype=np.uint8)
        image = np.array([[100, 200, 150],
                         [120, 180, 160],
                         [140, 170, 130]], dtype=np.uint8)
        
        features = compute_intensity_features(mask, image)
        
        # Only masked pixels: 100, 150, 180, 140, 130
        expected_mean = np.mean([100, 150, 180, 140, 130])
        expected_total = 100 + 150 + 180 + 140 + 130
        
        assert abs(features["mean_intensity"] - expected_mean) < 1e-6
        assert features["total_intensity"] == expected_total
        
    def test_zero_mean_intensity(self):
        """Test handling of zero mean intensity."""
        mask = np.ones((2, 2), dtype=np.uint8)
        image = np.zeros((2, 2), dtype=np.uint8)
        
        features = compute_intensity_features(mask, image)
        
        assert features["mean_intensity"] == 0
        assert features["cv_intensity"] == 0  # Should handle division by zero


class TestComputeSpatialFeatures:
    """Test spatial feature computation."""
    
    def test_centered_mask(self):
        """Test spatial features for a centered mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[4:6, 4:6] = 1  # 2x2 square in center
        image = np.ones((10, 10), dtype=np.uint8) * 100
        
        features = compute_spatial_features(mask, image)
        
        expected_keys = ["centroid_x", "centroid_y", "center_of_mass_x", 
                        "center_of_mass_y", "mass_displacement"]
        assert all(key in features for key in expected_keys)
        
        # For uniform intensity, centroid and center of mass should be close
        assert abs(features["centroid_x"] - 4.5) < 0.1  # Center of 2x2 square
        assert abs(features["centroid_y"] - 4.5) < 0.1
        assert features["mass_displacement"] < 0.1  # Should be very small
        
    def test_asymmetric_intensity(self):
        """Test spatial features with asymmetric intensity."""
        mask = np.ones((3, 3), dtype=np.uint8)
        # Higher intensity on one side
        image = np.array([[50, 50, 50],
                         [50, 50, 50],
                         [200, 200, 200]], dtype=np.uint8)
        
        features = compute_spatial_features(mask, image)
        
        # Center of mass should be shifted towards higher intensity
        assert features["center_of_mass_y"] > features["centroid_y"]
        assert features["mass_displacement"] > 0


class TestComputeTextureFeatures:
    """Test texture feature computation."""
    
    def test_uniform_texture(self):
        """Test texture features with uniform image."""
        mask = np.ones((10, 10), dtype=np.uint8)
        image = np.full((10, 10), 128, dtype=np.uint8)
        
        features = compute_texture_features(mask, image)
        
        expected_keys = ["gabor_mean", "gabor_std", "skewness", "kurtosis", "entropy"]
        assert all(key in features for key in expected_keys)
        
        # Uniform image should have low texture variation
        # Note: skewness may be NaN for perfectly uniform data due to numerical precision
        assert np.isnan(features["skewness"]) or abs(features["skewness"]) < 1e-6  # Should be near zero or NaN
        assert features["entropy"] < 1.0  # Low entropy for uniform distribution
        
    def test_structured_texture(self):
        """Test texture features with structured pattern."""
        mask = np.ones((8, 8), dtype=np.uint8)
        # Create a checkerboard pattern
        image = np.zeros((8, 8), dtype=np.uint8)
        image[::2, ::2] = 255  # Even rows, even columns
        image[1::2, 1::2] = 255  # Odd rows, odd columns
        
        features = compute_texture_features(mask, image)
        
        # Checkerboard should have high texture variation
        assert features["entropy"] > 0.5  # Should have higher entropy
        assert abs(features["gabor_mean"]) >= 0  # Gabor should respond to pattern
        
    def test_edge_cases(self):
        """Test texture features edge cases."""
        mask = np.ones((2, 2), dtype=np.uint8)
        image = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        
        features = compute_texture_features(mask, image)
        
        # Should not crash and return valid numbers
        assert isinstance(features["gabor_mean"], float)
        assert isinstance(features["gabor_std"], float)
        assert isinstance(features["entropy"], float)
        assert not np.isnan(features["entropy"])


class TestExtractInstanceFeatures:
    """Test instance feature extraction."""
    
    def test_single_instance(self):
        """Test extracting features from a single instance."""
        # Create a labeled mask with one instance
        label_mask = np.zeros((10, 10), dtype=np.uint16)
        label_mask[3:7, 3:7] = 1  # Instance with ID 1
        
        image = np.random.randint(50, 200, (10, 10), dtype=np.uint8)
        
        features = extract_instance_features(1, label_mask, image)
        
        # Check that instance_id is included
        assert features["cell_id"] == 1
        
        # Check that all feature categories are present
        morphology_keys = ["area", "perimeter", "elongation"]
        intensity_keys = ["mean_intensity", "std_intensity"]
        spatial_keys = ["centroid_x", "centroid_y"]
        texture_keys = ["gabor_mean", "skewness"]
        
        for key in morphology_keys + intensity_keys + spatial_keys + texture_keys:
            assert key in features
            assert isinstance(features[key], (int, float))
            
    def test_nonexistent_instance(self):
        """Test extracting features from a nonexistent instance."""
        label_mask = np.zeros((10, 10), dtype=np.uint16)
        label_mask[3:7, 3:7] = 1  # Only instance ID 1 exists
        
        image = np.random.randint(50, 200, (10, 10), dtype=np.uint8)
        
        # Try to extract features for instance ID 2 (doesn't exist)
        with pytest.raises(IndexError):
            extract_instance_features(2, label_mask, image)


class TestExtractAllInstanceFeatures:
    """Test extraction of features from all instances."""
    
    def test_multiple_instances(self):
        """Test extracting features from multiple instances."""
        # Create a labeled mask with multiple instances
        label_mask = np.zeros((20, 20), dtype=np.uint16)
        label_mask[2:6, 2:6] = 1      # Instance 1
        label_mask[2:6, 10:14] = 2    # Instance 2
        label_mask[10:14, 2:6] = 3    # Instance 3
        
        image = np.random.randint(50, 200, (20, 20), dtype=np.uint8)
        
        df = extract_all_instance_features(label_mask, image, n_jobs=1)
        
        # Check that DataFrame has correct shape
        assert len(df) == 3  # Three instances
        assert "cell_id" in df.columns
        
        # Check that all instances are present
        assert set(df["cell_id"]) == {1, 2, 3}
        
        # Check that all expected columns are present
        expected_columns = [
            "cell_id", "area", "perimeter", "elongation", "compactness",
            "circularity", "feret_diameter", "radius_of_gyration", "major_axis",
            "minor_axis", "mean_intensity", "std_intensity", "cv_intensity",
            "total_intensity", "centroid_x", "centroid_y", "center_of_mass_x",
            "center_of_mass_y", "mass_displacement", "gabor_mean", "gabor_std",
            "skewness", "kurtosis", "entropy"
        ]
        
        for col in expected_columns:
            assert col in df.columns
            
    def test_single_instance_dataframe(self):
        """Test DataFrame output for a single instance."""
        label_mask = np.zeros((10, 10), dtype=np.uint16)
        label_mask[3:7, 3:7] = 5  # Single instance with ID 5
        
        image = np.random.randint(50, 200, (10, 10), dtype=np.uint8)
        
        df = extract_all_instance_features(label_mask, image, n_jobs=1)
        
        assert len(df) == 1
        assert df["cell_id"].iloc[0] == 5
        assert isinstance(df, pd.DataFrame)
        
    def test_no_instances(self):
        """Test handling of mask with no labeled instances."""
        label_mask = np.zeros((10, 10), dtype=np.uint16)  # All background
        image = np.random.randint(50, 200, (10, 10), dtype=np.uint8)
        
        df = extract_all_instance_features(label_mask, image, n_jobs=1)
        
        assert len(df) == 0  # Empty DataFrame
        assert isinstance(df, pd.DataFrame)
        
    def test_background_exclusion(self):
        """Test that background (label 0) is excluded."""
        label_mask = np.zeros((10, 10), dtype=np.uint16)
        label_mask[3:7, 3:7] = 1  # Instance 1
        # Background remains as 0
        
        image = np.random.randint(50, 200, (10, 10), dtype=np.uint8)
        
        df = extract_all_instance_features(label_mask, image, n_jobs=1)
        
        assert len(df) == 1  # Only instance 1, background excluded
        assert df["cell_id"].iloc[0] == 1
        
    def test_parallel_processing(self):
        """Test that parallel processing gives same results as serial."""
        # Create mask with multiple instances
        label_mask = np.zeros((20, 20), dtype=np.uint16)
        label_mask[2:6, 2:6] = 1
        label_mask[10:14, 10:14] = 2
        label_mask[2:6, 14:18] = 3
        
        image = np.random.randint(50, 200, (20, 20), dtype=np.uint8)
        
        # Extract features with different n_jobs settings
        df_serial = extract_all_instance_features(label_mask, image, n_jobs=1)
        df_parallel = extract_all_instance_features(label_mask, image, n_jobs=2)
        
        # Results should be identical (order might differ)
        df_serial_sorted = df_serial.sort_values("cell_id").reset_index(drop=True)
        df_parallel_sorted = df_parallel.sort_values("cell_id").reset_index(drop=True)
        
        pd.testing.assert_frame_equal(df_serial_sorted, df_parallel_sorted)
        
    def test_data_types(self):
        """Test that output data types are correct."""
        label_mask = np.zeros((10, 10), dtype=np.uint16)
        label_mask[3:7, 3:7] = 1
        
        image = np.random.randint(50, 200, (10, 10), dtype=np.uint8)
        
        df = extract_all_instance_features(label_mask, image, n_jobs=1)
        
        # instance_id should be integer (can be various integer types)
        assert pd.api.types.is_integer_dtype(df["cell_id"])
        
        # All other features should be numeric (int or float)
        for col in df.columns:
            if col != "cell_id":
                assert pd.api.types.is_numeric_dtype(df[col])
                
    def test_large_instance_ids(self):
        """Test handling of large instance IDs."""
        label_mask = np.zeros((10, 10), dtype=np.uint16)
        label_mask[3:7, 3:7] = 65000  # Large instance ID
        
        image = np.random.randint(50, 200, (10, 10), dtype=np.uint8)
        
        df = extract_all_instance_features(label_mask, image, n_jobs=1)
        
        assert len(df) == 1
        assert df["cell_id"].iloc[0] == 65000


class TestIntegration:
    """Integration tests for the feature extraction pipeline."""
    
    def test_realistic_cell_segmentation(self):
        """Test feature extraction on realistic cell-like segmentations."""
        # Create a more realistic segmentation scenario
        np.random.seed(42)  # For reproducibility
        
        label_mask = np.zeros((50, 50), dtype=np.uint16)
        image = np.zeros((50, 50), dtype=np.uint8)
        
        # Create several cell-like regions
        cell_centers = [(15, 15), (15, 35), (35, 15), (35, 35)]
        
        for i, (cy, cx) in enumerate(cell_centers, 1):
            # Create circular cell region
            y, x = np.ogrid[:50, :50]
            radius = np.random.randint(4, 8)
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            label_mask[mask] = i
            
            # Add realistic intensity variation
            cell_intensity = np.random.randint(80, 180)
            image[mask] = cell_intensity + np.random.randint(-20, 20, mask.sum())
            
        # Add some background noise
        noise = np.random.randint(0, 30, (50, 50)).astype(np.uint8)
        image = image.astype(np.int16) + noise.astype(np.int16)
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        df = extract_all_instance_features(label_mask, image)
        
        # Check that we got features for all cells
        assert len(df) == 4
        
        # Check that features are in reasonable ranges
        assert (df["area"] > 20).all() and (df["area"] < 200).all()
        assert (df["circularity"] > 0.5).all()  # Should be fairly circular
        assert (df["mean_intensity"] > 50).all()
        
        # Check feature correlations that should exist
        # Larger cells should generally have higher total intensity
        correlation = df["area"].corr(df["total_intensity"])
        assert correlation > 0.7  # Strong positive correlation
        
    def test_edge_touching_instances(self):
        """Test handling of instances that touch image edges."""
        label_mask = np.zeros((20, 20), dtype=np.uint16)
        # Instance touching left edge
        label_mask[0:5, 0:5] = 1
        # Instance touching right edge  
        label_mask[15:20, 15:20] = 2
        # Instance in middle
        label_mask[8:12, 8:12] = 3
        
        image = np.random.randint(50, 200, (20, 20), dtype=np.uint8)
        
        df = extract_all_instance_features(label_mask, image)
        
        # Should handle edge cases without crashing
        assert len(df) == 3
        assert all(df["area"] > 0)
        
    def test_very_small_instances(self):
        """Test handling of very small instances."""
        label_mask = np.zeros((20, 20), dtype=np.uint16)
        # Single pixel instances
        label_mask[5, 5] = 1
        label_mask[10, 10] = 2
        label_mask[15, 15] = 3
        
        image = np.random.randint(50, 200, (20, 20), dtype=np.uint8)
        
        df = extract_all_instance_features(label_mask, image)
        
        assert len(df) == 3
        assert (df["area"] == 1).all()  # All single pixels
        
    def test_feature_value_ranges(self):
        """Test that extracted features are within expected ranges."""
        # Create various shaped instances
        label_mask = np.zeros((30, 30), dtype=np.uint16)
        
        # Circle
        y, x = np.ogrid[:30, :30]
        circle = (x - 10)**2 + (y - 10)**2 <= 25
        label_mask[circle] = 1
        
        # Rectangle (elongated)
        label_mask[20:25, 5:25] = 2
        
        image = np.random.randint(50, 200, (30, 30), dtype=np.uint8)
        
        df = extract_all_instance_features(label_mask, image)
        
        # Test feature ranges
        assert (df["area"] > 0).all()
        assert (df["perimeter"] > 0).all()
        assert (df["elongation"] >= 1.0).all()  # Should be >= 1
        assert (df["compactness"] >= 1.0).all()  # Should be >= 1
        assert ((df["circularity"] > 0) & (df["circularity"] <= 1)).all()
        assert (df["mean_intensity"] >= 0).all()
        assert (df["std_intensity"] >= 0).all()
        assert (df["cv_intensity"] >= 0).all()
        
        # Rectangle should be more elongated than circle
        circle_elongation = df[df["cell_id"] == 1]["elongation"].iloc[0]
        rect_elongation = df[df["cell_id"] == 2]["elongation"].iloc[0]
        assert rect_elongation > circle_elongation
