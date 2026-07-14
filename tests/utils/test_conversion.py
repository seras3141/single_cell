"""
Unit tests for conversion module (utils).
Tests for combining 2D images to 3D and splitting 3D images to 2D.
"""

import importlib.util as _ilu

import pytest
import numpy as np
import tempfile
import os
import re
from pathlib import Path
import tifffile as tiff
import shutil

from src.utils.conversion import combine_2d_to_3d, split_3d_to_2d

_ZARR_AVAILABLE = _ilu.find_spec("zarr") is not None
_H5PY_AVAILABLE = _ilu.find_spec("h5py") is not None
_skip_no_zarr = pytest.mark.skipif(not _ZARR_AVAILABLE, reason="zarr not installed")
_skip_no_h5py = pytest.mark.skipif(not _H5PY_AVAILABLE, reason="h5py not installed")


class TestCombine2DTo3D:
    """Test cases for combine_2d_to_3d function."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directories for testing."""
        temp_dir = tempfile.mkdtemp()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        yield {"temp": temp_dir, "input": input_dir, "output": output_dir}
        
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_2d_images(self, temp_dir):
        """Create sample 2D TIFF images for testing."""
        input_dir = temp_dir["input"]
        
        # Create test images with different z-slices and suffixes
        np.random.seed(42)
        
        # Sample A with BF suffix (3 z-slices)
        for z in range(1, 4):
            img_data = np.random.randint(0, 255, (64, 64), dtype=np.uint8) + z * 10
            tiff.imwrite(str(input_dir / f"sample_A_z{z}_BF.tif"), img_data)
        
        # Sample A with Cells suffix (3 z-slices) — values > 255 to require uint16 after dtype normalisation
        for z in range(1, 4):
            img_data = np.random.randint(256, 1000, (64, 64), dtype=np.uint16) + z * 5
            tiff.imwrite(str(input_dir / f"sample_A_z{z}_Cells.tif"), img_data)
        
        # Sample B with BF suffix (2 z-slices)
        for z in range(1, 3):
            img_data = np.random.randint(0, 255, (32, 32), dtype=np.uint8) + z * 20
            tiff.imwrite(str(input_dir / f"sample_B_z{z}_BF.tif"), img_data)
        
        # Sample without suffix (2 z-slices)
        for z in range(1, 3):
            img_data = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
            tiff.imwrite(str(input_dir / f"no_suffix_z{z}.tif"), img_data)
        
        return temp_dir

    def test_combine_basic_functionality(self, sample_2d_images):
        """Test basic combining functionality with standard pattern."""
        input_dir = sample_2d_images["input"]
        output_dir = sample_2d_images["output"]
        
        combine_2d_to_3d(input_dir, output_dir)
        
        # Check that 3D files were created
        output_files = list(output_dir.glob("*.tif"))
        output_names = [f.name for f in output_files]
        
        assert len(output_files) == 4  # sample_A_BF, sample_A_Cells, sample_B_BF, no_suffix
        assert "sample_A_BF_3d.tif" in output_names
        assert "sample_A_Cells_3d.tif" in output_names
        assert "sample_B_BF_3d.tif" in output_names
        assert "no_suffix_3d.tif" in output_names

    def test_3d_volume_properties(self, sample_2d_images):
        """Test that 3D volumes have correct dimensions and data."""
        input_dir = sample_2d_images["input"]
        output_dir = sample_2d_images["output"]
        
        combine_2d_to_3d(input_dir, output_dir)
        
        # Check sample_A_BF_3d.tif properties
        bf_3d = tiff.imread(str(output_dir / "sample_A_BF_3d.tif"))
        assert bf_3d.shape == (3, 64, 64)  # 3 z-slices, 64x64 each
        assert bf_3d.dtype == np.uint8
        
        # Check sample_A_Cells_3d.tif properties
        cells_3d = tiff.imread(str(output_dir / "sample_A_Cells_3d.tif"))
        assert cells_3d.shape == (3, 64, 64)
        assert cells_3d.dtype == np.uint16
        
        # Check sample_B_BF_3d.tif properties
        bf_b_3d = tiff.imread(str(output_dir / "sample_B_BF_3d.tif"))
        assert bf_b_3d.shape == (2, 32, 32)  # 2 z-slices, 32x32 each

    def test_z_order_preservation(self, sample_2d_images):
        """Test that z-order is preserved correctly."""
        input_dir = sample_2d_images["input"]
        output_dir = sample_2d_images["output"]
        
        combine_2d_to_3d(input_dir, output_dir)
        
        # Check that z-order is preserved (we added z*10 to each slice)
        bf_3d = tiff.imread(str(output_dir / "sample_A_BF_3d.tif"))
        
        # First slice should have lower average values than last slice
        assert np.mean(bf_3d[0]) < np.mean(bf_3d[2])

    def test_custom_pattern(self, temp_dir):
        """Test with custom regex pattern."""
        input_dir = temp_dir["input"]
        output_dir = temp_dir["output"]
        
        # Create files with different naming pattern
        np.random.seed(42)
        for z in range(1, 4):
            img_data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
            tiff.imwrite(str(input_dir / f"test-slice{z:02d}-ch1.tif"), img_data)
        
        # Custom pattern to match test-sliceXX-ch1.tif
        custom_pattern = r"(test)-slice(\d+)-(ch1)\.(tif|tiff)"
        combine_2d_to_3d(input_dir, output_dir, pattern=custom_pattern)
        
        output_files = list(output_dir.glob("*.tif"))
        assert len(output_files) == 1
        assert "test_ch1_3d.tif" in [f.name for f in output_files]

    def test_recursive_search(self, temp_dir):
        """Test recursive directory search."""
        input_dir = temp_dir["input"]
        output_dir = temp_dir["output"]
        
        # Create subdirectory with images
        subdir = input_dir / "subdir"
        subdir.mkdir()
        
        np.random.seed(42)
        for z in range(1, 3):
            img_data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
            tiff.imwrite(str(subdir / f"sub_sample_z{z}_BF.tif"), img_data)
        
        # Test without recursive (should find no files)
        combine_2d_to_3d(input_dir, output_dir, recursive=False)
        assert len(list(output_dir.glob("*.tif"))) == 0
        
        # Test with recursive (should find files)
        combine_2d_to_3d(input_dir, output_dir, recursive=True)
        output_files = list(output_dir.glob("*.tif"))
        assert len(output_files) == 1
        assert "sub_sample_BF_3d.tif" in [f.name for f in output_files]

    def test_empty_directory(self, temp_dir):
        """Test behavior with empty input directory."""
        input_dir = temp_dir["input"]
        output_dir = temp_dir["output"]
        
        # Should handle empty directory gracefully
        combine_2d_to_3d(input_dir, output_dir)
        assert len(list(output_dir.glob("*.tif"))) == 0

    def test_nonexistent_input_directory(self, temp_dir):
        """Test behavior with non-existent input directory."""
        nonexistent_dir = temp_dir["temp"] + "/nonexistent"
        output_dir = temp_dir["output"]
        
        with pytest.raises(AssertionError, match="Input directory .* does not exist"):
            combine_2d_to_3d(nonexistent_dir, output_dir)

    def test_output_directory_creation(self, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        input_dir = temp_dir["input"]
        output_dir = Path(temp_dir["temp"]) / "new_output"
        
        # Create a sample file
        np.random.seed(42)
        img_data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        tiff.imwrite(str(input_dir / "test_z1_BF.tif"), img_data)
        
        assert not output_dir.exists()
        combine_2d_to_3d(input_dir, output_dir)
        assert output_dir.exists()


class TestSplit3DTo2D:
    """Test cases for split_3d_to_2d function."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directories for testing."""
        temp_dir = tempfile.mkdtemp()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        yield {"temp": temp_dir, "input": input_dir, "output": output_dir}
        
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_3d_image(self, temp_dir):
        """Create a sample 3D TIFF image for testing."""
        input_dir = temp_dir["input"]
        
        # Create 3D volume (5 slices, 32x32 each)
        np.random.seed(42)
        volume = np.random.randint(0, 255, (5, 32, 32), dtype=np.uint8)
        
        # Add distinct values to each z-slice for verification
        for z in range(5):
            volume[z] += z * 10
        
        input_path = str(input_dir / "sample_3d.tif")
        tiff.imwrite(input_path, volume)
        
        return input_path, volume

    def test_split_basic_functionality(self, sample_3d_image, temp_dir):
        """Test basic splitting functionality."""
        input_path, original_volume = sample_3d_image
        output_dir = str(temp_dir["output"])
        
        split_3d_to_2d(input_path, output_dir)
        
        # Check that 2D files were created
        output_files = sorted(list(Path(output_dir).glob("*.tif")))
        assert len(output_files) == 5
        
        expected_names = [f"sample_z{i+1}.tif" for i in range(5)]
        actual_names = [f.name for f in output_files]
        assert actual_names == expected_names

    def test_split_with_suffix(self, sample_3d_image, temp_dir):
        """Test splitting with custom suffix."""
        input_path, original_volume = sample_3d_image
        output_dir = str(temp_dir["output"])
        
        split_3d_to_2d(input_path, output_dir, suffix="BF")
        
        # Check filenames with suffix
        output_files = sorted(list(Path(output_dir).glob("*.tif")))
        expected_names = [f"sample_z{i+1}_BF.tif" for i in range(5)]
        actual_names = [f.name for f in output_files]
        assert actual_names == expected_names

    def test_split_data_integrity(self, sample_3d_image, temp_dir):
        """Test that split data matches original volume."""
        input_path, original_volume = sample_3d_image
        output_dir = str(temp_dir["output"])
        
        split_3d_to_2d(input_path, output_dir)
        
        # Verify each slice matches original
        for z in range(5):
            slice_path = Path(output_dir) / f"sample_z{z+1}.tif"
            slice_data = tiff.imread(str(slice_path))
            
            np.testing.assert_array_equal(slice_data, original_volume[z])

    def test_different_input_formats(self, temp_dir):
        """Test with different input file formats and names."""
        input_dir = temp_dir["input"]
        output_dir = str(temp_dir["output"])
        
        # Test with different filename (already ends with _3d)
        np.random.seed(42)
        volume = np.random.randint(0, 255, (3, 16, 16), dtype=np.uint16)
        input_path = str(input_dir / "test_volume_3d.tif")
        tiff.imwrite(input_path, volume)
        
        split_3d_to_2d(input_path, output_dir, suffix="Cells")
        
        output_files = sorted(list(Path(output_dir).glob("*.tif")))
        expected_names = [f"test_volume_z{i+1}_Cells.tif" for i in range(3)]
        actual_names = [f.name for f in output_files]
        assert actual_names == expected_names

    def test_output_directory_creation(self, sample_3d_image, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        input_path, original_volume = sample_3d_image
        output_dir = str(Path(temp_dir["temp"]) / "new_output")
        
        assert not Path(output_dir).exists()
        split_3d_to_2d(input_path, output_dir)
        assert Path(output_dir).exists()
        
        # Verify files were created
        output_files = list(Path(output_dir).glob("*.tif"))
        assert len(output_files) == 5

    def test_nonexistent_input_file(self, temp_dir):
        """Test behavior with non-existent input file."""
        nonexistent_file = str(Path(temp_dir["temp"]) / "nonexistent.tif")
        output_dir = str(temp_dir["output"])
        
        # Should raise an error when trying to read non-existent file
        with pytest.raises(FileNotFoundError):
            split_3d_to_2d(nonexistent_file, output_dir)

    @_skip_no_zarr
    def test_split_zarr_input(self, temp_dir):
        """split_3d_to_2d should accept a .zarr volume via load_labels."""
        import zarr

        input_dir = temp_dir["input"]
        output_dir = str(temp_dir["output"])

        np.random.seed(0)
        volume = np.random.randint(0, 100, (3, 16, 16), dtype=np.uint16)
        zarr_path = str(input_dir / "sample_3d.zarr")
        z = zarr.open(zarr_path, mode="w", shape=volume.shape, dtype=volume.dtype)
        z[:] = volume

        split_3d_to_2d(zarr_path, output_dir)

        output_files = sorted(Path(output_dir).glob("*.tif"))
        assert len(output_files) == 3
        assert [f.name for f in output_files] == [f"sample_z{i+1}.tif" for i in range(3)]
        for z_idx, f in enumerate(output_files):
            np.testing.assert_array_equal(tiff.imread(str(f)), volume[z_idx])

    @_skip_no_h5py
    def test_split_hdf5_input(self, temp_dir):
        """split_3d_to_2d should accept a .h5 volume via load_labels."""
        import h5py

        input_dir = temp_dir["input"]
        output_dir = str(temp_dir["output"])

        np.random.seed(1)
        volume = np.random.randint(0, 100, (4, 16, 16), dtype=np.uint16)
        h5_path = str(input_dir / "sample_3d.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("labels", data=volume)

        split_3d_to_2d(h5_path, output_dir)

        output_files = sorted(Path(output_dir).glob("*.tif"))
        assert len(output_files) == 4
        assert [f.name for f in output_files] == [f"sample_z{i+1}.tif" for i in range(4)]
        for z_idx, f in enumerate(output_files):
            np.testing.assert_array_equal(tiff.imread(str(f)), volume[z_idx])

    @_skip_no_zarr
    def test_split_zarr_input_with_suffix(self, temp_dir):
        """Suffix is appended to each slice name when splitting a zarr volume."""
        import zarr

        input_dir = temp_dir["input"]
        output_dir = str(temp_dir["output"])

        volume = np.ones((2, 8, 8), dtype=np.uint16)
        zarr_path = str(input_dir / "sample_3d.zarr")
        z = zarr.open(zarr_path, mode="w", shape=volume.shape, dtype=volume.dtype)
        z[:] = volume

        split_3d_to_2d(zarr_path, output_dir, suffix="BF")

        names = {f.name for f in Path(output_dir).glob("*.tif")}
        assert names == {"sample_z1_BF.tif", "sample_z2_BF.tif"}

    def test_dtype_preservation(self, temp_dir):
        """Test that data type is preserved during splitting."""
        input_dir = temp_dir["input"]
        output_dir = str(temp_dir["output"])
        
        # Test with different dtypes
        dtypes_to_test = [np.uint8, np.uint16, np.float32]
        
        for dtype in dtypes_to_test:
            # Create volume with specific dtype
            volume = np.random.rand(2, 16, 16).astype(dtype)
            if dtype in [np.uint8, np.uint16]:
                volume = (volume * 255).astype(dtype)
            
            input_path = str(input_dir / f"test_{dtype.__name__}_3d.tif")
            tiff.imwrite(input_path, volume)
            
            split_3d_to_2d(input_path, output_dir)
            
            # Check that output slices have same dtype
            for z in range(2):
                slice_path = Path(output_dir) / f"test_{dtype.__name__}_z{z+1}.tif"
                slice_data = tiff.imread(str(slice_path))
                assert slice_data.dtype == dtype


class TestConversionIntegration:
    """Integration tests combining both functions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directories for testing."""
        temp_dir = tempfile.mkdtemp()
        input_dir = Path(temp_dir) / "input"
        temp_3d_dir = Path(temp_dir) / "temp_3d"
        final_2d_dir = Path(temp_dir) / "final_2d"
        input_dir.mkdir()
        temp_3d_dir.mkdir()
        final_2d_dir.mkdir()
        
        yield {
            "temp": temp_dir,
            "input": input_dir,
            "temp_3d": temp_3d_dir,
            "final_2d": final_2d_dir
        }
        
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    def test_round_trip_conversion(self, temp_dir):
        """Test converting 2D->3D->2D and verify data integrity."""
        input_dir = temp_dir["input"]
        temp_3d_dir = str(temp_dir["temp_3d"])
        final_2d_dir = str(temp_dir["final_2d"])
        
        # Create original 2D images with known data
        np.random.seed(42)
        original_slices = []
        for z in range(1, 4):
            img_data = np.random.randint(0, 255, (32, 32), dtype=np.uint8) + z * 5
            original_slices.append(img_data)
            tiff.imwrite(str(input_dir / f"test_z{z}_BF.tif"), img_data)
        
        # Convert to 3D
        combine_2d_to_3d(input_dir, temp_3d_dir)
        
        # Convert back to 2D
        volume_path = str(Path(temp_3d_dir) / "test_BF_3d.tif")
        split_3d_to_2d(volume_path, final_2d_dir, suffix="BF")
        
        # Verify round-trip integrity
        for z in range(3):
            final_slice_path = Path(final_2d_dir) / f"test_z{z+1}_BF.tif"
            final_slice = tiff.imread(str(final_slice_path))
            
            np.testing.assert_array_equal(final_slice, original_slices[z])

    def test_multiple_groups_round_trip(self, temp_dir):
        """Test round-trip with multiple image groups."""
        input_dir = temp_dir["input"]
        temp_3d_dir = str(temp_dir["temp_3d"])
        final_2d_dir = str(temp_dir["final_2d"])
        
        # Create multiple groups
        np.random.seed(42)
        groups = ["sampleA_BF", "sampleA_Cells", "sampleB_BF"]
        original_data = {}
        
        for group in groups:
            original_data[group] = []
            for z in range(1, 3):  # 2 slices each
                img_data = np.random.randint(0, 255, (16, 16), dtype=np.uint8)
                original_data[group].append(img_data)
                tiff.imwrite(str(input_dir / f"{group.replace('_', '_z' + str(z) + '_')}.tif"), img_data)
        
        # Convert to 3D
        combine_2d_to_3d(input_dir, temp_3d_dir)
        
        # Convert each 3D volume back to 2D
        for group in groups:
            volume_path = str(Path(temp_3d_dir) / f"{group}_3d.tif")
            split_3d_to_2d(volume_path, final_2d_dir)
        
        # Verify all groups maintained integrity
        for group in groups:
            for z in range(2):
                final_slice_path = Path(final_2d_dir) / f"{group}_z{z+1}.tif"
                final_slice = tiff.imread(str(final_slice_path))
                
                np.testing.assert_array_equal(final_slice, original_data[group][z])
