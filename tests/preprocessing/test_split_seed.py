import os
import tempfile
import pytest
from src.preprocessing.dataset_split import train_test_split_directory
from src.utils.file_utils import BF_IF_FileHandler


@pytest.fixture
def create_dummy_tiffs():
    def _create_dummy_tiffs(directory, n=10):
        os.makedirs(directory, exist_ok=True)
        for i in range(n):
            for z in range(5):
                mask_path = os.path.join(directory, f"Cells_R10-C{i:02d}-F0-Z{z}-T0.tif")
                with open(mask_path, 'wb') as f:
                    f.write(os.urandom(128))

                mask_path = os.path.join(directory, f"Cells_R12-C{i:02d}-F0-Z{z}-T0.tif")
                with open(mask_path, 'wb') as f:
                    f.write(os.urandom(128))

                for w in range(1, 3):
                    img_path = os.path.join(directory, f"t1_J{i:02d}_s1_w{w}_z{z+1}.tif")
                    with open(img_path, 'wb') as f:
                        f.write(os.urandom(128))

                    img_path = os.path.join(directory, f"t1_L{i:02d}_s1_w1_z{z+1}.tif")
                    with open(img_path, 'wb') as f:
                        f.write(os.urandom(128))
    return _create_dummy_tiffs


def test_split_random_seed_reproducibility(create_dummy_tiffs):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "Plate 2126")
        out_dir1 = os.path.join(tmpdir, "split1")
        out_dir2 = os.path.join(tmpdir, "split2")
        create_dummy_tiffs(data_dir, n=20)
        # First split
        result1 = train_test_split_directory(
            data_dir=data_dir,
            output_dir=out_dir1,
            test_size=0.3,
            random_state=123,
            image_pattern="*_w1_*.tif",
            mask_pattern="Cells_*.tif",
            file_handler=BF_IF_FileHandler()
        )
        # Second split with same seed
        result2 = train_test_split_directory(
            data_dir=data_dir,
            output_dir=out_dir2,
            test_size=0.3,
            random_state=123,
            image_pattern="*_w1_*.tif",
            mask_pattern="Cells_*.tif",
            file_handler=BF_IF_FileHandler()
        )

        assert set(result1['test_images']) == set(result2['test_images']), "Test split is not reproducible with same seed!"
        assert set(result1['train_images']) == set(result2['train_images']), "Train split is not reproducible with same seed!"
