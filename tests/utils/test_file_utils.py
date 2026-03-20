import pytest
from src.utils.file_utils import BF_IF_FileHandler, BlurFileHandler, DefaultFileHandler
import tempfile
import shutil
from pathlib import Path

@pytest.fixture(scope="module")
def mock_data_dirs():
    mock_data_dir = tempfile.mkdtemp()
    data_dir = Path(mock_data_dir) / "Plate 2126 Test Data"
    image_dir = data_dir
    mask_dir = data_dir / "masks"
    data_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    mock_image_files = [
        # Mock brightfield image files
        "t1_J03_s1_w1_z1.tif",
        "t1_J03_s1_w1_z2.tif",
        "t1_J03_s1_w1_z3.tif",
        # Different wavelength
        "t1_J03_s1_w2_z1.tif",
        "t1_J03_s1_w2_z2.tif",
        # Different sample
        "t1_J04_s1_w1_z1.tif",
        "t1_J04_s1_w1_z2.tif",
        "t1_J04_s1_w1_z3.tif",
        "t1_J04_s1_w2_z1.tif",
        "t1_J04_s1_w2_z2.tif",
        # Different plate
        "t1_L11_s1_w1_z1.tif",
        "t1_L11_s1_w1_z2.tif",
        "t1_L11_s1_w2_z1.tif",
    ]
    mock_mask_files = [
        "Cells_R10-C3-F0-Z0-T0.tif",
        "Cells_R10-C3-F0-Z1-T0.tif",
        "Cells_R10-C3-F0-Z2-T0.tif",
        "Cells_R10-C4-F0-Z0-T0.tif",
        "Cells_R10-C4-F0-Z1-T0.tif",
        "Cells_R10-C4-F0-Z2-T0.tif",
        "Cells_R12-C11-F0-Z0-T0.tif",
        "Cells_R12-C11-F0-Z1-T0.tif",
    ]

    for filename in mock_image_files:
        (image_dir / filename).touch()
    for filename in mock_mask_files:
        (mask_dir / filename).touch()
    yield {
        "mock_data_dir": mock_data_dir,
        "data_dir": data_dir,
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "mock_image_files": mock_image_files,
        "mock_mask_files": mock_mask_files
    }
    shutil.rmtree(mock_data_dir, ignore_errors=True)


class TestDefaultFileHandler:
    def setup_method(self):
        self.handler = DefaultFileHandler()

    def test_rename_image(self):
        input_path = 'some/path/BF Images/t1_I12_s1_w1_z10.tif'
        expected = 'I12_z10_BF.tif'
        assert self.handler.rename_image(input_path) == expected

    def test_rename_mask(self):
        input_path = 'some/path/BF Images/Cells_R1-C1-F1-Z1-T1.tif'
        expected = 'A01_z2_Cells.tif'
        assert self.handler.rename_mask(input_path) == expected

    def test_extract_group_id(self):
        filename = 'I12_t10_z10_BF.tif'
        assert self.handler.extract_unique_id(filename) == 'I12_t10'

    def test_file_handler(self, mock_data_dirs):
        file_handler = self.handler

        image_dir = mock_data_dirs["image_dir"]
        mask_dir = mock_data_dirs["mask_dir"]
        sample_image = str(image_dir / "t1_J03_s1_w1_z1.tif")
        renamed_image = file_handler.rename_image(sample_image)
        assert "J03" in renamed_image
        sample_mask = str(mask_dir / "Cells_R10-C3-F0-Z0-T0.tif")
        renamed_mask = file_handler.rename_mask(sample_mask)
        assert "J03" in renamed_mask
        group_id = file_handler.extract_unique_id(renamed_image)
        assert group_id == "J03"
        mask_group_id = file_handler.extract_unique_id(renamed_mask)
        assert mask_group_id == "J03"

class TestBFIFFileHandler:
    def setup_method(self):
        self.handler = BF_IF_FileHandler()

    def test_rename_image(self):
        # Example: Plate 2126/t1_A01_s1_w1_z1.tif -> p2126_A01_t1_z1_BF.tif
        input_path = 'Plate 2126/t1_A01_s1_w1_z1.tif'
        expected = 'p2126_A01_t1_z1_BF.tif'
        assert self.handler.rename_image(input_path) == expected

    def test_rename_mask(self):
        # Example: Plate 2126/Cells_R1-C1-F1-Z1-T1.tif -> p2126_A01_t2_z2_Cells.tif
        input_path = 'Plate 2126/Cells_R1-C1-F1-Z1-T1.tif'
        expected = 'p2126_A01_t2_z2_Cells.tif'
        assert self.handler.rename_mask(input_path) == expected

    def test_extract_group_id(self):
        filename = 'p2126_A01_t1_z1_BF.tif'
        assert self.handler.extract_unique_id(filename) == 'p2126_A01_t1'

class TestBFFileHandler:
    def setup_method(self):
        self.handler = BF_IF_FileHandler()

    def test_rename_image(self):
        # Example: Plate 1234/t1_B02_s1_w1_z3.tif -> p1234_B02_t1_z3_BF.tif
        input_path = 'Plate 1234/t1_B02_s1_w1_z3.tif'
        expected = 'p1234_B02_t1_z3_BF.tif'
        assert self.handler.rename_image(input_path) == expected

    def test_rename_image_different_wavelength(self):
        # Example: Plate 1234/t1_B02_s1_w2_z3.tif -> Error
        input_path = 'Plate 1234/t1_B02_s1_w2_z3.tif'
        with pytest.raises(ValueError):
            self.handler.rename_image(input_path)

    def test_rename_mask(self):
        # Example: Plate 1234/Cells_R2-C2-F0-Z3-T0.tif -> p1234_B02_t1_z4_Cells.tif
        input_path = 'Plate 1234/Cells_R2-C2-F0-Z3-T0.tif'
        expected = 'p1234_B02_t1_z4_Cells.tif'
        assert self.handler.rename_mask(input_path) == expected

    def test_extract_group_id(self):
        filename = 'p1234_B02_t1_z3_BF.tif'
        assert self.handler.extract_unique_id(filename) == 'p1234_B02_t1'


class TestBlurFileHandler:
    def setup_method(self):
        self.handler = BlurFileHandler()

    def test_rename_image(self):
        # Example: A01_BF_3d.tif with suffix 'blur32'
        input_path = 'p1234_A01_BF_3d.tif'
        suffix = '_blur32'
        expected = 'p1234_A01_BF_3d_blur32.tif'
        assert self.handler.rename_image(input_path, suffix) == expected

    def test_rename_image_missing_suffix(self):
        # Should raise ValueError if image suffix is missing
        with pytest.raises(ValueError):
            self.handler.rename_image('A01.tif', 'blur32')

    def test_extract_group_id(self):
        filename = 'A01_BF_3d.tif'
        assert self.handler.extract_unique_id(filename) == 'A01'

    def test_get_file_type(self):
        assert self.handler.get_file_type('p1234_A01_BF_3d.tif') == 'file_3D'
        assert self.handler.get_file_type('p1234_A01_z1_BF.tif') == 'file'
