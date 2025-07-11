import pytest
from src.utils.file_utils import BF_IF_FileHandler, BlurFileHandler

class TestBFIFFileHandler:
    def setup_method(self):
        self.handler = BF_IF_FileHandler()

    def test_rename_image(self):
        # Example: Plate 2126/t1_A01_s1_w1_z1.tif -> p2126_A01_z1_BF.tif
        input_path = 'Plate 2126/t1_A01_s1_w1_z1.tif'
        expected = 'p2126_A01_z1_BF.tif'
        assert self.handler.rename_image(input_path) == expected

    def test_rename_mask(self):
        # Example: Plate 2126/Cells_R1-C1-F1-Z1-T1.tif -> p2126_A01_z2_Cells.tif
        input_path = 'Plate 2126/Cells_R1-C1-F1-Z1-T1.tif'
        expected = 'p2126_A01_z2_Cells.tif'
        assert self.handler.rename_mask(input_path) == expected

    def test_extract_group_id(self):
        filename = 'p2126_A01_z1_BF.tif'
        assert self.handler.extract_group_id(filename) == 'p2126_A01'

class TestBlurFileHandler:
    def setup_method(self):
        self.handler = BlurFileHandler()

    def test_rename_image(self):
        # Example: A01_BF_3d.tif with suffix 'blur32'
        input_path = 'A01_BF_3d.tif'
        suffix = '_blur32'
        expected = 'A01_BF_3d_blur32.tif'
        assert self.handler.rename_image(input_path, suffix) == expected

    def test_rename_image_missing_suffix(self):
        # Should raise ValueError if image suffix is missing
        with pytest.raises(ValueError):
            self.handler.rename_image('A01.tif', 'blur32')

    def test_extract_group_id(self):
        filename = 'A01_BF_3d.tif'
        assert self.handler.extract_group_id(filename) == 'A01'
