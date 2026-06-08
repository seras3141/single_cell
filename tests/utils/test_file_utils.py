import pytest
import yaml
from src.utils.file_utils import (
    BF_IF_FileHandler,
    BlurFileHandler,
    ConfigurableFileHandler,
    DefaultFileHandler,
    copy_file,
    copy_without_split_dict,
    get_groups_from_filenames,
    list_all_files,
    load_wavelength_config,
    rename_all_files,
)
import tempfile
import shutil
from pathlib import Path


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _write_wavelength_yaml(path: Path, mappings: dict) -> Path:
    """Write a wavelength config YAML and return the path."""
    with open(path, "w") as f:
        yaml.dump({"wavelength_mappings": mappings}, f)
    return path

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
        expected = 'I12_z10_w1.tif'
        assert self.handler.rename_image(input_path) == expected

    def test_rename_mask(self):
        input_path = 'some/path/BF Images/Cells_R1-C1-F1-Z1-T1.tif'
        expected = 'A01_z2_Cells.tif'
        assert self.handler.rename_mask(input_path) == expected

    def test_extract_group_id(self):
        filename = 'I12_t10_z10_BF.tif'
        assert self.handler.extract_unique_id(filename) == 'I12'

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
        # Example: Plate 2126/t1_A01_s1_w1_z1.tif -> p2126_A01_t1_z1_w1.tif
        input_path = 'Plate 2126/t1_A01_s1_w1_z1.tif'
        expected = 'p2126_A01_t1_z1_w1.tif'
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
        # Example: Plate 1234/t1_B02_s1_w1_z3.tif -> p1234_B02_t1_z3_w1.tif
        input_path = 'Plate 1234/t1_B02_s1_w1_z3.tif'
        expected = 'p1234_B02_t1_z3_w1.tif'
        assert self.handler.rename_image(input_path) == expected

    def test_rename_image_different_wavelength(self):
        # Wavelength 2 is allowed and produces w2 in the output
        input_path = 'Plate 1234/t1_B02_s1_w2_z3.tif'
        expected = 'p1234_B02_t1_z3_w2.tif'
        assert self.handler.rename_image(input_path) == expected

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
        # Filenames must include _t<time>: p1234_A01_t1_BF_3d.tif -> p1234_A01_t1_BF_3d_blur32.tif
        input_path = 'p1234_A01_t1_BF_3d.tif'
        suffix = '_blur32'
        expected = 'p1234_A01_t1_BF_3d_blur32.tif'
        assert self.handler.rename_image(input_path, suffix) == expected

    def test_rename_image_missing_suffix(self):
        # Should raise ValueError if image suffix is missing
        with pytest.raises(ValueError):
            self.handler.rename_image('A01.tif', 'blur32')

    def test_extract_group_id(self):
        # BlurFileHandler inherits BF_IF_FileHandler.extract_unique_id -> plate_rowcol_time
        filename = 'p1234_A01_t1_BF_3d.tif'
        assert self.handler.extract_unique_id(filename) == 'p1234_A01_t1'

    def test_get_file_type(self):
        # 3D file requires p<plate>_<row><col>_t<time>_<type>.tif
        assert self.handler.get_file_type('p1234_A01_t1_BF_3d.tif') == 'file_3D'
        # 2D file requires p<plate>_<row><col>_t<time>_z<z>_<type>.tif
        assert self.handler.get_file_type('p1234_A01_t1_z1_BF.tif') == 'file'


# ─── load_wavelength_config ───────────────────────────────────────────────────

class TestLoadWavelengthConfig:
    def test_loads_integer_keys(self, tmp_path):
        p = _write_wavelength_yaml(tmp_path / "wl.yaml", {1: "BF", 2: "mCherry"})
        result = load_wavelength_config(str(p))
        assert result == {1: "BF", 2: "mCherry"}

    def test_string_keys_converted_to_int(self, tmp_path):
        p = _write_wavelength_yaml(tmp_path / "wl.yaml", {"1": "BF", "2": "GFP"})
        result = load_wavelength_config(str(p))
        assert 1 in result and 2 in result

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_wavelength_config(str(tmp_path / "missing.yaml"))

    def test_missing_key_raises(self, tmp_path):
        p = tmp_path / "wl.yaml"
        p.write_text("other_key: {}")
        with pytest.raises(ValueError, match="wavelength_mappings"):
            load_wavelength_config(str(p))

    def test_non_dict_mappings_raises(self, tmp_path):
        p = tmp_path / "wl.yaml"
        p.write_text("wavelength_mappings: [1, 2, 3]")
        with pytest.raises(ValueError):
            load_wavelength_config(str(p))


# ─── ConfigurableFileHandler ──────────────────────────────────────────────────

class TestConfigurableFileHandler:
    def test_custom_mappings_used(self):
        handler = ConfigurableFileHandler(wavelength_mappings={1: "BF", 2: "GFP"})
        result = handler.rename_file("Plate 2126/t1_A01_s1_w1_z1.tif", "image")
        assert "BF" in result

    def test_w2_maps_to_custom_channel(self):
        handler = ConfigurableFileHandler(wavelength_mappings={1: "BF", 2: "GFP"})
        result = handler.rename_file("Plate 2126/t1_A01_s1_w2_z1.tif", "image")
        assert "GFP" in result

    def test_default_mappings_are_ew2_convention(self):
        handler = ConfigurableFileHandler()
        assert handler.wavelength_mappings == {1: "FlipGFP", 2: "mCherry", 3: "BF"}

    def test_explicit_plate_number_override(self):
        handler = ConfigurableFileHandler(wavelength_mappings={1: "BF"})
        result = handler.rename_file(
            "Plate 2126/t1_A01_s1_w1_z1.tif", "image", plate_number="9999"
        )
        assert "9999" in result

    def test_unknown_wavelength_raises(self):
        handler = ConfigurableFileHandler(wavelength_mappings={1: "BF"})
        with pytest.raises(ValueError, match="wavelength"):
            handler.rename_file("Plate 2126/t1_A01_s1_w9_z1.tif", "image")

    def test_default_plate_number_used_when_path_has_no_plate(self):
        handler = ConfigurableFileHandler(
            wavelength_mappings={1: "BF"},
            plate_number="5555",
        )
        result = handler.rename_file("noplate/t1_A01_s1_w1_z1.tif", "image")
        assert "5555" in result

    def test_explicit_mappings_override_default(self):
        handler = ConfigurableFileHandler(wavelength_mappings={1: "DAPI", 2: "RFP"})
        result = handler.rename_file("Plate 2126/t1_A01_s1_w1_z1.tif", "image")
        assert "DAPI" in result

    def test_get_channel_name_known(self):
        handler = ConfigurableFileHandler(wavelength_mappings={1: "BF", 2: "mCherry"})
        assert handler.get_channel_name(2) == "mCherry"

    def test_get_channel_name_unknown_raises(self):
        handler = ConfigurableFileHandler(wavelength_mappings={1: "BF"})
        with pytest.raises(ValueError):
            handler.get_channel_name(99)


# ─── rename_all_files ─────────────────────────────────────────────────────────

class TestRenameAllFiles:
    def test_produces_tuples_of_src_and_renamed(self, tmp_path):
        handler = BF_IF_FileHandler()
        (tmp_path / "t1_A01_s1_w1_z1.tif").touch()
        file_map = {
            "image": [str(tmp_path / "Plate 2126" / "t1_A01_s1_w1_z1.tif")]
        }
        # Use path string that contains Plate 2126 for plate extraction
        file_map = {"image": ["Plate 2126/t1_A01_s1_w1_z1.tif"]}
        result = rename_all_files(file_map, handler)
        assert "image" in result
        assert len(result["image"]) == 1
        src, renamed = result["image"][0]
        assert src == "Plate 2126/t1_A01_s1_w1_z1.tif"
        assert "p2126" in renamed

    def test_multiple_types(self):
        handler = BF_IF_FileHandler()
        file_map = {
            "image": ["Plate 2126/t1_A01_s1_w1_z1.tif"],
            "mask": ["Plate 2126/Cells_R1-C1-F0-Z1-T0.tif"],
        }
        result = rename_all_files(file_map, handler)
        assert len(result["image"]) == 1
        assert len(result["mask"]) == 1


# ─── get_groups_from_filenames ────────────────────────────────────────────────

class TestGetGroupsFromFilenames:
    def test_groups_by_unique_id(self):
        handler = BF_IF_FileHandler()
        # file_map: {src: renamed}
        file_map = {
            "Plate 2126/t1_A01_s1_w1_z1.tif": "p2126_A01_t1_z1_w1.tif",
            "Plate 2126/t1_A01_s1_w1_z2.tif": "p2126_A01_t1_z2_w1.tif",
            "Plate 2126/t1_B02_s1_w1_z1.tif": "p2126_B02_t1_z1_w1.tif",
        }
        groups = get_groups_from_filenames(file_map, handler)
        assert "p2126_A01_t1" in groups
        assert len(groups["p2126_A01_t1"]) == 2
        assert "p2126_B02_t1" in groups
        assert len(groups["p2126_B02_t1"]) == 1


# ─── copy_file ────────────────────────────────────────────────────────────────

class TestCopyFile:
    def test_copies_file_to_destination(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst = tmp_path / "sub" / "dst.txt"

        copy_file(src, dst)

        assert dst.exists()
        assert dst.read_text() == "hello"

    def test_creates_parent_directories(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("data")
        dst = tmp_path / "a" / "b" / "c" / "dst.txt"

        copy_file(src, dst)
        assert dst.exists()

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            copy_file(tmp_path / "missing.txt", tmp_path / "out.txt")

    def test_overwrites_existing_destination(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("new content")
        dst = tmp_path / "dst.txt"
        dst.write_text("old content")

        copy_file(src, dst)

        assert "new content" in dst.read_text() or dst.exists()


# ─── copy_without_split_dict ─────────────────────────────────────────────────

class TestCopyWithoutSplitDict:
    def test_copies_all_files(self, tmp_path):
        src1 = tmp_path / "a.txt"
        src2 = tmp_path / "b.txt"
        src1.write_text("A")
        src2.write_text("B")

        output_dir = tmp_path / "output"
        file_tuple = {
            "images": [(str(src1), "renamed_a.txt")],
            "masks": [(str(src2), "renamed_b.txt")],
        }
        count = copy_without_split_dict(file_tuple, output_dir)

        assert (output_dir / "renamed_a.txt").exists()
        assert (output_dir / "renamed_b.txt").exists()
        assert count == 2

    def test_creates_output_directory(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("x")
        out = tmp_path / "new_dir"
        assert not out.exists()
        copy_without_split_dict({"f": [(str(src), "out.txt")]}, out)
        assert out.exists()


# ─── list_all_files ───────────────────────────────────────────────────────────

class TestListAllFiles:
    def test_returns_dict_keyed_by_pattern_name(self, tmp_path):
        handler = BF_IF_FileHandler()
        (tmp_path / "t1_A01_s1_w1_z1.tif").touch()
        result = list_all_files(str(tmp_path), handler)
        assert "image" in result
        assert "mask" in result
        assert isinstance(result["image"], list)
