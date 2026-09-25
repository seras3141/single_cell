"""Tests for the ``pyradiomics`` backend (no GPU, PyRadiomics or SimpleITK needed).

A fake backend is patched in at ``_resolve_backend``: a fake
``RadiomicsFeatureExtractor`` that records its settings and returns
``original_*``/``diagnostics_*`` values computed from the arrays it is given, and
a fake ``SimpleITK`` whose images just wrap the numpy array.
"""

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

import src.feature_extraction.feature_extraction_pipeline as fep
import src.feature_extraction.feature_extractor_pyradiomics as pyr
from src.feature_extraction.feature_extraction_pipeline import FeatureExtractionPipeline
from src.utils.config import ConfigManager
from src.utils.config_schemas import PyradiomicsConfig
from src.utils.data_exclusions import DataExclusions
from src.utils.image_utils import save_labels

REPO = Path(__file__).resolve().parents[2]


class FakeImage:
    def __init__(self, array):
        self.array = array


class FakeExtractor:
    def __init__(self, **settings):
        self.settings = settings
        self.enabled = []
        self.disabled_all = False
        self.labels_seen = []

    def disableAllFeatures(self):
        self.disabled_all = True

    def enableFeatureClassByName(self, name):
        self.enabled.append(name)

    reject_labels: set = set()  # labels whose execute() raises, like a bad ROI

    def execute(self, image, mask, label=None):
        self.labels_seen.append(label)
        if label in self.reject_labels:
            raise ValueError("mask has too few dimensions")
        region = mask.array == label
        return {
            "diagnostics_Versions_PyRadiomics": "fake",
            "diagnostics_Mask-original_BoundingBox": (0, 0, 1, 1),
            "original_firstorder_Mean": np.array(float(image.array[region].mean())),
            "original_shape2D_PixelSurface": np.array(float(region.sum())),
        }


class FakeSitk:
    def __init__(self):
        self.dtypes = []

    def GetImageFromArray(self, array):
        self.dtypes.append(array.dtype)
        return FakeImage(array)


@pytest.fixture
def fake_backend(monkeypatch):
    sitk = FakeSitk()
    featureextractor = SimpleNamespace(RadiomicsFeatureExtractor=FakeExtractor)
    monkeypatch.setattr(
        pyr, "_resolve_backend", lambda require_cuda: (featureextractor, sitk)
    )
    monkeypatch.setattr(pyr, "_EXTRACTOR_CACHE", {})
    return sitk


def _mask() -> np.ndarray:
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[0:6, 0:6] = 1  # 36 px, touches the border
    mask[10:20, 10:20] = 2  # 100 px, interior
    mask[25:28, 25:28] = 3  # 9 px, below min_pixels=20
    return mask


def _image() -> np.ndarray:
    return (np.arange(32 * 32).reshape(32, 32) % 97).astype(np.uint16)


# --- extractor -------------------------------------------------------------------


def test_rows_columns_and_min_pixels(fake_backend):
    df = pyr.get_radiomics_features(_mask(), _image(), PyradiomicsConfig())
    assert list(df["cell_id"]) == [1, 2]  # label 3 is below min_pixels, 0 excluded
    assert list(df["touches_border"]) == [True, False]
    assert df["original_shape2D_PixelSurface"].tolist() == [36.0, 100.0]
    assert df.attrs["n_skipped_small"] == 1 and df.attrs["n_label_errors"] == 0
    # Only features and the border flag: no timing/bookkeeping columns that a
    # numeric-column consumer would mistake for features.
    assert set(df.columns) == {
        "cell_id",
        "touches_border",
        "original_firstorder_Mean",
        "original_shape2D_PixelSurface",
    }


def test_rejected_label_is_skipped_not_fatal(fake_backend, monkeypatch):
    monkeypatch.setattr(FakeExtractor, "reject_labels", {2})
    df = pyr.get_radiomics_features(_mask(), _image(), PyradiomicsConfig())
    assert list(df["cell_id"]) == [1]
    assert df.attrs["n_label_errors"] == 1


def test_non_scalar_feature_becomes_nan(fake_backend, monkeypatch):
    real_execute = FakeExtractor.execute

    def odd(self, image, mask, label=None):
        out = real_execute(self, image, mask, label)
        out["original_firstorder_Mean"] = (
            None if label == 1 else out["original_firstorder_Mean"]
        )
        return out

    monkeypatch.setattr(FakeExtractor, "execute", odd)
    df = pyr.get_radiomics_features(_mask(), _image(), PyradiomicsConfig())
    assert df["original_firstorder_Mean"].dtype == np.float64
    assert np.isnan(df["original_firstorder_Mean"].iloc[0])


def test_feature_values_are_unwrapped_scalars(fake_backend):
    df = pyr.get_radiomics_features(_mask(), _image(), PyradiomicsConfig())
    expected = _image()[_mask() == 2].mean()
    assert df.loc[df["cell_id"] == 2, "original_firstorder_Mean"].item() == expected
    assert df["original_firstorder_Mean"].dtype == np.float64


def test_diagnostics_opt_in_are_stringified(fake_backend):
    cfg = PyradiomicsConfig(include_diagnostics=True)
    df = pyr.get_radiomics_features(_mask(), _image(), cfg)
    assert df["diagnostics_Versions_PyRadiomics"].tolist() == ["fake", "fake"]
    assert df["diagnostics_Mask-original_BoundingBox"].iloc[0] == "(0, 0, 1, 1)"


def test_extractor_settings_and_classes(fake_backend):
    cfg = PyradiomicsConfig(bin_width=10, feature_classes=["firstorder", "glcm"])
    pyr.get_radiomics_features(_mask(), _image(), cfg)
    (extractor,) = pyr._EXTRACTOR_CACHE.values()
    assert extractor.settings == {
        "binWidth": 10,
        "force2D": True,
        "normalize": True,
        "normalizeScale": 100,
    }
    assert extractor.disabled_all and extractor.enabled == ["firstorder", "glcm"]
    assert extractor.labels_seen == [1, 2]


def test_cache_keyed_on_extractor_settings_only(fake_backend):
    base = PyradiomicsConfig()
    for cfg in (
        base,
        replace(base, min_pixels=5),
        replace(base, include_diagnostics=True),
    ):
        pyr.get_radiomics_features(_mask(), _image(), cfg)
    assert len(pyr._EXTRACTOR_CACHE) == 1
    pyr.get_radiomics_features(_mask(), _image(), replace(base, bin_width=5))
    assert len(pyr._EXTRACTOR_CACHE) == 2


def test_dtypes_and_large_label_ids(fake_backend):
    mask = _mask().astype(np.uint32)
    mask[mask == 2] = 70_000  # above uint16
    df = pyr.get_radiomics_features(mask, _image(), PyradiomicsConfig())
    assert sorted(df["cell_id"]) == [1, 70_000]
    assert fake_backend.dtypes == [np.dtype(np.float32), np.dtype(np.uint32)]


def test_empty_and_all_small_masks(fake_backend):
    empty = pyr.get_radiomics_features(
        np.zeros((8, 8), np.uint16), np.ones((8, 8)), PyradiomicsConfig()
    )
    assert empty.empty and "cell_id" in empty.columns
    tiny = np.zeros((8, 8), np.uint16)
    tiny[1:3, 1:3] = 1
    small = pyr.get_radiomics_features(tiny, np.ones((8, 8)), PyradiomicsConfig())
    assert small.empty and small.attrs["n_skipped_small"] == 1
    assert fake_backend.dtypes == []  # no backend work when nothing is extracted


@pytest.mark.skipif(
    importlib.util.find_spec("radiomics") is not None, reason="radiomics is installed"
)
def test_missing_radiomics_raises_import_error():
    with pytest.raises(ImportError, match=".venv-pyradiomics"):
        pyr._resolve_backend(require_cuda=False)


def test_require_cuda_without_distribution_raises(monkeypatch):
    def missing(name):
        raise pyr.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(pyr.metadata, "distribution", missing)
    with pytest.raises(ImportError, match="pyradiomics-cuda"):
        pyr._resolve_backend(require_cuda=True)


# --- pipeline --------------------------------------------------------------------


def _dataset(root: Path, masks):
    img_dir, msk_dir = root / "imgs", root / "msks"
    img_dir.mkdir(parents=True)
    msk_dir.mkdir(parents=True)
    for name, mask in masks.items():
        tifffile.imwrite(img_dir / f"{name}_BF.tif", _image())
        if mask is not None:
            if mask.ndim == 3:
                tifffile.imwrite(msk_dir / f"{name}_pred_mask.tif", mask)
            else:
                save_labels(mask, msk_dir / f"{name}_pred_mask.tif")
    return img_dir, msk_dir


def _pipeline(tmp_path, method="pyradiomics", n_jobs=1, **output):
    return FeatureExtractionPipeline(
        config={
            "method": method,
            "n_jobs": n_jobs,
            "output": {
                "save_individual_files": False,
                "create_subdirs": False,
                **output,
            },
        },
        output_dir=str(tmp_path / "out"),
        exclusions=DataExclusions.empty(),
    )


def _run(pipeline, img_dir, msk_dir):
    return pipeline.process_batch(img_dir, msk_dir)


def test_pipeline_rows_carry_key_columns(tmp_path, fake_backend):
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07_t101_z10": _mask()})
    df = _run(_pipeline(tmp_path), img_dir, msk_dir)
    assert list(df["cell_id"]) == [1, 2]
    assert set(df["sample_id"]) == {"E07"}
    assert set(df["timepoint"]) == {"101"}
    assert set(df["z_index"]) == {10}


def test_pipeline_coverage_statuses(tmp_path, fake_backend):
    corrupt = np.stack([_mask()] * 3)  # 3-page stack: a per-file error
    img_dir, msk_dir = _dataset(
        tmp_path,
        {
            "pMF5V1_E07_t1_z1": _mask(),
            "pMF5V1_E07_t1_z2": np.zeros((32, 32), np.uint16),
            "pMF5V1_E07_t1_z3": corrupt,
        },
    )
    pipeline = _pipeline(tmp_path)
    _run(pipeline, img_dir, msk_dir)
    by_z = {r["z_index"]: r for r in pipeline.coverage_records}
    assert by_z[1]["status"] == "ok" and by_z[1]["n_cells"] == 2
    assert by_z[1]["n_skipped_small"] == 1
    assert by_z[2]["status"] == "empty" and by_z[2]["n_cells"] == 0
    assert by_z[3]["status"] == "error"
    assert "Coverage (images): empty=1, error=1, ok=1" in (
        pipeline.save_summary(pd.DataFrame()).read_text()
    )


def test_unpaired_mask_gets_a_coverage_record(tmp_path, fake_backend):
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07_t1_z1": _mask()})
    save_labels(_mask(), msk_dir / "pMF5V1_E07_t1_z2_pred_mask.tif")  # no image
    pipeline = _pipeline(tmp_path)
    _run(pipeline, img_dir, msk_dir)
    unpaired = [r for r in pipeline.coverage_records if r["status"] == "unpaired"]
    assert len(unpaired) == 1
    assert unpaired[0]["image_filename"] == "" and unpaired[0]["z_index"] == 2


@pytest.mark.skipif(
    importlib.util.find_spec("radiomics") is not None, reason="radiomics is installed"
)
def test_pipeline_fails_fast_without_backend(tmp_path):
    img_dir, msk_dir = _dataset(tmp_path, {"a": _mask(), "b": _mask()})
    pipeline = _pipeline(tmp_path)
    with pytest.raises(ImportError):
        _run(pipeline, img_dir, msk_dir)
    assert not pipeline.error_files


def test_parallel_workers_return_coverage(tmp_path):
    # A monkeypatched backend does not reach loky workers, so use regionprops.
    img_dir, msk_dir = _dataset(tmp_path, {f"s{i}": _mask() for i in range(3)})
    pipeline = _pipeline(tmp_path, method="regionprops", n_jobs=2)
    _run(pipeline, img_dir, msk_dir)
    assert sorted(r["status"] for r in pipeline.coverage_records) == ["ok"] * 3


def test_bad_pyradiomics_settings_fail_at_construction(tmp_path):
    with pytest.raises(ValueError, match="Invalid feature_extraction.pyradiomics"):
        FeatureExtractionPipeline(
            config={"method": "pyradiomics", "pyradiomics": {"bin_wdth": 5}},
            output_dir=str(tmp_path / "out"),
            exclusions=DataExclusions.empty(),
        )
    with pytest.raises(ValueError, match="bin_width must be > 0"):
        FeatureExtractionPipeline(
            config={"method": "pyradiomics", "pyradiomics": {"bin_width": 0}},
            output_dir=str(tmp_path / "out"),
            exclusions=DataExclusions.empty(),
        )


# --- output options --------------------------------------------------------------


@pytest.fixture
def pyarrow_present(monkeypatch):
    """Pretend pyarrow is importable and record to_parquet calls instead."""
    real_find_spec = fep.importlib.util.find_spec
    monkeypatch.setattr(
        fep.importlib.util,
        "find_spec",
        lambda name, *a: object() if name == "pyarrow" else real_find_spec(name, *a),
    )
    written = {}

    def fake_to_parquet(self, path, index=False):
        written[Path(path).name] = self.copy()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    return written


def test_per_well_outputs(tmp_path, fake_backend, pyarrow_present):
    img_dir, msk_dir = _dataset(
        tmp_path,
        {
            "pMF5V1_E07_t1_z1": _mask(),
            "pMF5V1_E07_t1_z2": _mask(),
            "pMF5V1_F08_t1_z1": np.zeros((32, 32), np.uint16),  # empty well
        },
    )
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    _run(pipeline, img_dir, msk_dir)
    assert sorted(pyarrow_present) == [
        "E07.parquet",
        "E07_coverage.parquet",
        "F08_coverage.parquet",
    ]
    assert len(pyarrow_present["E07.parquet"]) == 4
    assert list(pyarrow_present["F08_coverage.parquet"]["status"]) == ["empty"]
    assert not list((tmp_path / "out").glob("*.csv"))  # no per-image files


def test_per_well_refuses_to_merge_two_experiments(
    tmp_path, fake_backend, pyarrow_present
):
    root = tmp_path / "root"
    for exp in ("expA", "expB"):  # same well + filenames in two experiments
        _dataset(root / exp, {"pMF5V1_E07_t1_z1": _mask()})
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    with pytest.raises(ValueError, match="merge or overwrite"):
        pipeline.process_batch(root, root)


def test_per_well_refuses_to_overwrite_within_a_run(
    tmp_path, fake_backend, pyarrow_present
):
    a_img, a_msk = _dataset(tmp_path / "a", {"pMF5V1_E07_t1_z1": _mask()})
    b_img, b_msk = _dataset(tmp_path / "b", {"pMF5V1_E07_t2_z1": _mask()})
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    _run(pipeline, a_img, a_msk)
    with pytest.raises(ValueError, match="written earlier in this run"):
        _run(pipeline, b_img, b_msk)


def test_per_well_removes_stale_features_file(tmp_path, fake_backend, pyarrow_present):
    img_dir, msk_dir = _dataset(
        tmp_path, {"pMF5V1_E07_t1_z1": np.zeros((32, 32), np.uint16)}
    )
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    stale = pipeline.output_dir / "E07.parquet"
    stale.write_text("from an earlier run")
    _run(pipeline, img_dir, msk_dir)
    assert not stale.exists()
    assert "E07_coverage.parquet" in pyarrow_present


def test_single_image_mode_writes_per_well(tmp_path, fake_backend, pyarrow_present):
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07_t1_z1": _mask()})
    pipeline = _pipeline(
        tmp_path, format="parquet", granularity="well", save_combined_file=False
    )
    pipeline.process_single_image(
        img_dir / "pMF5V1_E07_t1_z1_BF.tif", msk_dir / "pMF5V1_E07_t1_z1_pred_mask.tif"
    )
    assert sorted(pyarrow_present) == ["E07.parquet", "E07_coverage.parquet"]


def test_output_option_validation(tmp_path, pyarrow_present):
    with pytest.raises(ValueError, match="requires output.format 'parquet'"):
        _pipeline(tmp_path, format="csv", granularity="well")
    with pytest.raises(ValueError, match="output.format"):
        _pipeline(tmp_path, format="xlsx")
    with pytest.raises(ValueError, match="output.granularity"):
        _pipeline(tmp_path, format="parquet", granularity="plate")


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is not None, reason="pyarrow is installed"
)
def test_parquet_without_pyarrow_fails_at_construction(tmp_path):
    with pytest.raises(ImportError, match="pyarrow"):
        _pipeline(tmp_path, format="parquet")


def test_default_output_is_unchanged_csv(tmp_path):
    img_dir, msk_dir = _dataset(tmp_path, {"a": _mask()})
    pipeline = _pipeline(tmp_path, method="regionprops", save_individual_files=True)
    _run(pipeline, img_dir, msk_dir)
    assert [p.name for p in (tmp_path / "out").iterdir() if p.suffix == ".csv"] == [
        "a_BF_features.csv"
    ]


def test_parquet_round_trip(tmp_path, fake_backend):
    pytest.importorskip("pyarrow")
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07_t1_z1": _mask()})
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    _run(pipeline, img_dir, msk_dir)
    df = pd.read_parquet(tmp_path / "out" / "E07.parquet")
    assert list(df["cell_id"]) == [1, 2]


# --- config ----------------------------------------------------------------------


def _write_config(tmp_path, block: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(f"feature_extraction:\n  method: pyradiomics\n{block}")
    return path


def test_config_block_loads_through_config_manager(tmp_path):
    path = _write_config(
        tmp_path, "  pyradiomics:\n    bin_width: 10\n    min_pixels: 50\n"
    )
    fe = ConfigManager(str(path)).to_dict()["feature_extraction"]
    assert fe["pyradiomics"]["bin_width"] == 10
    assert fe["pyradiomics"]["min_pixels"] == 50
    assert fe["pyradiomics"]["normalize_scale"] == 100  # schema default


@pytest.mark.parametrize(
    "block",
    [
        "  pyradiomics:\n    bin_wdth: 10\n",  # unknown key
        "  pyradiomics:\n    feature_classes: [wavelet]\n",  # unknown class
        "  pyradiomics:\n    min_pixels: 0\n",
    ],
)
def test_bad_config_block_fails(tmp_path, block):
    with pytest.raises(ValueError):
        ConfigManager(str(_write_config(tmp_path, block)))


def test_shipped_pyradiomics_config_loads():
    fe = ConfigManager(
        str(REPO / "config/feature_extraction_pyradiomics_config.yaml")
    ).to_dict()["feature_extraction"]
    assert fe["method"] == "pyradiomics"
    assert fe["output"]["granularity"] == "well"
    assert fe["pyradiomics"]["require_cuda"] is True
