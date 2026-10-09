"""Tests for the ``pyradiomics`` backend; all but the parity test use a fake backend."""

from dataclasses import replace
import sys
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
from src.utils.data_exclusions import DataExclusions, KnownMissing
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
        self.shapes_seen = []

    def disableAllFeatures(self):
        self.disabled_all = True

    def enableFeatureClassByName(self, name):
        self.enabled.append(name)

    reject_labels: set = set()  # execute() raises, like a rejected ROI

    def execute(self, image, mask, label=None):
        self.labels_seen.append(label)
        self.shapes_seen.append(mask.array.shape)
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

    def GetArrayFromImage(self, image):
        return image.array


class FakeImageOperations:
    def __init__(self):
        self.scales = []

    def normalizeImage(self, image, normalizeScale=1):
        self.scales.append(normalizeScale)
        return FakeImage(image.array.astype(np.float64))  # dtype as PyRadiomics


@pytest.fixture
def fake_backend(monkeypatch):
    sitk = FakeSitk()
    sitk.imageoperations = FakeImageOperations()
    featureextractor = SimpleNamespace(RadiomicsFeatureExtractor=FakeExtractor)
    monkeypatch.setattr(
        pyr,
        "_resolve_backend",
        lambda: (featureextractor, sitk.imageoperations, sitk),
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


def test_rows_columns_and_min_pixels(fake_backend):
    df = pyr.get_radiomics_features(_mask(), _image(), PyradiomicsConfig())
    assert list(df["cell_id"]) == [1, 2]  # label 3 is below min_pixels, 0 excluded
    assert list(df["touches_border"]) == [True, False]
    assert df["original_shape2D_PixelSurface"].tolist() == [36.0, 100.0]
    assert df.attrs["n_skipped_small"] == 1 and df.attrs["n_label_errors"] == 0
    # Extra numeric columns would be read as features.
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
        "normalize": False,
        "additionalInfo": False,
    }
    assert extractor.disabled_all and extractor.enabled == ["firstorder", "glcm"]
    assert extractor.labels_seen == [1, 2]


def test_cache_keyed_on_extractor_settings_only(fake_backend):
    base = PyradiomicsConfig()
    for cfg in (base, replace(base, min_pixels=5)):
        pyr.get_radiomics_features(_mask(), _image(), cfg)
    assert len(pyr._EXTRACTOR_CACHE) == 1
    pyr.get_radiomics_features(_mask(), _image(), replace(base, bin_width=5))
    pyr.get_radiomics_features(
        _mask(), _image(), replace(base, include_diagnostics=True)
    )
    assert len(pyr._EXTRACTOR_CACHE) == 3


def test_dtypes_and_large_label_ids(fake_backend):
    mask = _mask().astype(np.uint32)
    mask[mask == 2] = 70_000  # above uint16
    df = pyr.get_radiomics_features(mask, _image(), PyradiomicsConfig())
    assert sorted(df["cell_id"]) == [1, 70_000]
    # First call converts the image for normalisation, then (image, mask) per label.
    per_label = fake_backend.dtypes[1:]
    assert len(per_label) == 4
    assert per_label[0::2] == [np.dtype(np.float64)] * 2
    assert per_label[1::2] == [np.dtype(np.uint32)] * 2


def test_each_label_runs_on_its_padded_bounding_box(fake_backend):
    pyr.get_radiomics_features(_mask(), _image(), PyradiomicsConfig())
    (extractor,) = pyr._EXTRACTOR_CACHE.values()
    # bounding box + 2 px, clipped at the edge
    assert extractor.shapes_seen == [(8, 8), (14, 14)]


def test_image_normalised_once_unless_disabled(fake_backend):
    pyr.get_radiomics_features(_mask(), _image(), PyradiomicsConfig())
    assert fake_backend.imageoperations.scales == [100]
    pyr.get_radiomics_features(_mask(), _image(), PyradiomicsConfig(normalize=False))
    assert fake_backend.imageoperations.scales == [100]


def test_sparse_label_ids_do_not_size_the_counts(fake_backend):
    mask = _mask().astype(np.uint32)
    mask[mask == 2] = 4_000_000_000
    df = pyr.get_radiomics_features(mask, _image(), PyradiomicsConfig())
    assert sorted(df["cell_id"]) == [1, 4_000_000_000]
    assert df["original_shape2D_PixelSurface"].tolist() == [36.0, 100.0]
    assert df.attrs["n_skipped_small"] == 1


@pytest.mark.parametrize(
    "dtype, label", [(np.int64, -1), (np.uint64, 2**32 + 1)], ids=["neg", "big"]
)
def test_labels_outside_uint32_are_rejected(fake_backend, dtype, label):
    mask = _mask().astype(dtype)
    mask[mask == 2] = label
    with pytest.raises(ValueError, match="fit uint32"):
        pyr.get_radiomics_features(mask, _image(), PyradiomicsConfig())


def test_float_mask_is_rejected(fake_backend):
    with pytest.raises(ValueError, match="integer label image"):
        pyr.get_radiomics_features(
            _mask().astype(np.float32), _image(), PyradiomicsConfig()
        )


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


def test_real_backend_matches_uncropped_reference(monkeypatch):
    pyr._resolve_backend.cache_clear()
    monkeypatch.setattr(pyr, "_EXTRACTOR_CACHE", {})
    rng = np.random.default_rng(0)
    image = rng.integers(500, 5000, (64, 64)).astype(np.uint16)
    mask = np.zeros((64, 64), np.uint32)
    mask[:20, :25] = 1
    mask[35:60, 30:] = 70_000
    df = pyr.get_radiomics_features(mask, image, PyradiomicsConfig())
    features = [c for c in df.columns if c.startswith("original_")]
    assert len(df) == 2 and len(features) == 102
    assert df[features].notna().all().all()
    by_cell = df.set_index("cell_id")
    assert by_cell["touches_border"].all()

    import SimpleITK as sitk
    from radiomics import featureextractor

    cfg = PyradiomicsConfig()
    reference = featureextractor.RadiomicsFeatureExtractor(
        binWidth=cfg.bin_width,
        force2D=cfg.force_2d,
        normalize=cfg.normalize,
        normalizeScale=cfg.normalize_scale,
    )
    reference.disableAllFeatures()
    for name in cfg.feature_classes:
        reference.enableFeatureClassByName(name)
    image_itk = sitk.GetImageFromArray(image.astype(np.float32))
    mask_itk = sitk.GetImageFromArray(mask.astype(np.uint32))
    for label in (1, 70_000):
        expected = reference.execute(image_itk, mask_itk, label=label)
        for name in features:
            want = float(expected[name])
            assert by_cell.loc[label, name] == pytest.approx(
                want, rel=1e-4, abs=1e-6
            ), name


def test_missing_radiomics_raises_import_error(monkeypatch):
    pyr._resolve_backend.cache_clear()
    monkeypatch.setitem(sys.modules, "radiomics.featureextractor", None)
    with pytest.raises(ImportError, match="pyradiomics-cuda"):
        pyr._resolve_backend()
    pyr._resolve_backend.cache_clear()


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


def test_pipeline_fails_fast_without_backend(tmp_path, monkeypatch):
    def missing_backend():
        raise ImportError("backend unavailable")

    monkeypatch.setattr(pyr, "_resolve_backend", missing_backend)
    img_dir, msk_dir = _dataset(tmp_path, {"a": _mask(), "b": _mask()})
    pipeline = _pipeline(tmp_path)
    with pytest.raises(ImportError):
        _run(pipeline, img_dir, msk_dir)
    assert not pipeline.error_files


def test_parallel_workers_return_coverage(tmp_path):
    # Monkeypatches don't reach loky workers.
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


def _pretend_pyarrow(monkeypatch, installed: bool) -> None:
    real_find_spec = fep.importlib.util.find_spec
    monkeypatch.setattr(
        fep.importlib.util,
        "find_spec",
        lambda name, *a: (
            (object() if installed else None)
            if name == "pyarrow"
            else real_find_spec(name, *a)
        ),
    )


@pytest.fixture
def pyarrow_present(monkeypatch):
    """Pretend pyarrow is importable and record to_parquet calls instead."""
    _pretend_pyarrow(monkeypatch, installed=True)
    written = {}

    def fake_to_parquet(self, path, index=False):
        path = Path(getattr(path, "name", path))
        key = (
            f"{path.parent.name}/{path.name}"
            if path.parent.name == fep.COVERAGE_SUBDIR
            else path.name
        )
        written[key] = self.copy()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    return written


def _write_stale_coverage(pipeline, well, text):
    stale = pipeline.output_dir / fep.COVERAGE_SUBDIR / f"{well}.parquet"
    stale.parent.mkdir(exist_ok=True)
    stale.write_text(text)


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
        "coverage/E07.parquet",
        "coverage/F08.parquet",
    ]
    assert len(pyarrow_present["E07.parquet"]) == 4
    assert list(pyarrow_present["coverage/F08.parquet"]["status"]) == ["empty"]
    assert not list((tmp_path / "out").glob("*.csv"))  # no per-image files


def test_per_well_writes_real_files_with_coverage_in_subfolder(tmp_path, fake_backend):
    pytest.importorskip("pyarrow")
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07_t1_z1": _mask()})
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    _run(pipeline, img_dir, msk_dir)
    out = pipeline.output_dir
    assert [p.name for p in out.glob("*.parquet")] == ["E07.parquet"]
    assert [p.name for p in (out / fep.COVERAGE_SUBDIR).glob("*.parquet")] == [
        "E07.parquet"
    ]


def test_per_well_refuses_to_merge_two_experiments(
    tmp_path, fake_backend, pyarrow_present
):
    root = tmp_path / "root"
    for exp in ("expA", "expB"):  # same well + filenames in two experiments
        _dataset(root / exp, {"pMF5V1_E07_t1_z1": _mask()})
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    with pytest.raises(ValueError, match="more than one image directory"):
        pipeline.process_batch(root, root)
    assert fake_backend.dtypes == []  # refused before any extraction


def test_per_well_refuses_two_experiments_at_different_timepoints(
    tmp_path, fake_backend, pyarrow_present
):
    root = tmp_path / "root"
    _dataset(root / "expA", {"pMF5V1_E07_t1_z1": _mask()})
    _dataset(root / "expB", {"pMF5V1_E07_t2_z1": _mask()})
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    with pytest.raises(ValueError, match="more than one image directory"):
        pipeline.process_batch(root, root)
    assert fake_backend.dtypes == []


def test_per_well_refuses_a_second_directory_without_pairs(
    tmp_path, fake_backend, pyarrow_present
):
    root = tmp_path / "root"
    _dataset(root / "expA", {"pMF5V1_E07_t1_z1": _mask()})
    (root / "expB" / "imgs").mkdir(parents=True)
    tifffile.imwrite(root / "expB" / "imgs" / "pMF5V1_E07_t2_z0_BF.tif", _image())
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    with pytest.raises(ValueError, match="more than one image directory"):
        pipeline.process_batch(root, root)


def test_per_well_writes_only_wells_with_masks(tmp_path, fake_backend, pyarrow_present):
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07_t1_z1": _mask()})
    tifffile.imwrite(img_dir / "pMF5V1_F08_t1_z1_BF.tif", _image())
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    _write_stale_coverage(pipeline, "F08", "another task's well")
    _run(pipeline, img_dir, msk_dir)
    assert sorted(pyarrow_present) == ["E07.parquet", "coverage/E07.parquet"]
    assert [msg for _, msg in pipeline.error_files] == ["No matching mask"]


def test_per_well_extra_image_without_mask_fails_only_itself(
    tmp_path, fake_backend, pyarrow_present
):
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07_t1_z1": _mask()})
    tifffile.imwrite(img_dir / "pMF5V1_E07_t1_z1_old_BF.tif", _image())
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    _run(pipeline, img_dir, msk_dir)
    assert sorted(pyarrow_present) == ["E07.parquet", "coverage/E07.parquet"]
    assert [msg for _, msg in pipeline.error_files] == ["No matching mask"]


def test_per_well_duplicate_images_without_mask_own_no_well(
    tmp_path, fake_backend, pyarrow_present
):
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07_t1_z1": _mask()})
    tifffile.imwrite(img_dir / "pMF5V1_F08_t1_z1_BF.tif", _image())
    tifffile.imwrite(img_dir / "pMF5V1_F08_t1_z1.tif", _image())
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    _write_stale_coverage(pipeline, "F08", "another task's well")
    pipeline.process_batch(img_dir, msk_dir, image_patterns=["*_BF.tif", "*.tif"])
    assert sorted(pyarrow_present) == ["E07.parquet", "coverage/E07.parquet"]
    assert [msg for _, msg in pipeline.error_files] == [
        "Duplicate pairing key 'pMF5V1_F08_t1_z1'"
    ] * 2


def test_per_well_coverage_lists_unsegmented_z0(
    tmp_path, fake_backend, pyarrow_present
):
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07_t1_z1": _mask()})
    tifffile.imwrite(img_dir / "pMF5V1_E07_t1_z0_BF.tif", _image())
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    _run(pipeline, img_dir, msk_dir)
    coverage = pyarrow_present["coverage/E07.parquet"]
    assert sorted(coverage["status"]) == ["ok", "unsegmented"]


def test_per_well_refuses_to_overwrite_within_a_run(
    tmp_path, fake_backend, pyarrow_present
):
    a_img, a_msk = _dataset(tmp_path / "a", {"pMF5V1_E07_t1_z1": _mask()})
    b_img, b_msk = _dataset(tmp_path / "b", {"pMF5V1_E07_t2_z1": _mask()})
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    _run(pipeline, a_img, a_msk)
    with pytest.raises(ValueError, match="written earlier in this run"):
        _run(pipeline, b_img, b_msk)


@pytest.mark.parametrize("earlier", ["E07.parquet", "coverage/E07.parquet"])
def test_per_well_refuses_earlier_outputs(
    tmp_path, fake_backend, pyarrow_present, earlier
):
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07_t1_z1": _mask()})
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    (pipeline.output_dir / earlier).parent.mkdir(exist_ok=True)
    (pipeline.output_dir / earlier).write_text("from an earlier run")
    with pytest.raises(ValueError, match="exists from an earlier run"):
        _run(pipeline, img_dir, msk_dir)
    assert fake_backend.dtypes == [] and not pyarrow_present
    assert (pipeline.output_dir / earlier).read_text() == "from an earlier run"


def test_per_well_refuses_a_root_level_coverage_file_from_the_old_layout(
    tmp_path, fake_backend, pyarrow_present
):
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07_t1_z1": _mask()})
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    old_layout = pipeline.output_dir / "E07_coverage.parquet"
    old_layout.write_text("from an earlier run")
    with pytest.raises(ValueError, match="exists from an earlier run"):
        _run(pipeline, img_dir, msk_dir)
    assert fake_backend.dtypes == [] and not pyarrow_present


def test_well_only_name_still_checks_earlier_outputs(
    tmp_path, fake_backend, pyarrow_present
):
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07": _mask()})
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    _write_stale_coverage(pipeline, "E07", "from an earlier run")
    with pytest.raises(ValueError, match="exists from an earlier run"):
        _run(pipeline, img_dir, msk_dir)


def test_zero_padded_timepoints_collide(tmp_path, fake_backend, pyarrow_present):
    img_dir, msk_dir = _dataset(
        tmp_path, {"pMF5V1_E07_t1_z1": _mask(), "pMF5V2_E07_t001_z1": _mask()}
    )
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    with pytest.raises(ValueError, match="share well/timepoint/z"):
        _run(pipeline, img_dir, msk_dir)


def test_per_well_skips_names_without_a_well(tmp_path, fake_backend, pyarrow_present):
    img_dir, msk_dir = _dataset(
        tmp_path, {"pMF5V1_E07_t1_z1": _mask(), "unparsed": _mask()}
    )
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    df = _run(pipeline, img_dir, msk_dir)
    assert set(df["sample_id"]) == {"E07"}
    assert sorted(pyarrow_present) == ["E07.parquet", "coverage/E07.parquet"]
    assert pipeline.error_files == [
        (str(img_dir / "unparsed_BF.tif"), "No well/timepoint/z in filename")
    ]


def test_per_well_refuses_duplicate_unpaired_masks(
    tmp_path, fake_backend, pyarrow_present
):
    root = tmp_path / "root"
    _dataset(root / "expA", {"pMF5V1_E07_t1_z1": _mask()})
    for exp in ("expA", "expB"):
        (root / exp / "msks").mkdir(parents=True, exist_ok=True)
        save_labels(_mask(), root / exp / "msks" / "pMF5V1_E07_t1_z2_pred_mask.tif")
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    with pytest.raises(ValueError, match="more than one mask directory"):
        pipeline.process_batch(root, root)
    assert fake_backend.dtypes == []  # refused before any extraction


def test_per_well_refuses_same_key_under_different_prefixes(
    tmp_path, fake_backend, pyarrow_present
):
    img_dir, msk_dir = _dataset(
        tmp_path, {"pMF5V1_E07_t1_z1": _mask(), "pMF5V2_E07_t1_z1": _mask()}
    )
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    with pytest.raises(ValueError, match="share well/timepoint/z"):
        _run(pipeline, img_dir, msk_dir)


def test_error_only_well_still_gets_coverage(tmp_path, fake_backend, pyarrow_present):
    img_dir, msk_dir = _dataset(
        tmp_path,
        {"pMF5V1_E07_t1_z1": _mask(), "pMF5V1_F08_t1_z1": np.stack([_mask()] * 3)},
    )
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    _run(pipeline, img_dir, msk_dir)
    assert sorted(pyarrow_present) == [
        "E07.parquet",
        "coverage/E07.parquet",
        "coverage/F08.parquet",
    ]
    assert list(pyarrow_present["coverage/F08.parquet"]["status"]) == ["error"]


def test_single_image_refuses_to_overwrite_a_well(
    tmp_path, fake_backend, pyarrow_present
):
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07_t1_z1": _mask()})
    pipeline = _pipeline(
        tmp_path, format="parquet", granularity="well", save_combined_file=False
    )
    (pipeline.output_dir / "E07.parquet").write_text("from a batch run")
    with pytest.raises(ValueError, match="exists from an earlier run"):
        pipeline.process_single_image(
            img_dir / "pMF5V1_E07_t1_z1_BF.tif",
            msk_dir / "pMF5V1_E07_t1_z1_pred_mask.tif",
        )
    assert not pyarrow_present


def test_per_well_writes_coverage_for_a_known_missing_batch(
    tmp_path, fake_backend, pyarrow_present
):
    exp = "HD1883 MF5V1 0-72h 20-03-26"
    img_dir, msk_dir = _dataset(tmp_path / exp, {})
    save_labels(_mask(), msk_dir / "pMF5V1_H09_t201_z4_pred_mask.tif")
    pipeline = FeatureExtractionPipeline(
        config={
            "method": "pyradiomics",
            "n_jobs": 1,
            "output": {
                "save_individual_files": False,
                "create_subdirs": False,
                "format": "parquet",
                "granularity": "well",
            },
        },
        output_dir=str(tmp_path / "out"),
        exclusions=DataExclusions(
            known_missing=(KnownMissing(exp, "H09", 201, 4, "BF", "absent"),)
        ),
    )
    _run(pipeline, img_dir, msk_dir)
    assert sorted(pyarrow_present) == ["coverage/H09.parquet"]
    assert list(pyarrow_present["coverage/H09.parquet"]["status"]) == ["known_missing"]


def test_unpaired_only_batch_writes_coverage(tmp_path, fake_backend, pyarrow_present):
    img_dir, msk_dir = _dataset(tmp_path, {})
    save_labels(_mask(), msk_dir / "pMF5V1_E07_t1_z2_pred_mask.tif")
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    _run(pipeline, img_dir, msk_dir)
    assert sorted(pyarrow_present) == ["coverage/E07.parquet"]
    assert list(pyarrow_present["coverage/E07.parquet"]["status"]) == ["unpaired"]
    assert (str(msk_dir), "No valid image-mask pairs") in pipeline.error_files


def test_single_image_mode_writes_per_well(tmp_path, fake_backend, pyarrow_present):
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07_t1_z1": _mask()})
    pipeline = _pipeline(
        tmp_path, format="parquet", granularity="well", save_combined_file=False
    )
    pipeline.process_single_image(
        img_dir / "pMF5V1_E07_t1_z1_BF.tif", msk_dir / "pMF5V1_E07_t1_z1_pred_mask.tif"
    )
    assert sorted(pyarrow_present) == ["E07.parquet", "coverage/E07.parquet"]


def test_output_option_validation(tmp_path, pyarrow_present):
    with pytest.raises(ValueError, match="requires output.format 'parquet'"):
        _pipeline(tmp_path, format="csv", granularity="well")
    with pytest.raises(ValueError, match="output.format"):
        _pipeline(tmp_path, format="xlsx")
    with pytest.raises(ValueError, match="output.granularity"):
        _pipeline(tmp_path, format="parquet", granularity="plate")


def test_parquet_without_pyarrow_fails_at_construction(tmp_path, monkeypatch):
    _pretend_pyarrow(monkeypatch, installed=False)
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
    img_dir, msk_dir = _dataset(tmp_path, {"pMF5V1_E07_t1_z1": _mask()})
    pipeline = _pipeline(tmp_path, format="parquet", granularity="well")
    _run(pipeline, img_dir, msk_dir)
    df = pd.read_parquet(tmp_path / "out" / "E07.parquet")
    assert list(df["cell_id"]) == [1, 2]


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
