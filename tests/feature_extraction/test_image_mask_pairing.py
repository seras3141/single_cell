"""Tests for image/mask pairing in ``FeatureExtractionPipeline.match_files``.

Regression coverage for the prefix-pairing bug: matching used
``image_stem.startswith(mask_prefix)``, so a mask for timepoint ``t21`` also
matched image ``t211`` (one image assigned to two masks). The fix pairs by
*exact* wildcard-captured key instead.
"""

from pathlib import Path

from src.feature_extraction.feature_extraction_pipeline import (
    FeatureExtractionPipeline,
)


def _pipeline(tmp_path):
    return FeatureExtractionPipeline(
        config={"method": "regionprops"},
        output_dir=str(tmp_path / "out"),
    )


class TestPairKey:
    def test_boundary_safe(self):
        key = FeatureExtractionPipeline._pair_key
        assert key("pMF5V1_C09_t21_z10_BF.tif", "*_z10_BF.tif") == "pMF5V1_C09_t21"
        # t211 must NOT collapse to the t21 key.
        assert key("pMF5V1_C09_t211_z10_BF.tif", "*_z10_BF.tif") == "pMF5V1_C09_t211"

    def test_literal_dot(self):
        # '.' in the pattern is a literal dot, not a regex wildcard.
        assert FeatureExtractionPipeline._pair_key("aXbf.tif", "*_BF.tif") is None

    def test_no_match_returns_none(self):
        assert FeatureExtractionPipeline._pair_key("x_Cells.tif", "*_BF.tif") is None

    def test_image_and_mask_keys_align(self):
        key = FeatureExtractionPipeline._pair_key
        assert key("s_t3_z10_BF.tif", "*_z10_BF.tif") == key(
            "s_t3_z10_pred_mask.tif", "*_z10_pred_mask.tif"
        )


class TestMatchFiles:
    def test_t21_t211_no_collision(self, tmp_path):
        pipeline = _pipeline(tmp_path)
        images = [
            Path("pMF5V1_C09_t21_z10_BF.tif"),
            Path("pMF5V1_C09_t211_z10_BF.tif"),
        ]
        masks = [
            Path("pMF5V1_C09_t21_z10_pred_mask.tif"),
            Path("pMF5V1_C09_t211_z10_pred_mask.tif"),
        ]
        pairs = pipeline.match_files(
            images,
            masks,
            mask_patterns=["*_z10_pred_mask.tif"],
            image_patterns=["*_z10_BF.tif"],
        )
        # Two distinct pairs; no image reused.
        assert len(pairs) == 2
        assert len({img for img, _ in pairs}) == 2
        by_mask = {mask.name: img.name for img, mask in pairs}
        assert (
            by_mask["pMF5V1_C09_t21_z10_pred_mask.tif"] == "pMF5V1_C09_t21_z10_BF.tif"
        )
        assert (
            by_mask["pMF5V1_C09_t211_z10_pred_mask.tif"] == "pMF5V1_C09_t211_z10_BF.tif"
        )

    def test_z_plane_distinct(self, tmp_path):
        pipeline = _pipeline(tmp_path)
        images = [Path("s_t1_z0_BF.tif"), Path("s_t1_z10_BF.tif")]
        masks = [Path("s_t1_z0_pred_mask.tif"), Path("s_t1_z10_pred_mask.tif")]
        pairs = pipeline.match_files(
            images,
            masks,
            mask_patterns=["*_pred_mask.tif"],
            image_patterns=["*_BF.tif"],
        )
        by_mask = {mask.name: img.name for img, mask in pairs}
        assert by_mask["s_t1_z0_pred_mask.tif"] == "s_t1_z0_BF.tif"
        assert by_mask["s_t1_z10_pred_mask.tif"] == "s_t1_z10_BF.tif"

    def test_default_patterns_pair(self, tmp_path):
        pipeline = _pipeline(tmp_path)
        pairs = pipeline.match_files(
            [Path("well_A_BF.tif")],
            [Path("well_A_Cells.tif")],
        )
        assert len(pairs) == 1
        assert pairs[0][0].name == "well_A_BF.tif"

    def test_unmatched_mask_skipped(self, tmp_path):
        pipeline = _pipeline(tmp_path)
        images = [Path("s_t1_z10_BF.tif")]
        masks = [
            Path("s_t1_z10_pred_mask.tif"),
            Path("s_t9_z10_pred_mask.tif"),  # no image
        ]
        pairs = pipeline.match_files(
            images,
            masks,
            mask_patterns=["*_z10_pred_mask.tif"],
            image_patterns=["*_z10_BF.tif"],
        )
        assert len(pairs) == 1
        assert pairs[0][1].name == "s_t1_z10_pred_mask.tif"

    def test_duplicate_image_key_keeps_first(self, tmp_path):
        pipeline = _pipeline(tmp_path)
        # Two images collapsing to the same key (e.g. found via two patterns).
        images = [Path("s_t1_z10_BF.tif"), Path("s_t1_z10_BF.tif")]
        masks = [Path("s_t1_z10_pred_mask.tif")]
        pairs = pipeline.match_files(
            images,
            masks,
            mask_patterns=["*_z10_pred_mask.tif"],
            image_patterns=["*_z10_BF.tif"],
        )
        assert len(pairs) == 1
