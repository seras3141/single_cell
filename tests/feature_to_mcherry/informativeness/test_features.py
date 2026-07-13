"""Tests for feature_to_mcherry.informativeness.features."""

from __future__ import annotations

import pytest

from src.feature_to_mcherry.informativeness.features import select_morphology_features

MORPHOLOGY_PATTERNS = ["area", "perimeter", "*contrast"]
SUSPECT_PATTERNS = ["*mean_intensity"]


def test_select_morphology_features_matches_expected_subset() -> None:
    columns = [
        "area",
        "perimeter",
        "original_glcm_Contrast",
        "mean_intensity",
        "unrelated_column",
    ]

    selection = select_morphology_features(
        columns, MORPHOLOGY_PATTERNS, SUSPECT_PATTERNS
    )

    assert set(selection.all_columns) == {
        "area",
        "perimeter",
        "original_glcm_Contrast",
        "mean_intensity",
    }
    assert selection.suspect_columns == ["mean_intensity"]
    assert set(selection.clean_columns) == {
        "area",
        "perimeter",
        "original_glcm_Contrast",
    }
    assert selection.unmatched_columns == ["unrelated_column"]


def test_select_morphology_features_matching_is_case_insensitive() -> None:
    columns = ["AREA", "Original_GLCM_CONTRAST"]

    selection = select_morphology_features(
        columns, MORPHOLOGY_PATTERNS, SUSPECT_PATTERNS
    )

    assert set(selection.all_columns) == set(columns)


def test_select_morphology_features_raises_on_empty_match() -> None:
    columns = ["totally_unrelated", "another_column"]

    with pytest.raises(ValueError, match="No morphology feature columns matched"):
        select_morphology_features(columns, MORPHOLOGY_PATTERNS, SUSPECT_PATTERNS)


def test_select_morphology_features_suspect_column_not_double_counted_in_clean() -> (
    None
):
    # A column matching both a morphology pattern and a suspect pattern is tagged
    # suspect, not clean.
    columns = ["mean_intensity"]
    morphology_patterns = ["*intensity*"]
    suspect_patterns = ["*mean_intensity"]

    selection = select_morphology_features(
        columns, morphology_patterns, suspect_patterns
    )

    assert selection.all_columns == ["mean_intensity"]
    assert selection.suspect_columns == ["mean_intensity"]
    assert selection.clean_columns == []
