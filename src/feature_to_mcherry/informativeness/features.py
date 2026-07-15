"""Morphology feature selection by configurable name/pattern matching."""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass
from typing import List, Sequence

logger = logging.getLogger(__name__)


@dataclass
class MorphologySelection:
    """Result of selecting morphology feature columns from a feature table.

    Attributes
    ----------
    all_columns : list[str]
        Every column matching ``morphology_feature_patterns`` or
        ``suspect_feature_patterns`` (union), in the input column order.
    suspect_columns : list[str]
        Subset of ``all_columns`` matching ``suspect_feature_patterns`` — size/
        thickness/focus proxies (e.g. raw intensity) rather than unambiguous
        morphology.
    clean_columns : list[str]
        ``all_columns`` minus ``suspect_columns``.
    unmatched_columns : list[str]
        Feature columns that matched neither pattern set, for logging/debugging.
    """

    all_columns: List[str]
    suspect_columns: List[str]
    clean_columns: List[str]
    unmatched_columns: List[str]


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatchcase(lowered, pattern.lower()) for pattern in patterns)


def select_morphology_features(
    feature_columns: Sequence[str],
    morphology_feature_patterns: Sequence[str],
    suspect_feature_patterns: Sequence[str],
) -> MorphologySelection:
    """Select an interpretable morphology subset from a feature table's columns.

    Matching is case-insensitive ``fnmatch`` (supports exact names, ``prefix*``,
    ``*suffix``, and ``*substring*``), since feature-extraction backends differ in
    naming (regionprops vs pyradiomics).

    Parameters
    ----------
    feature_columns : Sequence[str]
        Candidate feature column names (excluding ``CELL_KEY`` columns).
    morphology_feature_patterns : Sequence[str]
        Patterns for size/shape/texture columns.
    suspect_feature_patterns : Sequence[str]
        Patterns for size/thickness/focus-proxy columns (e.g. raw intensity),
        retained but tagged separately.

    Returns
    -------
    MorphologySelection

    Raises
    ------
    ValueError
        If no feature column matches any pattern.
    """
    suspect_columns = [
        column
        for column in feature_columns
        if _matches_any(column, suspect_feature_patterns)
    ]
    morphology_matches = [
        column
        for column in feature_columns
        if _matches_any(column, morphology_feature_patterns)
    ]
    suspect_set = set(suspect_columns)
    all_columns = list(dict.fromkeys(morphology_matches + suspect_columns))
    clean_columns = [column for column in all_columns if column not in suspect_set]
    unmatched_columns = [
        column for column in feature_columns if column not in all_columns
    ]

    if not all_columns:
        raise ValueError(
            "No morphology feature columns matched morphology_feature_patterns="
            f"{list(morphology_feature_patterns)} or suspect_feature_patterns="
            f"{list(suspect_feature_patterns)} among available columns: "
            f"{list(feature_columns)}"
        )

    logger.info(
        "Selected %d morphology feature columns (%d suspect, %d clean); "
        "%d unmatched columns ignored: %s",
        len(all_columns),
        len(suspect_columns),
        len(clean_columns),
        len(unmatched_columns),
        unmatched_columns,
    )

    return MorphologySelection(
        all_columns=all_columns,
        suspect_columns=suspect_columns,
        clean_columns=clean_columns,
        unmatched_columns=unmatched_columns,
    )
