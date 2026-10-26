"""Join parity: do the feature and target tables hold the same cells?

Each table is reduced to its ``CELL_KEY`` rows and compared pairwise. The tables come
from the analysis loaders, so the comparison is on the population the baselines use:
the stack-exclusion registry is applied, and target rows with a NaN percentile are
dropped (their raw count is reported). A gap, a key repeated within one table, an empty
table or a null key is a failure, reported with a per-well and per-timepoint breakdown
so an acquisition dropout can be told from a real mismatch.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..data.contract import CELL_KEY, normalize_cell_key
from ..data.loaders import load_features_from_directory, load_targets

logger = logging.getLogger(__name__)

MAX_GAP_ROWS = 10_000
_BREAKDOWN_ROWS = 20


@dataclass
class PairParity:
    left: str
    right: str
    matched: int
    left_only: int
    right_only: int

    @property
    def exact(self) -> bool:
        return self.left_only == 0 and self.right_only == 0


@dataclass
class ParityReport:
    label: str
    rows: Dict[str, int] = field(default_factory=dict)
    raw_rows: Dict[str, int] = field(default_factory=dict)
    duplicates: Dict[str, int] = field(default_factory=dict)
    null_keys: Dict[str, int] = field(default_factory=dict)
    pairs: List[PairParity] = field(default_factory=list)
    n_gap_rows: int = 0
    gaps_truncated: bool = False
    gap_by_stack: List[Dict[str, object]] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        return (
            bool(self.rows)
            and all(self.rows.values())
            and all(pair.exact for pair in self.pairs)
            and not any(self.duplicates.values())
            and not any(self.null_keys.values())
        )

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["pairs"] = [{**asdict(pair), "exact": pair.exact} for pair in self.pairs]
        data["exact"] = self.exact
        return data


def load_feature_keys(
    directory: Path, pattern: str, id_column: str = "cell_id"
) -> pd.DataFrame:
    features = load_features_from_directory(
        Path(directory), id_column=id_column, pattern=pattern
    )
    return features[CELL_KEY].copy()


def load_target_keys(csv_path: Path) -> pd.DataFrame:
    return load_targets(Path(csv_path))[CELL_KEY].copy()


def count_raw_target_rows(csv_path: Path) -> int:
    return len(pd.read_csv(csv_path, usecols=["cell_id"]))


def _gap_keys(
    left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str
) -> pd.DataFrame:
    merged = left.drop_duplicates().merge(
        right.drop_duplicates(), on=CELL_KEY, how="outer", indicator=True
    )
    gaps = merged[merged["_merge"] != "both"].copy()
    gaps["pair"] = f"{left_name}|{right_name}"
    gaps["only_in"] = gaps["_merge"].map(
        {"left_only": left_name, "right_only": right_name}
    )
    return gaps.drop(columns="_merge")


def compare_tables(
    label: str,
    tables: Dict[str, pd.DataFrame],
    raw_rows: Optional[Dict[str, int]] = None,
) -> Tuple[ParityReport, pd.DataFrame]:
    """Pairwise parity of the ``CELL_KEY`` rows of every table in ``tables``."""
    report = ParityReport(label=label, raw_rows=dict(raw_rows or {}))
    normalized = {}
    for name, keys in tables.items():
        report.rows[name] = len(keys)
        report.null_keys[name] = int(keys[CELL_KEY].isna().any(axis=1).sum())
        normalized[name] = normalize_cell_key(keys)
        report.duplicates[name] = int(normalized[name].duplicated(CELL_KEY).sum())

    gap_frames = []
    for left_name, right_name in itertools.combinations(normalized, 2):
        gaps = _gap_keys(
            normalized[left_name], normalized[right_name], left_name, right_name
        )
        n_left = int((gaps["only_in"] == left_name).sum())
        matched = len(normalized[left_name].drop_duplicates()) - n_left
        report.pairs.append(
            PairParity(left_name, right_name, matched, n_left, len(gaps) - n_left)
        )
        gap_frames.append(gaps)

    gaps = pd.concat(gap_frames, ignore_index=True) if gap_frames else pd.DataFrame()
    report.n_gap_rows = len(gaps)
    report.gaps_truncated = len(gaps) > MAX_GAP_ROWS
    if len(gaps):
        breakdown = (
            gaps.groupby(["pair", "only_in", "sample_id", "timepoint"])
            .size()
            .rename("n_cells")
            .reset_index()
            .sort_values("n_cells", ascending=False)
        )
        report.gap_by_stack = breakdown.head(_BREAKDOWN_ROWS).to_dict("records")
    return report, gaps


def write_report(
    report: ParityReport, gaps: pd.DataFrame, report_path: Path
) -> Optional[Path]:
    """Write the JSON report, and the gap keys beside it when there are any.

    The gaps file holds at most ``MAX_GAP_ROWS`` rows, shared evenly by the pairs so
    that no pair is cut out. A gaps file left by an earlier run is removed when this
    run has no gaps.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    gap_path = report_path.with_suffix(".gaps.csv")
    if gaps.empty:
        gap_path.unlink(missing_ok=True)
        return None
    per_pair = max(1, MAX_GAP_ROWS // gaps["pair"].nunique())
    gaps.groupby("pair").head(per_pair).to_csv(gap_path, index=False)
    return gap_path
