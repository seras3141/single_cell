"""Observed-subset completeness checks for raw dataset inventories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

DEFAULT_EXPECTED_Z_INDICES = tuple(range(1, 21))
DEFAULT_PROJECTION_Z_INDEX = 0

GROUP_COLUMNS = ["plate_id", "time_point", "well_id"]
ISSUE_COLUMNS = [
    "plate_id",
    "time_point",
    "well_id",
    "row",
    "col",
    "issue_type",
    "severity",
    "channel",
    "z_index",
    "details",
    "expected_file_path",
]
COMPLETENESS_COLUMNS = [
    "plate_id",
    "time_point",
    "well_id",
    "row",
    "col",
    "content",
    "channels_present",
    "z_indices_present",
    "missing_channels",
    "missing_core_z",
    "unexpected_z_indices",
    "n_channels_present",
    "n_core_z_present",
    "n_expected_channels",
    "n_expected_core_z",
    "has_projection",
    "is_channel_complete",
    "is_core_z_complete",
    "is_complete",
]


def _sorted_tuple(values: Iterable[Any]) -> Tuple[Any, ...]:
    return tuple(sorted(values))


def _empty_completeness_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=COMPLETENESS_COLUMNS)


def _empty_issue_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=ISSUE_COLUMNS)


def _group_keys(group: pd.DataFrame) -> Dict[str, Any]:
    first = group.iloc[0]
    return {
        "plate_id": first["plate_id"],
        "time_point": first["time_point"],
        "well_id": first["well_id"],
        "row": first["row"],
        "col": first["col"],
    }


def build_completeness_table(
    inventory: pd.DataFrame,
    expected_channels: Sequence[str],
    expected_z_indices: Sequence[int] = DEFAULT_EXPECTED_Z_INDICES,
    projection_z_index: Optional[int] = DEFAULT_PROJECTION_Z_INDEX,
) -> pd.DataFrame:
    """Summarize channel/z completeness for each observed plate/time/well group."""
    if inventory.empty:
        return _empty_completeness_dataframe()

    expected_channel_set = set(expected_channels)
    expected_z_set = set(int(z) for z in expected_z_indices)
    allowed_z = set(expected_z_set)
    if projection_z_index is not None:
        allowed_z.add(int(projection_z_index))

    records = []
    for _, group in inventory.groupby(GROUP_COLUMNS, sort=True):
        keys = _group_keys(group)
        channels_present = set(group["channel"].dropna().astype(str))
        z_indices_present = set(group["z_index"].dropna().astype(int))
        missing_channels = expected_channel_set - channels_present
        missing_core_z = expected_z_set - z_indices_present
        unexpected_z = z_indices_present - allowed_z
        core_z_present = expected_z_set & z_indices_present
        has_projection = (
            projection_z_index is None or int(projection_z_index) in z_indices_present
        )
        content = group["content"].dropna().iloc[0] if "content" in group else None

        records.append(
            {
                **keys,
                "content": content,
                "channels_present": _sorted_tuple(channels_present),
                "z_indices_present": _sorted_tuple(z_indices_present),
                "missing_channels": _sorted_tuple(missing_channels),
                "missing_core_z": _sorted_tuple(missing_core_z),
                "unexpected_z_indices": _sorted_tuple(unexpected_z),
                "n_channels_present": len(channels_present),
                "n_core_z_present": len(core_z_present),
                "n_expected_channels": len(expected_channel_set),
                "n_expected_core_z": len(expected_z_set),
                "has_projection": has_projection,
                "is_channel_complete": not missing_channels,
                "is_core_z_complete": not missing_core_z,
                "is_complete": (
                    not missing_channels
                    and not missing_core_z
                    and has_projection
                    and not unexpected_z
                ),
            }
        )

    return pd.DataFrame.from_records(records, columns=COMPLETENESS_COLUMNS)


def _issue(
    keys: Dict[str, Any],
    issue_type: str,
    severity: str,
    details: str,
    channel: Optional[str] = None,
    z_index: Optional[int] = None,
    expected_file_path: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        **keys,
        "issue_type": issue_type,
        "severity": severity,
        "channel": channel,
        "z_index": z_index,
        "details": details,
        "expected_file_path": expected_file_path,
    }


def _extract_projection_subdir(
    inventory: pd.DataFrame,
    data_root: Path,
    projection_z_index: int,
) -> Optional[str]:
    """Return the relative subdirectory name for projection (z=0) files, or None."""
    z0_rows = inventory[inventory["z_index"] == projection_z_index]
    if z0_rows.empty:
        return None
    sample_path = Path(str(z0_rows.iloc[0]["file_path"]))
    try:
        rel = sample_path.relative_to(data_root)
    except ValueError:
        return None
    return rel.parts[0] if len(rel.parts) >= 2 else None


def _build_channel_wavelength_map(inventory: pd.DataFrame) -> Dict[str, int]:
    """Map channel name → integer wavelength index from the inventory."""
    result: Dict[str, int] = {}
    for _, row in (
        inventory[["channel", "wavelength"]].dropna().drop_duplicates().iterrows()
    ):
        result[str(row["channel"])] = int(row["wavelength"])
    return result


def _expected_rel_path(
    time_point: int,
    row: str,
    col: int,
    z_index: int,
    wavelength: Optional[int],
    projection_z_index: Optional[int],
    projection_subdir: Optional[str],
    site: int = 1,
) -> str:
    """Construct the expected path relative to data_root for a missing file.

    If *wavelength* is None, uses ``w*`` so the result is a glob pattern.
    """
    w_part = str(wavelength) if wavelength is not None else "*"
    filename = f"t{time_point}_{row}{col:02d}_s{site}_w{w_part}_z{z_index}.tif"
    if (
        projection_z_index is not None
        and z_index == int(projection_z_index)
        and projection_subdir
    ):
        return f"{projection_subdir}/{filename}"
    return filename


def _present_pairs(group: pd.DataFrame) -> Set[Tuple[str, int]]:
    return set(
        zip(
            group["channel"].dropna().astype(str),
            group["z_index"].dropna().astype(int),
        )
    )


def find_dataset_issues(
    inventory: pd.DataFrame,
    expected_channels: Sequence[str],
    expected_z_indices: Sequence[int] = DEFAULT_EXPECTED_Z_INDICES,
    projection_z_index: Optional[int] = DEFAULT_PROJECTION_Z_INDEX,
    data_root: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Create an actionable observed-well issue report.

    When *data_root* is provided, each issue row that corresponds to a single
    missing file includes an ``expected_file_path`` column with the path
    relative to *data_root*.  For issue types that span multiple files
    (``missing_z``, ``missing_projection_z``), the path uses a ``w*`` glob
    for the wavelength component.  For issue types that do not map 1-to-1 to
    a file (``missing_channel``, ``unexpected_*``), the column is ``None``.
    """
    if inventory.empty:
        return _empty_issue_dataframe()

    expected_channels_list = list(dict.fromkeys(expected_channels))
    expected_channel_set = set(expected_channels_list)
    expected_z_list = [int(z) for z in expected_z_indices]
    expected_z_set = set(expected_z_list)
    allowed_z = set(expected_z_set)
    if projection_z_index is not None:
        allowed_z.add(int(projection_z_index))

    # Pre-compute path-construction helpers when data_root is supplied.
    _data_root: Optional[Path] = Path(data_root) if data_root is not None else None
    _projection_subdir: Optional[str] = None
    _ch_wl: Dict[str, int] = {}
    if _data_root is not None:
        _ch_wl = _build_channel_wavelength_map(inventory)
        if projection_z_index is not None:
            _projection_subdir = _extract_projection_subdir(
                inventory, _data_root, int(projection_z_index)
            )

    def _exp_path(z_index: int, channel: Optional[str], keys: Dict[str, Any]) -> Optional[str]:
        if _data_root is None:
            return None
        wavelength = _ch_wl.get(channel) if channel is not None else None
        return _expected_rel_path(
            time_point=int(keys["time_point"]),
            row=str(keys["row"]),
            col=int(keys["col"]),
            z_index=z_index,
            wavelength=wavelength,
            projection_z_index=projection_z_index,
            projection_subdir=_projection_subdir,
        )

    issues: List[Dict[str, Any]] = []
    for _, group in inventory.groupby(GROUP_COLUMNS, sort=True):
        keys = _group_keys(group)
        channels_present = set(group["channel"].dropna().astype(str))
        z_indices_present = set(group["z_index"].dropna().astype(int))
        pairs = _present_pairs(group)

        if "content" in group and (group["content"] == "unmapped").any():
            issues.append(
                _issue(
                    keys,
                    "unexpected_layout",
                    "error",
                    "Observed well is not present in the MF5v1 layout "
                    "annotation table.",
                )
            )

        missing_channels = expected_channel_set - channels_present
        for channel in sorted(missing_channels):
            issues.append(
                _issue(
                    keys,
                    "missing_channel",
                    "error",
                    f"Expected channel '{channel}' is absent for this observed well.",
                    channel=channel,
                )
            )

        for z_index in expected_z_list:
            if z_index not in z_indices_present:
                issues.append(
                    _issue(
                        keys,
                        "missing_z",
                        "error",
                        f"Expected core z-slice z{z_index} is absent.",
                        z_index=z_index,
                        expected_file_path=_exp_path(z_index, None, keys),
                    )
                )
                continue

            for channel in expected_channels_list:
                if channel in missing_channels:
                    continue
                if (channel, z_index) not in pairs:
                    issues.append(
                        _issue(
                            keys,
                            "missing_channel_z",
                            "error",
                            f"Expected channel '{channel}' is missing at z{z_index}.",
                            channel=channel,
                            z_index=z_index,
                            expected_file_path=_exp_path(z_index, channel, keys),
                        )
                    )

        if projection_z_index is not None:
            projection_z = int(projection_z_index)
            if projection_z not in z_indices_present:
                issues.append(
                    _issue(
                        keys,
                        "missing_projection_z",
                        "warning",
                        f"Projection z-slice z{projection_z} is absent.",
                        z_index=projection_z,
                        expected_file_path=_exp_path(projection_z, None, keys),
                    )
                )
            else:
                for channel in expected_channels_list:
                    if channel in missing_channels:
                        continue
                    if (channel, projection_z) not in pairs:
                        issues.append(
                            _issue(
                                keys,
                                "missing_projection_channel",
                                "warning",
                                f"Projection z{projection_z} is missing "
                                f"channel '{channel}'.",
                                channel=channel,
                                z_index=projection_z,
                                expected_file_path=_exp_path(projection_z, channel, keys),
                            )
                        )

        for z_index in sorted(z_indices_present - allowed_z):
            issues.append(
                _issue(
                    keys,
                    "unexpected_z",
                    "error",
                    f"Observed z{z_index}, but expected z-slices are z0 and z1-z20.",
                    z_index=z_index,
                )
            )

    if not issues:
        return _empty_issue_dataframe()

    return pd.DataFrame.from_records(issues, columns=ISSUE_COLUMNS)
