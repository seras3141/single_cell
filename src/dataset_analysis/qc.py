"""Observed-subset completeness checks for raw dataset inventories."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
) -> Dict[str, Any]:
    return {
        **keys,
        "issue_type": issue_type,
        "severity": severity,
        "channel": channel,
        "z_index": z_index,
        "details": details,
    }


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
) -> pd.DataFrame:
    """Create an actionable observed-well issue report."""
    if inventory.empty:
        return _empty_issue_dataframe()

    expected_channels_list = list(dict.fromkeys(expected_channels))
    expected_channel_set = set(expected_channels_list)
    expected_z_list = [int(z) for z in expected_z_indices]
    expected_z_set = set(expected_z_list)
    allowed_z = set(expected_z_set)
    if projection_z_index is not None:
        allowed_z.add(int(projection_z_index))

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
