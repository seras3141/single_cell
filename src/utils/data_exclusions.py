"""Registry of known-missing inputs and excluded stacks.

Loaded from ``config/data_exclusions.yaml``.

Two kinds of entry, both keyed by the processed-experiment directory name (e.g.
``"HD1883 MF5V1 0-72h 20-03-26"``), because the plate prefix in filenames
(``pMF5V1``) is shared by every culture and cannot identify one:

- ``known_missing``: an input slice that is genuinely absent from the raw data.
  Feature extraction treats the resulting unpaired mask as expected, not as an
  error.
- ``excluded_stacks``: a (sample, timepoint) stack that is present but unusable.
  It is still extracted; analysis loaders drop its rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)

DEFAULT_EXCLUSIONS_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "data_exclusions.yaml"
)


@dataclass(frozen=True)
class KnownMissing:
    """An input slice absent from the raw data."""

    experiment: str
    sample: str
    timepoint: int
    z: int
    channel: str
    reason: str


@dataclass(frozen=True)
class ExcludedStack:
    """A (sample, timepoint) stack whose rows must not enter analysis."""

    experiment: str
    sample: str
    timepoint: int
    reason: str


@dataclass(frozen=True)
class DataExclusions:
    """Parsed registry. Use :func:`load_data_exclusions` to build one."""

    known_missing: Tuple[KnownMissing, ...] = ()
    excluded_stacks: Tuple[ExcludedStack, ...] = ()

    @classmethod
    def empty(cls) -> "DataExclusions":
        """A registry with no entries (disables all exclusions)."""
        return cls()

    @property
    def experiments(self) -> FrozenSet[str]:
        """Every experiment name that has at least one entry."""
        return frozenset(entry.experiment for entry in self.known_missing) | frozenset(
            entry.experiment for entry in self.excluded_stacks
        )

    def experiment_of(self, path: Union[str, Path]) -> Optional[str]:
        """Return the registered experiment whose name is a component of ``path``.

        Matches on path components rather than resolving the filesystem, so it
        needs no directory listing and gives the same answer through the
        ``data/`` symlink.
        """
        parts = set(Path(path).parts)
        matches = [name for name in self.experiments if name in parts]
        return matches[0] if len(matches) == 1 else None

    def is_known_missing(
        self, experiment: str, sample: str, timepoint: int, z: int, channel: str
    ) -> bool:
        """True if this exact input slice is registered as absent."""
        return any(
            entry.experiment == experiment
            and entry.sample == sample
            and entry.timepoint == int(timepoint)
            and entry.z == int(z)
            and entry.channel == channel
            for entry in self.known_missing
        )

    def excluded_stacks_for(self, experiment: str) -> FrozenSet[Tuple[str, int]]:
        """(sample, timepoint) pairs excluded for ``experiment``."""
        return frozenset(
            (entry.sample, entry.timepoint)
            for entry in self.excluded_stacks
            if entry.experiment == experiment
        )


_KNOWN_MISSING_FIELDS: Dict[str, type] = {
    "experiment": str,
    "sample": str,
    "timepoint": int,
    "z": int,
    "channel": str,
    "reason": str,
}
_EXCLUDED_STACK_FIELDS: Dict[str, type] = {
    "experiment": str,
    "sample": str,
    "timepoint": int,
    "reason": str,
}


def _parse_entry(raw: Any, fields: Dict[str, type], section: str, index: int) -> Dict:
    if not isinstance(raw, dict):
        raise ValueError(f"data exclusions: {section}[{index}] must be a mapping")
    missing = [name for name in fields if name not in raw]
    unknown = [name for name in raw if name not in fields]
    if missing or unknown:
        raise ValueError(
            f"data exclusions: {section}[{index}] missing keys {missing}, "
            f"unknown keys {unknown}"
        )
    for name, kind in fields.items():
        value = raw[name]
        # bool is an int subclass; a YAML ``yes`` must not pass as a timepoint.
        if isinstance(value, bool) or not isinstance(value, kind):
            raise ValueError(
                f"data exclusions: {section}[{index}].{name} must be "
                f"{kind.__name__}, got {value!r}"
            )
    return raw


def load_data_exclusions(
    path: Union[str, Path, None] = None,
) -> DataExclusions:
    """Load and validate the exclusions registry.

    Args:
        path: Registry YAML. Defaults to the tracked ``config/data_exclusions.yaml``.
            A missing default file yields an empty registry; a missing explicit
            path raises.

    Raises:
        FileNotFoundError: an explicit ``path`` does not exist.
        ValueError: the file is malformed (wrong shape, missing/unknown keys, types).
    """
    registry_path = DEFAULT_EXCLUSIONS_PATH if path is None else Path(path)
    if not registry_path.exists():
        if path is not None:
            raise FileNotFoundError(f"data exclusions file not found: {registry_path}")
        logger.warning("No data exclusions registry at %s; none applied", registry_path)
        return DataExclusions.empty()

    with open(registry_path) as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"data exclusions: {registry_path} must be a mapping")
    unknown_sections = set(raw) - {"known_missing", "excluded_stacks"}
    if unknown_sections:
        raise ValueError(
            f"data exclusions: unknown sections {sorted(unknown_sections)} "
            f"in {registry_path}"
        )

    known_missing = tuple(
        KnownMissing(**_parse_entry(entry, _KNOWN_MISSING_FIELDS, "known_missing", i))
        for i, entry in enumerate(raw.get("known_missing") or [])
    )
    excluded_stacks = tuple(
        ExcludedStack(
            **_parse_entry(entry, _EXCLUDED_STACK_FIELDS, "excluded_stacks", i)
        )
        for i, entry in enumerate(raw.get("excluded_stacks") or [])
    )
    return DataExclusions(known_missing=known_missing, excluded_stacks=excluded_stacks)
