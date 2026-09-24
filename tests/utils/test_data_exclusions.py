"""Tests for the data-exclusions registry (``src/utils/data_exclusions.py``)."""

from pathlib import Path

import pytest

from src.utils.data_exclusions import (
    DEFAULT_EXCLUSIONS_PATH,
    DataExclusions,
    ExcludedStack,
    KnownMissing,
    load_data_exclusions,
)

EXP = "HD1883 MF5V1 0-72h 20-03-26"

VALID = f"""
known_missing:
  - {{experiment: "{EXP}", sample: H09, timepoint: 201, z: 4, channel: BF, reason: r}}
excluded_stacks:
  - {{experiment: "{EXP}", sample: H09, timepoint: 201, reason: r}}
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "exclusions.yaml"
    path.write_text(text)
    return path


def test_valid_registry_parses(tmp_path):
    reg = load_data_exclusions(_write(tmp_path, VALID))
    assert reg.known_missing == (KnownMissing(EXP, "H09", 201, 4, "BF", "r"),)
    assert reg.excluded_stacks == (ExcludedStack(EXP, "H09", 201, "r"),)
    assert reg.is_known_missing(EXP, "H09", 201, 4, "BF")
    assert not reg.is_known_missing(EXP, "H09", 201, 5, "BF")
    assert not reg.is_known_missing(EXP, "H09", 201, 4, "mCherry")
    assert reg.excluded_stacks_for(EXP) == frozenset({("H09", 201)})
    assert reg.excluded_stacks_for("other") == frozenset()


def test_tracked_default_registry_loads():
    assert DEFAULT_EXCLUSIONS_PATH.exists()
    assert isinstance(load_data_exclusions(), DataExclusions)  # parses cleanly


def _stack(fields: str) -> str:
    return f'excluded_stacks:\n  - {{experiment: "{EXP}", {fields}}}\n'


@pytest.mark.parametrize(
    "text, message",
    [
        (_stack("sample: H09, reason: r"), "missing keys"),
        (_stack("sample: H09, timepoint: 201, reason: r, well: x"), "unknown keys"),
        (_stack("sample: H09, timepoint: t201, reason: r"), "timepoint must be int"),
        (_stack("sample: H09, timepoint: yes, reason: r"), "timepoint must be int"),
        ("stacks: []\n", "unknown sections"),
        ("- a\n", "must be a mapping"),
    ],
)
def test_malformed_registry_raises(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_data_exclusions(_write(tmp_path, text))


def test_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data_exclusions(tmp_path / "nope.yaml")


def test_empty_file_is_empty_registry(tmp_path):
    assert load_data_exclusions(_write(tmp_path, "")) == DataExclusions.empty()


def test_experiment_of_matches_path_component(tmp_path):
    reg = load_data_exclusions(_write(tmp_path, VALID))
    mask = Path("data/base") / EXP / "inference_tracked/cellpose_sam/final_2d/x.tif"
    assert reg.experiment_of(mask) == EXP
    # A prefix or substring of a component is not a match.
    assert reg.experiment_of(Path("data") / (EXP + "_copy") / "x.tif") is None
    assert reg.experiment_of("results/HD1883/x.csv") is None
    assert DataExclusions.empty().experiment_of(mask) is None
