"""Workflow helpers for the threshold activity notebook and scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re
import shutil

from cell_activity_labeler.utils.file_utils import (
    ConfigurableFileHandler,
    copy_without_split_dict,
    list_all_files,
    rename_all_files,
)


def create_configured_file_handler(
    wavelength_mappings: dict[int, str] | None = None,
    plate_number: str | None = None,
) -> ConfigurableFileHandler:
    """Create a file handler from optional wavelength and plate settings."""
    clean_mappings = {
        int(index): channel
        for index, channel in (wavelength_mappings or {}).items()
        if str(channel).strip()
    }
    return ConfigurableFileHandler(
        wavelength_mappings=clean_mappings or None,
        plate_number=plate_number or None,
    )


def discover_raw_metadata(raw_dir: Path | str) -> tuple[list[str], list[str], list[str]]:
    """Discover time points, sample IDs, and plate numbers in a raw data directory."""
    raw_path = Path(raw_dir).expanduser()
    if not raw_path.exists():
        return [], [], []

    time_points: set[str] = set()
    sample_ids: set[str] = set()
    plate_numbers: set[str] = set()

    all_files = list(raw_path.rglob("*.tif")) + list(raw_path.rglob("*.tiff"))
    for file_path in all_files:
        plate_match = re.search(r"Plate\s*(\d+)", str(file_path))
        if plate_match:
            plate_numbers.add(f"Plate {plate_match.group(1)}")

        sample_match = re.search(r"([A-Z]\d+)", file_path.name)
        if sample_match:
            sample_ids.add(sample_match.group(1))

        time_match = re.search(r"t(\d+)_", file_path.name)
        if time_match:
            time_points.add(f"t{time_match.group(1)}")

    return sorted(time_points), sorted(sample_ids), sorted(plate_numbers)


def discover_preprocessed_samples(
    prep_dir: Path | str,
    file_handler: ConfigurableFileHandler,
    detail_logger=None,
) -> tuple[list[str], list[str]]:
    """Discover sample IDs and plate numbers in a preprocessed data directory."""
    prep_path = Path(prep_dir).expanduser()
    if not prep_path.exists():
        return [], []

    sample_ids: set[str] = set()
    plate_numbers: set[str] = set()
    all_files = list(prep_path.rglob("*.tif")) + list(prep_path.rglob("*.tiff"))

    for file_path in all_files:
        plate_match = file_handler.extract_plate_number(file_path.name)
        sample_match = file_handler.extract_sample_id(file_path.name)

        if plate_match:
            plate_numbers.add(plate_match)
        if sample_match:
            sample_ids.add(sample_match)
        if detail_logger is not None:
            detail_logger(
                f"File: {file_path.name}, Extracted sample ID: {sample_match}, "
                f"Extracted plate number: {plate_match}"
            )

    return sorted(sample_ids), sorted(plate_numbers)


def preprocess_raw_files(
    raw_dir: Path | str,
    output_dir: Path | str,
    file_handler: ConfigurableFileHandler,
    selected_times: Iterable[str] | None = None,
    selected_samples: Iterable[str] | None = None,
) -> int:
    """Copy and rename raw files matching selected time points or samples."""
    raw_path = Path(raw_dir).expanduser()
    prep_path = Path(output_dir).expanduser()
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_path}")

    selected_time_set = {value.strip() for value in selected_times or [] if value.strip()}
    selected_sample_set = {value.strip().upper() for value in selected_samples or [] if value.strip()}
    if not (selected_time_set or selected_sample_set):
        raise ValueError("Select at least one time point or sample.")

    all_files = list_all_files(str(raw_path), file_handler=file_handler)
    renamed_file_tuples = rename_all_files(all_files, file_handler)
    filtered_files: dict[str, list[tuple[str, str]]] = {}

    for file_type, file_pairs in renamed_file_tuples.items():
        filtered_files[file_type] = []
        for src, renamed in file_pairs:
            if selected_sample_set:
                sample_match = re.search(r"([A-Z]\d+)", renamed)
                if not sample_match or sample_match.group(1).upper() not in selected_sample_set:
                    continue

            if selected_time_set:
                time_match = re.search(r"t(\d+)_", renamed)
                if not time_match or f"t{time_match.group(1)}" not in selected_time_set:
                    continue

            filtered_files[file_type].append((src, renamed))

    if not any(filtered_files.values()):
        return 0

    prep_path.mkdir(parents=True, exist_ok=True)
    return copy_without_split_dict(filtered_files, prep_path)


def split_preprocessed_by_sample(
    prep_dir: Path | str,
    output_dir: Path | str,
    file_handler: ConfigurableFileHandler,
    negative_controls: Iterable[str] | None = None,
    samples: Iterable[str] | None = None,
) -> dict[str, object]:
    """Split preprocessed TIFF files into negative-control and sample folders."""
    prep_path = Path(prep_dir).expanduser()
    split_path = Path(output_dir).expanduser()
    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessed directory does not exist: {prep_path}")

    neg_controls = {value.strip().upper() for value in negative_controls or [] if value.strip()}
    sample_ids = {value.strip().upper() for value in samples or [] if value.strip()}
    if not (neg_controls or sample_ids):
        raise ValueError("Specify at least one negative control or sample cell.")

    all_files = list(prep_path.rglob("*.tif")) + list(prep_path.rglob("*.tiff"))
    if not all_files:
        return {
            "available_samples": [],
            "valid_negative_controls": set(),
            "valid_samples": set(),
            "invalid_negative_controls": neg_controls,
            "invalid_samples": sample_ids,
            "controls_count": 0,
            "samples_count": 0,
            "unmatched": 0,
            "controls_dir": split_path / "negative_controls",
            "samples_dir": split_path / "samples",
        }

    available_samples = {
        sample_id.upper()
        for file_path in all_files
        if (sample_id := file_handler.extract_sample_id(file_path.name))
    }
    valid_negative_controls = neg_controls & available_samples
    valid_samples = sample_ids & available_samples

    controls_dir = split_path / "negative_controls"
    samples_dir = split_path / "samples"
    controls_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    controls_count = 0
    samples_count = 0
    unmatched = 0

    for src_file in all_files:
        sample_id = file_handler.extract_sample_id(src_file.name)
        if sample_id is None:
            unmatched += 1
            continue

        sample_id_upper = sample_id.upper()
        if sample_id_upper in valid_negative_controls:
            dst_dir = controls_dir
            controls_count += 1
        elif sample_id_upper in valid_samples:
            dst_dir = samples_dir
            samples_count += 1
        else:
            continue

        dst_file = dst_dir / src_file.name
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)

    return {
        "available_samples": sorted(available_samples),
        "valid_negative_controls": valid_negative_controls,
        "valid_samples": valid_samples,
        "invalid_negative_controls": neg_controls - available_samples,
        "invalid_samples": sample_ids - available_samples,
        "controls_count": controls_count,
        "samples_count": samples_count,
        "unmatched": unmatched,
        "controls_dir": controls_dir,
        "samples_dir": samples_dir,
    }
