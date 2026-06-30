from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.dataset_analysis import (  # noqa: E402
    build_completeness_table,
    build_dataset_inventory,
    build_dataset_summary,
    build_plate_annotation_dataframe,
    find_dataset_issues,
    load_expected_channels,
    plot_channel_completeness,
    plot_control_distribution,
    plot_drug_distribution,
    plot_plate_coverage,
    plot_z_completeness,
)

LAYOUT_PATH = Path("docs/layout/MF5v1_plate_layout.json")


def _touch_image(
    root: Path,
    well_id: str = "B02",
    wavelength: int = 1,
    z_index: int = 1,
    time_point: int = 1,
) -> Path:
    path = (
        root
        / "Plate 2426"
        / "BF Images"
        / f"t{time_point}_{well_id}_s1_w{wavelength}_z{z_index}.tif"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _build_inventory_for_files(
    tmp_path: Path, files: list[tuple[int, int]]
) -> pd.DataFrame:
    for wavelength, z_index in files:
        _touch_image(tmp_path, wavelength=wavelength, z_index=z_index)
    return build_dataset_inventory(tmp_path, LAYOUT_PATH)


def test_layout_annotation_handles_drugs_and_controls() -> None:
    annotations = build_plate_annotation_dataframe(LAYOUT_PATH)

    b02 = annotations.loc[annotations["well_id"] == "B02"].iloc[0]
    assert b02["content"] == "drug"
    assert b02["drug"] == "Eprenetapopt"
    assert b02["concentration_uM"] == 100.0

    k01 = annotations.loc[annotations["well_id"] == "K01"].iloc[0]
    assert k01["content"] == "empty"

    k02 = annotations.loc[annotations["well_id"] == "K02"].iloc[0]
    assert k02["content"] == "control"
    assert k02["control"] == "Staurosporine"

    m02 = annotations.loc[annotations["well_id"] == "M02"].iloc[0]
    assert m02["control"] == "Benzethonium Chloride"

    m04 = annotations.loc[annotations["well_id"] == "M04"].iloc[0]
    assert m04["control"] == "DMSO"


def test_inventory_parses_raw_filenames_and_joins_layout(tmp_path: Path) -> None:
    inventory = _build_inventory_for_files(tmp_path, [(1, 1), (2, 1), (3, 1)])

    assert len(inventory) == 3
    assert set(inventory["channel"]) == {"BF", "mCherry", "FlipGFP"}
    assert set(inventory["wavelength"]) == {1, 2, 3}
    assert inventory["plate_id"].unique().tolist() == ["p2426"]
    assert inventory["time_point"].unique().tolist() == [1]
    assert inventory["well_id"].unique().tolist() == ["B02"]
    assert inventory["content"].unique().tolist() == ["drug"]
    assert inventory["drug"].unique().tolist() == ["Eprenetapopt"]


def test_expected_channels_load_in_wavelength_order() -> None:
    assert load_expected_channels() == ["FlipGFP", "mCherry", "BF"]


def test_missing_channel_and_projection_are_reported_for_observed_wells_only(
    tmp_path: Path,
) -> None:
    inventory = _build_inventory_for_files(tmp_path, [(1, 1), (2, 1)])
    issues = find_dataset_issues(
        inventory,
        ["BF", "mCherry", "FlipGFP"],
        expected_z_indices=[1],
    )

    assert set(issues["well_id"]) == {"B02"}
    missing_channel = issues[issues["issue_type"] == "missing_channel"].iloc[0]
    assert missing_channel["channel"] == "BF"
    assert missing_channel["severity"] == "error"

    projection = issues[issues["issue_type"] == "missing_projection_z"].iloc[0]
    assert projection["z_index"] == 0
    assert projection["severity"] == "warning"


def test_missing_core_z_and_unexpected_z_are_errors(tmp_path: Path) -> None:
    inventory = _build_inventory_for_files(
        tmp_path,
        [(1, 1), (2, 1), (3, 1), (1, 21)],
    )
    issues = find_dataset_issues(
        inventory,
        ["BF", "mCherry", "FlipGFP"],
        expected_z_indices=[1, 2],
    )

    missing_z = issues[issues["issue_type"] == "missing_z"].iloc[0]
    assert missing_z["z_index"] == 2
    assert missing_z["severity"] == "error"

    unexpected_z = issues[issues["issue_type"] == "unexpected_z"].iloc[0]
    assert unexpected_z["z_index"] == 21
    assert unexpected_z["severity"] == "error"


def test_channel_missing_at_specific_core_z_is_reported(tmp_path: Path) -> None:
    inventory = _build_inventory_for_files(
        tmp_path,
        [(1, 1), (2, 1), (3, 1), (1, 2), (3, 2)],
    )
    issues = find_dataset_issues(
        inventory,
        ["BF", "mCherry", "FlipGFP"],
        expected_z_indices=[1, 2],
    )

    assert "missing_channel" not in set(issues["issue_type"])
    missing_channel_z = issues[issues["issue_type"] == "missing_channel_z"].iloc[0]
    assert missing_channel_z["channel"] == "mCherry"
    assert missing_channel_z["z_index"] == 2
    assert missing_channel_z["severity"] == "error"


def test_summary_and_plot_helpers_smoke(tmp_path: Path) -> None:
    inventory = _build_inventory_for_files(
        tmp_path,
        [(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1)],
    )
    expected_channels = ["BF", "mCherry", "FlipGFP"]
    completeness = build_completeness_table(
        inventory,
        expected_channels,
        expected_z_indices=[1],
    )
    issues = find_dataset_issues(
        inventory,
        expected_channels,
        expected_z_indices=[1],
    )
    summary = build_dataset_summary(
        inventory,
        issues,
        expected_channels,
        expected_z_indices=[1],
    )

    assert completeness["is_complete"].all()
    assert issues.empty
    assert summary["expected_files_for_observed_wells"] == 6

    figures = [
        plot_plate_coverage(inventory),
        plot_channel_completeness(completeness, expected_channels),
        plot_z_completeness(completeness),
        plot_control_distribution(inventory),
        plot_drug_distribution(inventory),
    ]
    for figure in figures:
        assert hasattr(figure, "savefig")
        plt.close(figure)
