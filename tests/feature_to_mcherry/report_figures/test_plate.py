"""Tests for feature_to_mcherry.report_figures.plate, against the real plate layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.feature_to_mcherry.report_figures.plate import (
    load_plate_layout,
    plate_grid_shape,
    well_to_drug_dose,
)

PLATE_LAYOUT_PATH = Path("config/MF5v1_plate_layout.json")


@pytest.fixture(scope="module")
def layout() -> dict:
    return load_plate_layout(PLATE_LAYOUT_PATH)


def test_plate_grid_shape(layout: dict) -> None:
    rows, cols = plate_grid_shape(layout)
    assert len(rows) == 16
    assert len(cols) == 24
    assert rows[0] == "A"
    assert cols[0] == 1 and cols[-1] == 24


def test_row_c_is_doxorubicin_highest_conc(layout: dict) -> None:
    # Row C, column 2 -> quadrant 1, offset 2 -> conc_1 (highest).
    drug, conc = well_to_drug_dose("C02", layout)
    assert drug == "Doxorubicin"
    assert conc == 1.0


def test_offset_1_is_empty_no_drug(layout: dict) -> None:
    # Row C, column 1 -> offset 1 -> empty.
    drug, conc = well_to_drug_dose("C01", layout)
    assert drug == "Doxorubicin"
    assert conc is None


def test_empty_row_returns_none_none(layout: dict) -> None:
    drug, conc = well_to_drug_dose("A05", layout)
    assert drug is None
    assert conc is None


def test_control_row_m_benzethonium(layout: dict) -> None:
    drug, conc = well_to_drug_dose("M02", layout)
    assert drug == "Benzethonium Chloride"
    assert conc is None


def test_control_row_k_staurosporine_has_no_concentration_ladder(layout: dict) -> None:
    drug, conc = well_to_drug_dose("K03", layout)
    assert drug == "Staurosporine"
    assert conc is None


def test_replicated_across_quadrants(layout: dict) -> None:
    # Column 8 is quadrant 2, offset 2 -> same tier as column 2 (quadrant 1, offset 2).
    drug_q1, conc_q1 = well_to_drug_dose("C02", layout)
    drug_q2, conc_q2 = well_to_drug_dose("C08", layout)
    assert drug_q1 == drug_q2
    assert conc_q1 == conc_q2


def test_malformed_well_id_raises() -> None:
    with pytest.raises(ValueError):
        well_to_drug_dose("07", {})
