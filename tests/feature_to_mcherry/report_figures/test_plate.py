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
    # Row C, column 1 -> plate-edge spacer column, not part of any quadrant -> empty.
    drug, conc = well_to_drug_dose("C01", layout)
    assert drug == "Doxorubicin"
    assert conc is None


def test_column_7_is_quadrant_2_top_dose_not_empty(layout: dict) -> None:
    # Regression test: column 7 is quadrant 2's *first* real data column (position 1,
    # the highest concentration), not an empty spacer. A prior version of the layout
    # modelled quadrants as uniform 6-column blocks (empty, then 5 data columns) and
    # wrongly labelled every quadrant-2+ leading column (7, 13, 19) as empty.
    drug, conc = well_to_drug_dose("E07", layout)
    assert drug == "Navitoclax"
    assert conc == 75.0


def test_columns_12_13_are_the_midline_gap(layout: dict) -> None:
    # The 2-column gap between quadrants 2 and 3 sits at columns 12-13, not at the
    # start of quadrant 2 (column 7) or quadrant 3 (column 13 alone).
    for well in ("C12", "C13"):
        drug, conc = well_to_drug_dose(well, layout)
        assert drug == "Doxorubicin"
        assert conc is None


def test_column_24_is_the_trailing_edge_gap(layout: dict) -> None:
    drug, conc = well_to_drug_dose("C24", layout)
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
    # Column 7 is quadrant 2's position 1 -> same tier as column 2, quadrant 1's
    # position 1 (both the top/highest-concentration column of their quadrant).
    drug_q1, conc_q1 = well_to_drug_dose("C02", layout)
    drug_q2, conc_q2 = well_to_drug_dose("C07", layout)
    assert drug_q1 == drug_q2
    assert conc_q1 == conc_q2

    # And column 8 (quadrant 2, position 2) should match column 3 (quadrant 1,
    # position 2) -- NOT column 2, which a uniform 6-column-period reading would
    # have wrongly implied.
    drug_q1b, conc_q1b = well_to_drug_dose("C03", layout)
    drug_q2b, conc_q2b = well_to_drug_dose("C08", layout)
    assert drug_q1b == drug_q2b
    assert conc_q1b == conc_q2b


def test_malformed_well_id_raises() -> None:
    with pytest.raises(ValueError):
        well_to_drug_dose("07", {})
