"""Well -> drug/dose lookup against ``config/MF5v1_plate_layout.json``.

Plate geometry: 4 replicate blocks of 5 real data columns each (NOT a uniform
6-column period — see ``quadrants.empty_columns``/``quadrants.Q1``-``Q4`` in the
layout JSON). Empty/spacer columns are the two plate edges (1, 24) and a 2-column
gap at the midline (12, 13); there is no gap between quadrants 1/2 or 3/4. Within
each quadrant, position 1-5 (in column order) carries concentration tiers 1-5
(high to low) for drug rows, or Benzethonium-Chloride/DMSO for the two control
rows (M, N) — those use a *different* column pattern
(``control_column_pattern_within_quadrant``) and don't have a concentration
ladder (there's no ``concentrations_uM`` for controls in the drugs dict). Row K/L
(Staurosporine) are drug-pattern-shaped controls: they have a ``drug`` key but no
dose ladder either.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

_WELL_RE = re.compile(r"^([A-Za-z])(\d+)$")
_CONTROL_ROWS = ("M", "N")


def load_plate_layout(path: Path) -> dict:
    """Load the plate-layout JSON."""
    with Path(path).open() as f:
        return json.load(f)


def plate_grid_shape(layout: dict) -> Tuple[List[str], List[int]]:
    """Row letters and column numbers, from ``layout["dimensions"]``."""
    dims = layout["dimensions"]
    return list(dims["row_labels"]), list(range(1, dims["columns"] + 1))


def _position_within_quadrant(col: int, layout: dict) -> Optional[int]:
    """Position (1-5) of ``col`` within its quadrant's real data columns.

    Returns ``None`` if ``col`` is one of the plate's non-data spacer columns.
    Quadrant boundaries are irregular (edge gaps at columns 1/24, a 2-column
    gap at the midline 12-13, no gap between quadrants 1/2 or 3/4) and are
    looked up explicitly against ``quadrants.empty_columns``/``quadrants.Q1``-
    ``Q4`` — never derived from a modular formula, which would wrongly assume
    a uniform period-6 grid.
    """
    quadrants = layout.get("quadrants", {})
    if col in set(quadrants.get("empty_columns", [])):
        return None
    for quadrant_number in (1, 2, 3, 4):
        columns = quadrants.get(f"Q{quadrant_number}", {}).get("columns", [])
        if col in columns:
            return columns.index(col) + 1
    return None


def well_to_drug_dose(
    well_id: str, layout: dict
) -> Tuple[Optional[str], Optional[float]]:
    """Map a well id (e.g. ``"E07"``) to ``(drug_or_control_name, concentration_uM)``.

    Either element may be ``None``: no drug/control at that well (empty row or a
    non-data spacer column), or a drug/control with no defined concentration ladder.
    """
    match = _WELL_RE.match(well_id.strip())
    if not match:
        raise ValueError(f"Malformed well id: {well_id!r}")
    row, col_str = match.group(1).upper(), match.group(2)
    col = int(col_str)

    row_info = layout["row_assignments"].get(row)
    if row_info is None or row_info.get("content") == "empty":
        return None, None

    position = _position_within_quadrant(col, layout)
    position_key = f"position_{position}" if position is not None else None

    if row in _CONTROL_ROWS:
        tier = (
            layout["control_column_pattern_within_quadrant"].get(position_key)
            if position_key is not None
            else None
        )
        if tier is None or tier == "empty":
            return None, None
        return tier, None

    drug = row_info.get("drug")
    tier = (
        layout["column_pattern_within_quadrant"].get(position_key)
        if position_key is not None
        else None
    )
    if tier is None or tier == "empty":
        return drug, None

    concentrations = layout.get("drugs", {}).get(drug, {}).get("concentrations_uM", {})
    return drug, concentrations.get(tier)
