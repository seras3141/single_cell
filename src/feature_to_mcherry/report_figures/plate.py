"""Well -> drug/dose lookup against ``config/MF5v1_plate_layout.json``.

Plate geometry: 4 identical 6-column quadrants; within each quadrant, column offset 1
is empty and offsets 2-6 carry concentration tiers 1-5 (high to low) for drug rows, or
Benzethonium-Chloride/DMSO for the two control rows (M, N) — those use a *different*
column pattern (``control_column_pattern_within_quadrant``) and don't have a
concentration ladder (there's no ``concentrations_uM`` for controls in the drugs
dict). Row K/L (Staurosporine) are drug-pattern-shaped controls: they have a ``drug``
key but no dose ladder either.
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


def well_to_drug_dose(
    well_id: str, layout: dict
) -> Tuple[Optional[str], Optional[float]]:
    """Map a well id (e.g. ``"E07"``) to ``(drug_or_control_name, concentration_uM)``.

    Either element may be ``None``: no drug/control at that well (empty row or empty
    column offset), or a drug/control with no defined concentration ladder.
    """
    match = _WELL_RE.match(well_id.strip())
    if not match:
        raise ValueError(f"Malformed well id: {well_id!r}")
    row, col_str = match.group(1).upper(), match.group(2)
    col = int(col_str)

    row_info = layout["row_assignments"].get(row)
    if row_info is None or row_info.get("content") == "empty":
        return None, None

    offset = (col - 1) % layout["quadrants"]["columns_per_quadrant"] + 1
    offset_key = f"offset_{offset}"

    if row in _CONTROL_ROWS:
        tier = layout["control_column_pattern_within_quadrant"].get(offset_key)
        if tier is None or tier == "empty":
            return None, None
        return tier, None

    drug = row_info.get("drug")
    tier = layout["column_pattern_within_quadrant"].get(offset_key)
    if tier is None or tier == "empty":
        return drug, None

    concentrations = layout.get("drugs", {}).get(drug, {}).get("concentrations_uM", {})
    return drug, concentrations.get(tier)
