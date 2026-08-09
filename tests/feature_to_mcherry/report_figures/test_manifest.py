"""Tests for feature_to_mcherry.report_figures.manifest."""

from __future__ import annotations

import json
from pathlib import Path

from src.feature_to_mcherry.report_figures.manifest import FigureStatus, ManifestWriter


def test_figure_status_rejects_invalid_status() -> None:
    import pytest

    with pytest.raises(ValueError, match="status"):
        FigureStatus(figure_id="A1", title="x", status="nope")


def test_manifest_writer_round_trips_json(tmp_path: Path) -> None:
    writer = ManifestWriter()
    writer.add(
        FigureStatus(
            figure_id="A1",
            title="Cross-experiment Ridge floor ranking",
            status="generated",
            output_paths=["results/report_figures/a1.png"],
        )
    )
    writer.add(
        FigureStatus(
            figure_id="D3",
            title="scPortrait PCA scree",
            status="blocked",
            missing=["pca_explained_variance"],
            note="no saved PCA artifact on HPC",
        )
    )

    writer.write(tmp_path)

    manifest_json = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest_json) == 2
    assert manifest_json[0]["figure_id"] == "A1"
    assert manifest_json[1]["status"] == "blocked"

    manifest_md = (tmp_path / "manifest.md").read_text()
    assert "A1" in manifest_md
    assert "blocked" in manifest_md
