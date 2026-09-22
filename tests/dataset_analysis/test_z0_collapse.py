"""Tests for z0 collapse metrics, merging-vs-loss signatures, and per-drug figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_analysis.z0_collapse import (
    SIGNATURE_AMBIGUOUS,
    build_shape_comparison,
    plot_shape_grid,
    reduce_over_z,
    SIGNATURE_COLUMNS,
    SIGNATURE_LOSS,
    SIGNATURE_MERGING,
    SIGNATURE_STABLE,
    plot_drug_dose_panel,
    summarize_z0_collapse,
    summarize_z0_signature,
    write_drug_dose_figures,
)

LAYOUT_PATH = Path(__file__).resolve().parents[2] / "config" / "MF5v1_plate_layout.json"
LAYOUT = json.loads(LAYOUT_PATH.read_text())

TIMEPOINTS = [1, 11, 21, 31, 41, 51]


def _trajectory(well, counts, coverages, areas, saturated=False, error=None):
    """One well's z0_population rows."""
    return pd.DataFrame(
        {
            "sample_id": well,
            "timepoint": TIMEPOINTS[: len(counts)],
            "ti": TIMEPOINTS[: len(counts)],
            "n_objects": counts,
            "coverage_fraction": coverages,
            "mean_object_area_fraction": areas,
            "max_label": counts,
            "dtype": "uint16",
            "dtype_saturated": saturated,
            "fov_pixels": 1000,
            "error": error,
        }
    )


def _merging_well(well="E07"):
    """Count halves, coverage holds, objects grow: real aggregation."""
    counts = [400, 350, 300, 250, 220, 200]
    coverage = [0.30, 0.29, 0.28, 0.27, 0.26, 0.26]
    areas = [c / n for c, n in zip(coverage, counts)]
    return _trajectory(well, counts, coverage, areas)


def _loss_well(well="E08"):
    """Count and coverage both collapse, object size flat: detections disappearing."""
    counts = [400, 200, 100, 50, 25, 12]
    coverage = [0.30, 0.15, 0.075, 0.037, 0.019, 0.009]
    areas = [c / n for c, n in zip(coverage, counts)]
    return _trajectory(well, counts, coverage, areas)


def _stable_well(well="M11"):
    counts = [400, 395, 405, 398, 402, 399]
    coverage = [0.30, 0.30, 0.31, 0.30, 0.30, 0.30]
    areas = [c / n for c, n in zip(coverage, counts)]
    return _trajectory(well, counts, coverage, areas)


class TestSummarizeZ0Signature:
    def test_emits_the_declared_columns(self):
        df = summarize_z0_signature(_merging_well(), "TestExp")
        assert list(df.columns) == list(SIGNATURE_COLUMNS)

    def test_identifies_merging(self):
        df = summarize_z0_signature(_merging_well(), "TestExp")
        assert df.loc[0, "signature"] == SIGNATURE_MERGING
        assert df.loc[0, "mean_area_ratio"] > 1.15
        assert df.loc[0, "coverage_retained"] > 0.5

    def test_identifies_detection_loss(self):
        df = summarize_z0_signature(_loss_well(), "TestExp")
        assert df.loc[0, "signature"] == SIGNATURE_LOSS
        assert df.loc[0, "coverage_retained"] < 0.5
        assert df.loc[0, "mean_area_ratio"] == pytest.approx(1.0, abs=0.05)

    def test_a_flat_well_is_stable_not_a_verdict(self):
        df = summarize_z0_signature(_stable_well(), "TestExp")
        assert df.loc[0, "signature"] == SIGNATURE_STABLE

    def test_mixed_evidence_is_ambiguous_not_forced(self):
        """Coverage held but objects did not grow — the §5 table's incoherent cell."""
        counts = [400, 300, 200, 150, 120, 100]
        coverage = [0.30, 0.29, 0.28, 0.27, 0.27, 0.26]
        areas = [0.30 / 400] * 6  # size pinned flat despite the count falling
        df = summarize_z0_signature(_trajectory("E09", counts, coverage, areas), "X")
        assert df.loc[0, "signature"] == SIGNATURE_AMBIGUOUS

    def test_one_row_per_well(self):
        table = pd.concat([_merging_well("E07"), _loss_well("E08")])
        df = summarize_z0_signature(table, "TestExp")
        assert list(df["well"]) == ["E07", "E08"]
        assert list(df["signature"]) == [SIGNATURE_MERGING, SIGNATURE_LOSS]

    def test_carries_annotation_from_the_collapse_summary(self):
        summary = pd.DataFrame(
            {
                "experiment": ["TestExp"],
                "well": ["E07"],
                "drug": ["Navitoclax"],
                "dose_rank": [1.0],
                "concentration_uM": [75.0],
                "is_dmso": [False],
            }
        )
        df = summarize_z0_signature(
            _merging_well(), "TestExp", collapse_summary=summary
        )
        assert df.loc[0, "drug"] == "Navitoclax"
        assert df.loc[0, "concentration_uM"] == 75.0

    def test_rejects_a_frame_missing_a_measure(self):
        table = _merging_well().drop(columns=["coverage_fraction"])
        with pytest.raises(ValueError, match="coverage_fraction"):
            summarize_z0_signature(table, "TestExp")

    def test_all_background_well_does_not_crash(self):
        zeros = [0] * 6
        table = _trajectory("E10", zeros, zeros, [float("nan")] * 6)
        df = summarize_z0_signature(table, "TestExp")
        assert len(df) == 1


class TestSummarizeZ0Collapse:
    def _write(self, tmp_path, name, frame):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        return path

    def test_recomputes_t_cross_on_the_z0_counts(self, tmp_path):
        table = pd.concat([_loss_well("E07"), _stable_well("M11")])
        path = self._write(tmp_path, "Exp1", table)

        summary = summarize_z0_collapse({"Exp1": path}, dmso_wells={"Exp1": "M11"})

        assert set(summary["well"]) == {"E07", "M11"}
        collapsed = summary[summary["well"] == "E07"].iloc[0]
        assert collapsed["t_cross_peak"] is not None
        assert not pd.isna(collapsed["t_cross_peak"])
        healthy = summary[summary["well"] == "M11"].iloc[0]
        assert pd.isna(healthy["t_cross_peak"])

    def test_excludes_width_flagged_rows_from_the_metric(self, tmp_path):
        """Plan §6.2: a flagged count must never be consumed as a measurement."""
        clean = _loss_well("E07")
        flagged = clean.copy()
        flagged["dtype_saturated"] = [False, False, True, True, True, True]
        summary_clean = summarize_z0_collapse(
            {"Exp1": self._write(tmp_path, "clean", clean)}
        )
        summary_flagged = summarize_z0_collapse(
            {"Exp1": self._write(tmp_path, "flagged", flagged)}
        )
        assert (
            summary_flagged.loc[0, "end_n_cells"] != summary_clean.loc[0, "end_n_cells"]
        )

    def test_excludes_unreadable_rows(self, tmp_path):
        table = _loss_well("E07")
        table["error"] = [None, None, None, None, "boom", "boom"]
        summary = summarize_z0_collapse({"Exp1": self._write(tmp_path, "e", table)})
        # ti 41/51 are dropped, so the trailing-3 mean is taken over ti 11/21/31.
        assert summary.loc[0, "end_n_cells"] == pytest.approx(
            np.mean([200.0, 100.0, 50.0])
        )

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no z0_population.csv"):
            summarize_z0_collapse({"Exp1": tmp_path / "absent.csv"})

    def test_no_experiments_raises(self):
        with pytest.raises(ValueError, match="no experiments"):
            summarize_z0_collapse({})


class TestDrugDoseFigures:
    def _series_and_wells(self):
        series = pd.concat(
            [_merging_well("E07"), _loss_well("E08"), _stable_well("M11")]
        )
        wells = pd.DataFrame(
            {
                "well": ["E07", "E08", "M11"],
                "drug": ["Navitoclax", "Navitoclax", "DMSO"],
                "dose_rank": [1.0, 2.0, np.nan],
                "concentration_uM": [75.0, 25.0, np.nan],
                "is_dmso": [False, False, True],
            }
        )
        return series, wells

    def test_writes_a_panel_figure(self, tmp_path):
        series, wells = self._series_and_wells()
        out = tmp_path / "panel.png"
        plot_drug_dose_panel(series, wells, "Exp1", "Navitoclax", out, dmso_well="M11")
        assert out.is_file() and out.stat().st_size > 0

    def test_unknown_drug_raises(self, tmp_path):
        series, wells = self._series_and_wells()
        with pytest.raises(ValueError, match="no wells for drug"):
            plot_drug_dose_panel(
                series, wells, "Exp1", "Nonexistent", tmp_path / "x.png"
            )

    def test_one_figure_per_drug_excluding_dmso(self, tmp_path):
        series, wells = self._series_and_wells()
        written = write_drug_dose_figures(series, wells, "Exp1", tmp_path, "M11")
        assert [p.name for p in written] == ["z0_confluence_Navitoclax.png"]
        assert all(p.is_file() for p in written)

    def test_creates_the_output_directory(self, tmp_path):
        series, wells = self._series_and_wells()
        nested = tmp_path / "a" / "b"
        write_drug_dose_figures(series, wells, "Exp1", nested, "M11")
        assert nested.is_dir()


class TestClassifyEdgeCases:
    """Total collapse and unmeasurable wells must not read as healthy."""

    def test_a_well_losing_every_detection_is_loss_not_stable(self):
        counts = [500, 400, 200, 0, 0, 0]
        coverage = [0.30, 0.24, 0.12, 0.0, 0.0, 0.0]
        areas = [0.30 / 500] * 3 + [float("nan")] * 3
        df = summarize_z0_signature(_trajectory("E07", counts, coverage, areas), "X")
        assert np.isinf(df.loc[0, "count_drop_factor"])
        assert df.loc[0, "signature"] == SIGNATURE_LOSS

    def test_an_unmeasurable_well_is_ambiguous_not_stable(self):
        nans = [float("nan")] * 6
        df = summarize_z0_signature(_trajectory("E08", nans, nans, nans), "X")
        assert df.loc[0, "signature"] == SIGNATURE_AMBIGUOUS


class TestBuildShapeComparison:
    def _z0(self):
        return pd.DataFrame(
            {
                "sample_id": ["E07"] * 4,
                "ti": [1, 11, 21, 31],
                "n_objects": [400.0, 200.0, 100.0, 50.0],
            }
        )

    def _tracked(self):
        return pd.DataFrame(
            {
                "sample_id": ["E07"] * 4,
                "ti": [1, 11, 21, 31],
                "n_cells": [800.0, 400.0, 200.0, 100.0],
            }
        )

    def test_normalises_each_source_to_its_own_peak(self):
        """Absolute counts differ between trees; only the shape may be compared."""
        out = build_shape_comparison(
            {"z0": (self._z0(), "n_objects"), "tracked": (self._tracked(), "n_cells")},
            "Exp1",
        )
        z0 = out[out["source"] == "z0"].sort_values("ti")
        tracked = out[out["source"] == "tracked"].sort_values("ti")
        # counts differ 2x throughout, so the normalised shapes must coincide
        assert list(z0["value_norm"]) == pytest.approx([1.0, 0.5, 0.25, 0.125])
        assert list(tracked["value_norm"]) == pytest.approx([1.0, 0.5, 0.25, 0.125])

    def test_emits_the_declared_columns(self):
        out = build_shape_comparison({"z0": (self._z0(), "n_objects")}, "Exp1")
        assert list(out.columns) == [
            "experiment",
            "well",
            "ti",
            "source",
            "value",
            "value_norm",
        ]

    def test_drops_unmeasurable_rows(self):
        frame = self._z0()
        frame.loc[1, "n_objects"] = float("nan")
        out = build_shape_comparison({"z0": (frame, "n_objects")}, "Exp1")
        assert len(out) == 3

    def test_all_zero_well_does_not_divide_by_zero(self):
        frame = pd.DataFrame(
            {"sample_id": ["E07"] * 3, "ti": [1, 11, 21], "n_objects": [0.0, 0.0, 0.0]}
        )
        out = build_shape_comparison({"z0": (frame, "n_objects")}, "Exp1")
        assert out["value_norm"].isna().all()

    def test_rejects_a_frame_missing_a_column(self):
        with pytest.raises(ValueError, match="missing columns"):
            build_shape_comparison({"z0": (self._z0(), "absent")}, "Exp1")

    def test_no_sources_raises(self):
        with pytest.raises(ValueError, match="no sources"):
            build_shape_comparison({}, "Exp1")

    def test_plots_a_facet_grid(self, tmp_path):
        out = build_shape_comparison({"z0": (self._z0(), "n_objects")}, "Exp1")
        path = plot_shape_grid(out, "Exp1", tmp_path / "grid.png", dmso_well="E07")
        assert path.is_file() and path.stat().st_size > 0

    def test_plotting_an_empty_comparison_raises(self, tmp_path):
        full = build_shape_comparison({"z0": (self._z0(), "n_objects")}, "Exp1")
        empty = full.iloc[:0]
        with pytest.raises(ValueError, match="no wells"):
            plot_shape_grid(empty, "Exp1", tmp_path / "x.png")


class TestReduceOverZ:
    def _multiz(self):
        return pd.DataFrame(
            {
                "sample_id": ["M11"] * 6,
                "ti": [1, 1, 1, 11, 11, 11],
                "z_index": [1, 2, 3, 1, 2, 3],
                "n_objects": [100.0, 250.0, 180.0, 40.0, 30.0, 60.0],
                "coverage_fraction": [0.10, 0.25, 0.18, 0.04, 0.03, 0.06],
            }
        )

    def test_picks_the_plane_with_the_most_detections(self):
        out = reduce_over_z(self._multiz()).sort_values("ti")
        assert list(out["best_slice_n_objects"]) == [250.0, 60.0]
        assert list(out["best_z"]) == [2, 3]

    def test_reports_the_mean_across_planes_too(self):
        out = reduce_over_z(self._multiz()).sort_values("ti")
        assert out.iloc[0]["mean_z_coverage"] == pytest.approx((0.10 + 0.25 + 0.18) / 3)
        assert list(out["n_planes"]) == [3, 3]

    def test_all_unmeasurable_timepoint_yields_nan_not_a_crash(self):
        frame = self._multiz()
        frame.loc[frame["ti"] == 11, "n_objects"] = float("nan")
        out = reduce_over_z(frame).sort_values("ti")
        assert np.isnan(out.iloc[1]["best_slice_n_objects"])
        assert out.iloc[1]["n_planes"] == 0

    def test_rejects_a_frame_without_z_index(self):
        with pytest.raises(ValueError, match="z_index"):
            reduce_over_z(self._multiz().drop(columns=["z_index"]))


class TestMeanAreaAnchoring:
    def test_an_all_background_first_frame_does_not_force_ambiguous(self):
        """first_area averaged over the edge window, like end_area.

        Anchoring on a single frame let one NaN (and NaN is truthy, so the old
        `if first_area` guard missed it) demote a genuinely merging well.
        """
        counts = [400, 350, 300, 250, 220, 200]
        coverage = [0.30, 0.29, 0.28, 0.27, 0.26, 0.26]
        areas = [float("nan")] + [c / n for c, n in zip(coverage[1:], counts[1:])]
        df = summarize_z0_signature(_trajectory("E07", counts, coverage, areas), "X")
        assert np.isfinite(df.loc[0, "mean_area_ratio"])
        assert df.loc[0, "signature"] == SIGNATURE_MERGING
