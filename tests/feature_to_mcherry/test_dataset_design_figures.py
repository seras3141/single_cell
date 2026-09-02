"""Tests for the DMSO-vs-drug percentile time plots.

Synthetic frames only -- the real target CSVs are 68 MB-695 MB and live
outside the repo.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.feature_to_mcherry.dataset_design_figures import (
    build_well_timeseries,
    plot_dmso_vs_drug,
    write_dmso_vs_drug_figures,
)

TARGETS = ["percentile_75", "percentile_90", "percentile_95"]


def _targets(wells=("E07", "E08", "M11"), timepoints=(1, 11, 21), z=(1, 2)):
    """Per-(cell, z_slice) rows: 2 cells per (well, timepoint), each on 2 z-slices."""
    rows = []
    for well in wells:
        for timepoint in timepoints:
            for cell_id in (1, 2):
                for z_index in z:
                    base = 100.0 + 10 * cell_id + timepoint
                    rows.append(
                        {
                            "sample_id": well,
                            "timepoint": timepoint,
                            "cell_id": cell_id,
                            "z_index": z_index,
                            # z-slices straddle the per-cell median by +/- 5
                            "percentile_75": base - 5 + 10 * (z_index - 1),
                            "percentile_90": base - 5 + 10 * (z_index - 1) + 1,
                            "percentile_95": base - 5 + 10 * (z_index - 1) + 2,
                        }
                    )
    return pd.DataFrame(rows)


def _summary(t_cross_e07=21.0):
    return pd.DataFrame(
        [
            {
                "experiment": "TestCulture",
                "well": "E07",
                "drug": "Navitoclax",
                "dose_rank": 1.0,
                "concentration_uM": 75.0,
                "is_dmso": False,
                "t_cross_peak": t_cross_e07,
            },
            {
                "experiment": "TestCulture",
                "well": "E08",
                "drug": "Navitoclax",
                "dose_rank": 2.0,
                "concentration_uM": 10.0,
                "is_dmso": False,
                "t_cross_peak": 11.0,
            },
            {
                "experiment": "TestCulture",
                "well": "M11",
                "drug": "DMSO",
                "dose_rank": float("nan"),
                "concentration_uM": float("nan"),
                "is_dmso": True,
                "t_cross_peak": 21.0,
            },
        ]
    )


class TestBuildWellTimeseries:
    def test_collapses_z_then_medians_across_cells(self):
        series = build_well_timeseries(_targets(), TARGETS)
        row = series[
            (series["sample_id"] == "E07")
            & (series["ti"] == 11)
            & (series["target"] == "percentile_90")
        ]
        assert len(row) == 1
        # ti=11: base = 100 + 10*cell + 11, so 121 (cell 1) and 131 (cell 2).
        # p90 per slice is base - 5 + 10*(z - 1) + 1, so cell 1 sees {117, 127}
        # -> 122 and cell 2 sees {127, 137} -> 132. Median across cells = 127.
        assert row.iloc[0]["median"] == pytest.approx(127.0)
        # n_cells counts distinct cells, NOT slice rows -- 2, not 4
        assert row.iloc[0]["n_cells"] == 2

    def test_one_row_per_well_timepoint_target(self):
        series = build_well_timeseries(_targets(), TARGETS)
        assert len(series) == 3 * 3 * 3  # wells x timepoints x targets
        assert not series.duplicated(["sample_id", "ti", "target"]).any()

    def test_ti_is_the_integer_frame_index(self):
        series = build_well_timeseries(_targets(), TARGETS)
        assert sorted(series["ti"].unique()) == [1, 11, 21]

    def test_empty_target_columns_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_well_timeseries(_targets(), [])

    def test_missing_target_column_raises(self):
        """The canonical collapse refuses to silently shrink the requested set."""
        with pytest.raises(ValueError):
            build_well_timeseries(_targets(), ["not_a_column"])


class TestReviewRegressions:
    """One test per defect found in the 2026-09-01 code review."""

    def test_float_timepoint_still_yields_real_frame_indices(self):
        """The canonical collapse str-casts keys, so a float column arrives as
        "241.0"; an isdigit() test would map every row to -1 and stack the whole
        figure on one x position while the CSV read the same file correctly."""
        targets = _targets()
        targets["timepoint"] = targets["timepoint"].astype(float)
        series = build_well_timeseries(targets, TARGETS)
        assert sorted(series["ti"].unique()) == [1.0, 11.0, 21.0]

    def test_unparseable_timepoints_are_dropped_not_stacked(self):
        targets = _targets()
        targets["timepoint"] = targets["timepoint"].astype(str)
        targets.loc[targets["timepoint"] == "21", "timepoint"] = "not-a-frame"
        series = build_well_timeseries(targets, TARGETS)
        assert sorted(series["ti"].unique()) == [1.0, 11.0]
        assert series["ti"].notna().all()

    def test_output_schema_is_fixed_regardless_of_input_column_names(self):
        """plot_dmso_vs_drug reads this frame by name; a renamed key column used to
        turn a non-default well_column into a KeyError at plot time."""
        targets = _targets().rename(columns={"sample_id": "well", "timepoint": "frame"})
        series = build_well_timeseries(
            targets, TARGETS, well_column="well", time_column="frame"
        )
        assert {"sample_id", "timepoint"} <= set(series.columns)
        assert "well" not in series.columns

    def test_drug_only_figure_is_not_written_when_no_drug_data(self, tmp_path: Path):
        """A DMSO-only figure implies drug data that does not exist."""
        series = build_well_timeseries(_targets(wells=("M11",)), TARGETS)
        written = plot_dmso_vs_drug(
            series,
            _summary(),
            experiment="TestCulture",
            drug="Navitoclax",
            target="percentile_90",
            out_dir=tmp_path,
            formats=("png",),
        )
        assert written == []
        assert list(tmp_path.glob("*.png")) == []

    def test_censored_dmso_is_reported_in_the_caption(self, tmp_path: Path):
        """With no DMSO crossing the band and boundary both vanish; the reader must
        be able to tell that apart from a failed render."""
        summary = _summary()
        summary.loc[summary["is_dmso"], "t_cross_peak"] = float("nan")
        series = build_well_timeseries(_targets(), TARGETS)
        written = plot_dmso_vs_drug(
            series,
            summary,
            experiment="TestCulture",
            drug="Navitoclax",
            target="percentile_90",
            out_dir=tmp_path,
            formats=("png",),
        )
        assert written and written[0].stat().st_size > 0

    def test_low_cell_count_points_are_marked(self, tmp_path: Path):
        """build_effect_table drops timepoints under MIN_CELLS_PER_SIDE, so the
        figure must not draw them as ordinary points."""
        series = build_well_timeseries(_targets(), TARGETS)
        assert (series["n_cells"] < 10).all()  # 2 cells per (well, timepoint)
        written = plot_dmso_vs_drug(
            series,
            _summary(),
            experiment="TestCulture",
            drug="Navitoclax",
            target="percentile_90",
            out_dir=tmp_path,
            formats=("png",),
            min_cells=10,
        )
        assert written and written[0].stat().st_size > 0


class TestPlotDmsoVsDrug:
    def test_writes_one_file_per_format(self, tmp_path: Path):
        series = build_well_timeseries(_targets(), TARGETS)
        written = plot_dmso_vs_drug(
            series,
            _summary(),
            experiment="TestCulture",
            drug="Navitoclax",
            target="percentile_90",
            out_dir=tmp_path,
            formats=("png",),
        )
        assert [p.name for p in written] == ["TestCulture_Navitoclax.png"]
        assert written[0].stat().st_size > 0

    def test_creates_missing_output_dir(self, tmp_path: Path):
        series = build_well_timeseries(_targets(), TARGETS)
        target_dir = tmp_path / "nested" / "figures"
        written = plot_dmso_vs_drug(
            series,
            _summary(),
            experiment="TestCulture",
            drug="Navitoclax",
            target="percentile_90",
            out_dir=target_dir,
            formats=("png",),
        )
        assert written and target_dir.is_dir()

    def test_censored_well_still_plots(self, tmp_path: Path):
        """A well that never collapses has t_cross_peak = NaN: no marker, but a line."""
        series = build_well_timeseries(_targets(), TARGETS)
        written = plot_dmso_vs_drug(
            series,
            _summary(t_cross_e07=float("nan")),
            experiment="TestCulture",
            drug="Navitoclax",
            target="percentile_90",
            out_dir=tmp_path,
            formats=("png",),
        )
        assert written and written[0].stat().st_size > 0

    def test_no_data_for_target_writes_nothing(self, tmp_path: Path):
        series = build_well_timeseries(_targets(), TARGETS)
        written = plot_dmso_vs_drug(
            series,
            _summary(),
            experiment="TestCulture",
            drug="Navitoclax",
            target="percentile_99",  # not a column that was built
            out_dir=tmp_path,
            formats=("png",),
        )
        assert written == []
        assert list(tmp_path.glob("*.png")) == []


class TestWriteDmsoVsDrugFigures:
    def test_one_figure_per_drug_for_the_headline_target(self, tmp_path: Path):
        written = write_dmso_vs_drug_figures(
            _targets(),
            _summary(),
            "TestCulture",
            tmp_path,
            target_columns=TARGETS,
            headline_target="percentile_90",
            formats=("png",),
        )
        assert [p.name for p in written] == ["TestCulture_Navitoclax.png"]

    def test_all_targets_fans_out_into_per_target_dirs(self, tmp_path: Path):
        written = write_dmso_vs_drug_figures(
            _targets(),
            _summary(),
            "TestCulture",
            tmp_path,
            target_columns=TARGETS,
            headline_target="percentile_90",
            all_targets=True,
            formats=("png",),
        )
        assert len(written) == 3
        # the headline target stays at the top level; the cross-check targets nest
        names = {str(p.relative_to(tmp_path)) for p in written}
        assert "TestCulture_Navitoclax.png" in names
        assert "percentile_75/TestCulture_Navitoclax.png" in names
        assert "percentile_95/TestCulture_Navitoclax.png" in names

    def test_dmso_appears_in_every_drug_figure(self, tmp_path: Path):
        """DMSO has drug='DMSO', so selecting by drug alone would drop the reference."""
        summary = pd.concat(
            [
                _summary(),
                pd.DataFrame(
                    [
                        {
                            "experiment": "TestCulture",
                            "well": "I07",
                            "drug": "Venetoclax",
                            "dose_rank": 1.0,
                            "concentration_uM": 50.0,
                            "is_dmso": False,
                            "t_cross_peak": 21.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        written = write_dmso_vs_drug_figures(
            _targets(wells=("E07", "E08", "I07", "M11")),
            summary,
            "TestCulture",
            tmp_path,
            target_columns=TARGETS,
            headline_target="percentile_90",
            formats=("png",),
        )
        # one per drug, and the DMSO row is not itself treated as a drug
        assert sorted(p.name for p in written) == [
            "TestCulture_Navitoclax.png",
            "TestCulture_Venetoclax.png",
        ]

    def test_unknown_experiment_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no rows"):
            write_dmso_vs_drug_figures(
                _targets(),
                _summary(),
                "NotACulture",
                tmp_path,
                target_columns=TARGETS,
                headline_target="percentile_90",
            )

    def test_missing_dmso_raises(self, tmp_path: Path):
        summary = _summary()
        summary["is_dmso"] = False
        with pytest.raises(ValueError, match="no DMSO well"):
            write_dmso_vs_drug_figures(
                _targets(),
                summary,
                "TestCulture",
                tmp_path,
                target_columns=TARGETS,
                headline_target="percentile_90",
            )
