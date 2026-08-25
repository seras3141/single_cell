"""Re-derive every number quoted in the dataset-design deliverable from its source data.

The deliverable (``docs/feature_to_mcherry/dataset_design_report/
dataset_design_assessment.md``) and its entry in the month's summary both quote figures
that come from four generated files. Prose drifts from data silently -- during this
analysis a regenerated table left stale figures in a collaborator-facing README, and a
"dose-ordered" claim turned out to contradict the very table it cited. This script is
the guard: it recomputes each quoted figure and fails loudly on any mismatch.

It deliberately checks **negative** claims too -- that Venetoclax is *not* dose-ordered,
that no Ew2-2 Venetoclax well reaches *moderate* -- because those are the ones a
spot-check by eye tends to miss.

Exits non-zero on any mismatch, so it can gate an sbatch run or a CI step.

Usage::

    python scripts/verify_dataset_design_numbers.py
    python scripts/verify_dataset_design_numbers.py --repo-root /path/to/checkout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
from scipy.stats import spearmanr

#: This file lives in ``<repo>/scripts/``, so the repo root is two levels up.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]

TARGET = "percentile_90"
LARGE = "large shift observed"
MODERATE = "moderate"
NO_LARGE = "no large shift observed in this imaged well"
NOT_ESTIMABLE = "not estimable (window too narrow)"

#: Romano et al. cut-points actually implemented by ``classify_shift``. 0.147 is
#: deliberately absent -- see dataset_design.py.
BANDS = [0.33, 0.474]


class Checker:
    """Collects pass/fail results so every check runs before the script exits."""

    def __init__(self) -> None:
        self.failures: List[str] = []

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        print(f"  {'OK ' if ok else 'BAD'} {label}{f'  {detail}' if detail else ''}")
        if not ok:
            self.failures.append(label)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    args = ap.parse_args()
    root: Path = args.repo_root

    report_dir = root / "docs/feature_to_mcherry/dataset_design_report"
    figures_dir = root / "docs/feature_to_mcherry/figures/feature_vs_time_pre_collapse"
    summary_csv = (
        root / "results/dataset_analysis/all_experiments_cell_population_summary.csv"
    )

    missing = [
        p
        for p in (
            report_dir / "drug_effect_two_windows.csv",
            report_dir / "density_vs_collapse.csv",
            report_dir / "cutoff_table.csv",
            summary_csv,
        )
        if not p.is_file()
    ]
    if missing:
        print("cannot verify -- missing inputs:")
        for p in missing:
            print(f"  {p}")
        return 2

    summ = pd.read_csv(summary_csv)
    eff = pd.read_csv(report_dir / "drug_effect_two_windows.csv")
    dens = pd.read_csv(report_dir / "density_vs_collapse.csv")
    cut = pd.read_csv(report_dir / "cutoff_table.csv")
    p90 = eff[eff.target == TARGET]
    c = Checker()

    print("=== 7a: cutoff table matches the Step 1 summary ===")
    for _, r in cut.iterrows():
        dmso = summ[(summ.experiment == r.experiment) & summ.is_dmso]
        c.check(
            f"{r.experiment} t_cross={r.t_cross_peak:.0f}",
            not dmso.empty
            and float(dmso.iloc[0]["t_cross_peak"]) == float(r.t_cross_peak),
        )

    print("\n=== 7b: density vs collapse ===")
    pooled = dens[dens.scope.str.startswith("pooled")].iloc[0]
    c.check("pooled rho +0.489", f"{pooled.rho:.3f}" == "0.489", f"({pooled.rho:.4f})")
    c.check("pooled p ~0.001", round(float(pooled.pvalue), 3) == 0.001)
    per = dens[~dens.scope.str.startswith("pooled")]
    c.check("no single culture significant", bool((per.pvalue > 0.05).all()))
    c.check("two cultures have the opposite sign", int((per.rho < 0).sum()) == 2)

    print("\n=== 7c': pre-confluence -- the headline ===")
    pre = p90[p90.window == "pre_confluence"]
    vc = pre.label.value_counts()
    c.check("no large shift anywhere", int(vc.get(LARGE, 0)) == 0)
    c.check("exactly 1 moderate", int(vc.get(MODERATE, 0)) == 1)
    c.check("15 no-large", int(vc.get(NO_LARGE, 0)) == 15)
    c.check("24 not estimable", int(vc.get(NOT_ESTIMABLE, 0)) == 24)
    mod = pre[pre.label == MODERATE].iloc[0]
    c.check(
        "the one moderate is HD1883 E07 Navitoclax 75uM",
        (mod.experiment, mod.well, mod.drug, mod.concentration_uM)
        == ("HD1883", "E07", "Navitoclax", 75.0),
    )
    c.check(
        "its delta is -0.469, i.e. 0.005 short of large",
        f"{mod.cliffs_delta:.3f}" == "-0.469"
        and abs(0.474 - abs(mod.cliffs_delta) - 0.005) < 0.001,
        f"(margin {0.474 - abs(mod.cliffs_delta):.4f})",
    )

    print("\n=== 7c': full timecourse ===")
    full = p90[p90.window == "full_timecourse"]
    big = full[full.label == LARGE]
    c.check("8 large shifts", len(big) == 8, f"({len(big)})")
    c.check("every one is a suppression", bool((big.direction == "suppresses").all()))

    print("\n=== the per-well window bound was applied ===")
    c.check("window_max_ti column present", "window_max_ti" in eff.columns)
    earlier = 0
    for exp in summ.experiment.unique():
        w = summ[summ.experiment == exp]
        dmso_rows = w[w.is_dmso]
        if dmso_rows.empty:
            continue
        dmso_cross = dmso_rows.iloc[0]["t_cross_peak"]
        drug = w[~w.is_dmso]
        earlier += int(
            (drug["t_cross_peak"].notna() & (drug["t_cross_peak"] < dmso_cross)).sum()
        )
    c.check(
        "15 of 40 drug wells collapse before their reference",
        earlier == 15,
        f"({earlier})",
    )

    print("\n=== dose-ordering claims, positive AND negative ===")
    nav = full[(full.experiment == "HD1509") & (full.drug == "Navitoclax")]
    rho, _ = spearmanr(nav.concentration_uM, nav.cliffs_delta)
    c.check(
        "HD1509 Navitoclax rank-ordered (rho -1.00)",
        abs(rho + 1.0) < 1e-9,
        f"({rho:+.3f})",
    )
    ven = full[(full.experiment == "HD1509") & (full.drug == "Venetoclax")]
    ven_by_dose = ven.set_index("concentration_uM")["cliffs_delta"]
    c.check(
        "HD1509 Venetoclax is NOT monotonic (5uM exceeds 50uM)",
        abs(ven_by_dose[5.0]) > abs(ven_by_dose[50.0]),
        f"(|{ven_by_dose[5.0]:.3f}| > |{ven_by_dose[50.0]:.3f}|)",
    )
    ven2 = full[(full.experiment == "Ew2-2") & (full.drug == "Venetoclax")]
    c.check(
        "no Ew2-2 Venetoclax well reaches moderate",
        not ven2.label.isin([MODERATE, LARGE]).any(),
    )

    print("\n=== boundary fragility: the smallest margin to a REAL band edge ===")
    have = p90[p90.cliffs_delta.notna()]
    margins = have.cliffs_delta.abs().apply(lambda v: min(abs(v - b) for b in BANDS))
    c.check(
        "smallest margin ~2.5e-3",
        abs(margins.min() - 0.00249) < 1e-4,
        f"({margins.min():.3e})",
    )

    print("\n=== Step 5': within-window trend is per-cell and flat ===")
    trend_files = sorted(figures_dir.glob("*/within_window_trend.csv"))
    if trend_files:
        trend = pd.concat([pd.read_csv(p) for p in trend_files])
        c.check(
            "rho_within spans 0.059-0.095",
            f"{trend.rho_within.min():.3f}" == "0.059"
            and f"{trend.rho_within.max():.3f}" == "0.095",
            f"({trend.rho_within.min():.4f}..{trend.rho_within.max():.4f})",
        )
    else:
        c.check("within_window_trend.csv found", False, f"(none under {figures_dir})")

    print()
    if c.failures:
        print(f"FAILED {len(c.failures)} check(s): {c.failures}")
        return 1
    print("ALL DATASET-DESIGN NUMBERS VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
