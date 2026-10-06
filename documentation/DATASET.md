# Dataset

This document covers the dataset used with this pipeline: its experimental design, file naming convention, plate layout, and directory structure.

---

## Overview

72-hour drug-response time-lapse screen of **five Ewing sarcoma cell cultures** against a panel of targeted therapies, acquired on a 384-well plate (pMF5V1) with three fluorescence channels.

| Property | Value |
|---|---|
| Plate ID | pMF5V1 |
| Plate format | 384-well |
| Duration | 0–72 h |
| Channels | BF, mCherry, FlipGFP (see channel mapping below) |
| Z-slices per well per timepoint | 20 core (z1–z20) + 1 projection (z0) |
| Timepoint index interval | **10 min** — one index step = 10 min of wall clock |
| Brightfield cadence | **every 10 min** (every timepoint index) |
| Fluorescence cadence | **every 100 min** (every 10th index) — limited by phototoxicity |

### Acquisition cadence and why fluorescence is sparse

Brightfield is acquired at **every** timepoint (10-minute intervals). The two
fluorescence channels (mCherry, FlipGFP) are acquired only **every 100 minutes**
— every 10th timepoint index — because repeated fluorescence excitation is
phototoxic and would perturb the very drug response being measured.

**This is a deliberate property of the experiment, not a data-export defect.**
It follows that:

- Fluorescence timepoints always land on indices `t1, t11, t21, t31, …`, which is
  exactly what is observed in every experiment folder.
- **mCherry time resolution is capped at 100 min and cannot be improved by
  re-exporting.** Any analysis needing finer mCherry sampling has to be redesigned
  around the 100-minute grid, not blocked waiting for denser data.
- Only brightfield can be densified, and only where it was subsampled on export.

---

## File Naming Convention

Raw TIFF files follow this pattern:

```
t{timepoint}_{wellID}_s1_w{wavelength}_z{zslice}.tif
```

| Token | Meaning | Example values |
|---|---|---|
| `t{n}` | Timepoint index | `t1`, `t11`, `t21` |
| `{wellID}` | Plate well (row + column) | `C09`, `D07` |
| `s1` | Site — always 1 | `s1` |
| `w{n}` | Wavelength / channel | see channel mapping below |
| `z{n}` | Z-slice | `z0` = projection, `z1`–`z20` = core slices |

**Example:** `t21_C09_s1_w1_z05.tif` → timepoint 21, well C09, z-slice 5.

Each experiment folder also contains a `{name}_Projection/` subfolder with maximum-intensity projections (`z0` files).

### Channel mapping

The assignment of channels to wavelength slots (`w1`, `w2`, `w3`) differs between experiments. mCherry is always `w2`; BF and FlipGFP swap:

| Experiment | w1 | w2 | w3 |
|---|---|---|---|
| HD1509, SA110 | BF | mCherry | FlipGFP |
| Ew2-1 PMU421, Ew2-2, **HD1883** | FlipGFP | mCherry | BF |

In code, these are the named constants `WAVELENGTH_MAPPINGS_HD_SA` and `WAVELENGTH_MAPPINGS_EW2` in `src/utils/file_utils.py`, resolved per experiment through `EXPERIMENT_WAVELENGTH_MAPPINGS`. All pipeline entry points accept an `--experiment-name` flag that resolves the correct mapping automatically.

> **Corrected 2026-08-06:** HD1883 carries BF on `w3` (measured mean ≈3450–3500 vs ≈200–320 on `w1`/`w2`), so it groups with Ew2.
> The duplicate table in `src/cell_activity_labeler/utils/file_utils.py` has HD1509/HD1883 transposed — see [Known Data Issues](#duplicate-channel-mapping-table-in-cell_activity_labeler).

---

## Plate Layout

Full machine-readable layout: `config/MF5v1_plate_layout.json`.

### Quadrant structure

The 384-well plate is divided into **4 identical quadrants**, each spanning 6 columns. The drug panel is replicated across all quadrants.

| Quadrant | Columns |
|---|---|
| Q1 | 1–6 |
| Q2 | 7–12 |
| Q3 | 13–18 |
| Q4 | 19–24 |

Within each quadrant, **column offset 1 is empty**; offsets 2–6 carry concentrations 1–5 (high to low).

### Row assignments

| Row | Content | Drug / Control |
|---|---|---|
| A | Empty | — |
| B | Drug | Eprenetapopt (replicate 1) |
| C | Drug | Doxorubicin (replicate 1) |
| D | Drug | Doxorubicin (replicate 2) |
| E | Drug | Navitoclax (replicate 1) |
| F | Drug | Navitoclax (replicate 2) |
| G | Drug | Selinexor (replicate 1) |
| H | Drug | Selinexor (replicate 2) |
| I | Drug | Venetoclax (replicate 1) |
| J | Drug | Venetoclax (replicate 2) |
| K | Positive control | Staurosporine (replicate 1) |
| L | Positive control | Staurosporine (replicate 2) |
| M | Mixed control | BenzethoniumCl (offsets 2–3) / DMSO (offsets 4–6), replicate 1 |
| N | Mixed control | BenzethoniumCl (offsets 2–3) / DMSO (offsets 4–6), replicate 2 |
| O | Drug | Eprenetapopt (replicate 2) |
| P | Empty | — |

> **Replication:** 2 rows per drug × 4 quadrants = **8 wells per concentration per plate**.

### Drug concentrations

| Drug | Class | Target | Conc 1 (µM) | Conc 2 (µM) | Conc 3 (µM) | Conc 4 (µM) | Conc 5 (µM) |
|---|---|---|---|---|---|---|---|
| Doxorubicin | Chemotherapy | Topoisomerase II | 1.0 | 0.1 | 0.01 | 0.001 | 0.0001 |
| Eprenetapopt (APR-246) | Apoptosis | p53 activator | 100.0 | 10.0 | 1.0 | 0.1 | 0.001 |
| Navitoclax | Apoptosis | Bcl-2, Bcl-XL | 75.0 | 10.0 | 1.0 | 0.1 | 0.01 |
| Selinexor | Apoptosis | XPO1 | 100.0 | 10.0 | 1.0 | 0.1 | 0.001 |
| Venetoclax | Apoptosis | Bcl-2 | 50.0 | 10.0 | 5.0 | 1.0 | 0.01 |

Concentrations decrease from column offset 2 (conc 1, highest) to offset 6 (conc 5, lowest).

### Controls

| Control | Type | Rows | Notes |
|---|---|---|---|
| Staurosporine | Positive (cytotoxic) | K, L | Pan-kinase inhibitor; induces broad apoptosis |
| Benzethonium Chloride | Positive (cytotoxic) | M, N (offsets 2–3) | Cytotoxic detergent |
| DMSO | Negative (vehicle) | M, N (offsets 4–6) | Solvent used to dissolve drugs; baseline control |

---

## Multi-Culture Experiment Structure

Each of the five subfolders in the raw data corresponds to a **different Ewing sarcoma cell culture** (not a different experimental setting). All cultures were screened on the same pMF5V1 plate, but each culture was only imaged on **2 of the 5 drugs**, chosen based on that culture's known drug sensitivities.

- Example: culture 1 → drugs 1 & 2; culture 2 → drugs 1 & 4; etc.
- Wells that appear in more than one subfolder represent the **same drug well imaged with a different culture** — not duplicate experiments.

### Drug assignments per experiment

| Culture | Drug 1 | Drug 2 | Control |
|---|---|---|---|
| SA110 | Navitoclax | Selinexor | DMSO (`N11`) |
| HD1509 | Navitoclax | Venetoclax | DMSO (`N11`) |
| HD1883 | Navitoclax | Selinexor | DMSO (`N11`) |
| Ew2-1 PMU421 | Doxorubicin | Navitoclax | DMSO (`M11`) |
| Ew2-2 | **Venetoclax** | Navitoclax | DMSO (`M11`) |

> **Corrected 2026-08-17:** `Ew2-2` is Venetoclax + Navitoclax (rows I/J and E/F; no `C`/`D` wells, so no Doxorubicin), per `config/MF5v1_plate_layout.json`.
>
> **Cross-culture consequence:** Doxorubicin appears in `Ew2-1` only (n=1, confounded with that culture); Venetoclax in `HD1509` and `Ew2-2` (n=2); Selinexor in `HD1883` and `SA110`. Only Navitoclax (all 5) supports a drug-level cross-culture claim.

Each culture is imaged at **4 of the 5 concentration tiers** for each of its two drugs —
plate columns 7–10, highest dose to lowest — plus one DMSO well at column 11. There is
**exactly one well per (drug, concentration)**, so the design contains no within-condition
replicate wells.

### Negative control selection per experiment

Each experiment contains exactly one negative control well: either **M11** or **N11** (both are DMSO wells in quadrant Q2). The choice is made manually at imaging time based on cell quality at timepoint 0 — whichever well shows better cell distribution without artifacts or out-of-frame cells is selected. This means:

- Every experiment has **at least one DMSO control**.
- The control well identity (M11 vs N11) must be checked per-experiment before normalizing features.

> **Implication for normalization:** Feature normalization must use the per-experiment control well rather than a fixed well ID. The pipeline should look up the control well for each subfolder/experiment independently.

---

## Experiments

### Experiment 1 — SA110

| Property | Value |
|---|---|
| Folder | `SA110 MF5V1 0-72h 13-02-26` |
| Acquisition date | 2026-02-13 |
| Raw files | 4,971 |
| Raw size | 9.71 GB |
| Timepoints present | t1, t11, t21, …, t361 (37 timepoints) |
| Dataset summary generated | Yes |

**QC Issues**

| Issue type | Count |
|---|---|
| missing_z | 5,328 |
| missing_channel_z | 24 |
| **Total errors** | **5,352** |

**Processing status**

| Step | Status |
|---|---|
| Split (2D) | Done |
| 3D stacks | Done |
| Blur heatmaps | Done |
| Segmentation (Cellpose-SAM) | Not started |
| Tracking | Not started |
| mCherry activity labels | Not started |

---

### Experiment 2 — HD1509

| Property | Value |
|---|---|
| Folder | `HD1509 MF5V1 0-72h 23-02-26` |
| Acquisition date | 2026-02-23 |
| Raw files | 4,768 |
| Raw size | 9.31 GB |
| Timepoints present | t1, t11, t21, …, t361 (37 timepoints) |
| Dataset summary generated | Yes |

**QC Issues**

| Issue type | Count |
|---|---|
| missing_z | 5,400 |
| missing_channel_z | 11 |
| **Total errors** | **5,411** |

**Processing status**

| Step | Status |
|---|---|
| Split (2D) | Done |
| 3D stacks | Done |
| Blur heatmaps | Done |
| Segmentation (Cellpose-SAM) | Done |
| Tracking | Done |
| mCherry activity labels | Not started |

---

### Experiment 3 — HD1883

| Property | Value |
|---|---|
| Folder | `HD1883 MF5V1 0-72h 20-03-26` |
| Acquisition date | 2026-03-20 |
| Raw files | 4,321 |
| Raw size | 8.44 GB |
| Timepoints present | t1, t11, t21, …, t351 (36 timepoints) |
| Dataset summary generated | Yes |

**QC Issues**

| Issue type | Count |
|---|---|
| missing_z | 5,360 |
| missing_channel_z | 11 |
| **Total errors** | **5,371** |

**Processing status**

| Step | Status |
|---|---|
| Split (2D) | Done |
| 3D stacks | Done |
| Blur heatmaps | Done |
| Segmentation (Cellpose-SAM) | Done |
| Tracking | Not started |
| mCherry activity labels | Not started |

---

### Experiment 4 — Ew2-1 PMU421

| Property | Value |
|---|---|
| Folder | `Ew2-1 MF5V1 0-72h 06-03-26` |
| Acquisition date | 2026-03-06 |
| Raw files | 4,296 |
| Raw size | 8.39 GB |
| Timepoints present | t1, t11, t21, …, t351 (35 timepoints) |
| Dataset summary generated | Yes |

**QC Issues**

| Issue type | Count |
|---|---|
| missing_z | 5,180 |
| missing_channel_z | 9 |
| **Total errors** | **5,189** |

**Processing status**

| Step | Status |
|---|---|
| Split (2D) | Done |
| 3D stacks | Done |
| Blur heatmaps | Done |
| Segmentation (Cellpose-SAM) | Done |
| Tracking | Not started |
| mCherry activity labels | Not started |

---

### Experiment 5 — Ew2-2

| Property | Value |
|---|---|
| Folder | `Ew2-2 MF5V1 072h 17-04-26` |
| Acquisition date | 2026-04-17 |
| Raw files | TBD |
| Raw size | TBD |
| Timepoints present | TBD |
| Dataset summary generated | No |

**QC Issues:** Not yet assessed (dataset summary not yet generated).

**Processing status**

| Step | Status |
|---|---|
| Split (2D) | Done |
| 3D stacks | Done |
| Blur heatmaps | Done |
| Segmentation (Cellpose-SAM) | Not started |
| Tracking | Not started |
| mCherry activity labels | Not started |

---

## Known Data Issues

### Incomplete timepoints in some experiments (NextCloud sync issue)

**Status:** Under investigation (as of 2026-06-29).

**Observed:** Some experiment folders appear to have fewer than the expected 351 timepoints (e.g., t11–t151 instead of t1–t351).

**Expected:** All datasets should contain 351 timepoints, covering the full 72-hour timelapse range.

**Root cause (likely):** Synchronisation gap during NextCloud transfer. The researcher confirmed that the compressed images on the source PC are complete with 351 timepoints.

**Action items:**
1. Identify which experiment folders are missing timepoints (compare file counts against 351 × number of wells × channels × z-slices).
2. Re-sync the affected folders from NextCloud / source PC.
3. Re-run mCherry metrics for HD1509 and HD1883 once full timepoints are confirmed available (see Analysis Channels table).

### The "Timepoints 1-40" re-export covers only the first ~6.5 h

**Status:** Confirmed 2026-08-06. **This is the significant limitation of that share.**

The `MF5V1 Timepoints 1-40` share (see [Raw data](#raw-data)) was provided to
address the sparse-timepoint issue above. At 10 min per timepoint index it spans
**t1–t40 = 390 min ≈ 6.5 h of a 72-hour experiment**, so the folder names
(`… 0-72h T40`) are misleading — the content is the opening ~9 % of the
time-course, not the whole thing.

Coverage of the two raw trees, in wall-clock terms:

| Tree | Indices | BF cadence | IF cadence | Wall-clock covered |
|---|---|---|---|---|
| `MF5V1 Timelapse samples 19.03.2024` | t1, t11, …, t351 | 100 min | 100 min | **≈58 h** |
| `MF5V1 Timepoints 1-40` | t1–t40 | **10 min** | 100 min | **≈6.5 h** |

Consequences:

- **For mCherry the older tree is strictly better** — 36 fluorescence timepoints
  across ~58 h, versus **4** across 5 h in the new share. The new export adds
  nothing for mCherry and covers far less of the experiment.
- **What the new share uniquely provides is dense brightfield** (10-min cadence)
  for the first 6.5 h — valuable for early-response morphology, and *would* be
  valuable for cross-timepoint cell tracking if that were implemented. No such
  tracking exists in this pipeline today: `CellTracker3D`/trackpy only links cell
  instances across the z-slices of a single timepoint's stack, never across
  timepoints, so BF cadence has no effect on the tracking that actually runs.
- **The two trees are complementary, not superseding.** Neither alone gives dense
  BF across the full 72 h.
- Late drug response (24 / 48 / 72 h) is **absent** from the new share entirely.

The actionable request is therefore **dense brightfield for t41 onward**, which is
physically possible because BF was acquired every 10 min and merely subsampled on
export. Densifying the fluorescence channels is *not* possible — see
[Acquisition cadence](#acquisition-cadence-and-why-fluorescence-is-sparse).

Every experiment in the new share holds a uniform **9,072 files** (8,640 z-stacks
+ 432 projections): brightfield contributes 9 wells × 40 t × 20 z = 7,200, and the
two fluorescence channels 2 × 9 × 4 × 20 = 1,440.

> **Open question:** at 10 min/index, index 351 corresponds to 58.3 h, not the
> 72 h stated elsewhere in this document. Either acquisition ran ~58 h, the
> interval is slightly longer than 10 min, or indices continue past 351. The raw
> TIFFs carry no `DateTime` tag (minimal tag set only), so this could not be
> settled from the files — worth confirming with the acquisition log.

### Duplicate channel-mapping table in `cell_activity_labeler`

**Status:** Open, not fixed (2026-08-06).

`EXPERIMENT_WAVELENGTH_MAPPINGS` exists in **two** places:

| File | HD1509 | HD1883 | Correct? |
|---|---|---|---|
| `src/utils/file_utils.py` | BF = `w1` | BF = `w3` | ✅ matches measurement |
| `src/cell_activity_labeler/utils/file_utils.py` | BF = `w3` | BF = `w1` | ❌ transposed |

The sub-package copy has drifted and mislabels brightfield for those two
experiments. mCherry (`w2`) is identical in both conventions, so mCherry metrics
and activity labelling are unaffected — but anything resolving *brightfield* via
the sub-package's table would read the FlipGFP channel instead. Left unchanged
pending a check of what consumed it; deduplicating the table (sub-package
importing the canonical mapping) is the durable fix.

---

## Data Format

### Raw data

- **Format:** 2D TIFF, one file per z-slice per timepoint per well per channel
- **Uniform file size:** 2,097,274 bytes (1024×1024 16-bit + header)
- **Locations:**
  - `data/MF5V1 Timelapse samples 19.03.2024/` — original pull, `t1, t11, …, t351` (36 timepoints, every 10th frame)
  - `data/MF5V1 Timepoints 1-40/` — 2026-08 re-export, `t1`–`t40` dense for brightfield only (45,360 files, 88.6 GiB)
- **Remote sources:**
  - https://hub.dkfz.de/s/HZLeBtBcwsezKKB (original)
  - https://hub.dkfz.de/s/9JAfiWL7cZjNxcr (`MF5V1 Timepoints 1-40`)

> **The two trees share a common time axis and can be combined directly.**
> A given index means the same wall-clock moment in both (index × 10 min), and
> the overlapping files are **byte-identical** — verified for HD1509 across
> `t1/t11/t21/t31` × `w1–w3` × `z1/z10/z20` (36/36 identical). The trees differ
> only in *which* indices they contain, so their union is
> `t1`–`t40` dense plus `t41, t51, …, t351` every 100 min.
>
> Corrected: an earlier revision warned the indices were not comparable; they are on the same scale.

---

### Processed data

- **Location:** `data/MF5V1_processed Timelapse samples 19.03.2024/`
- **Formats:** TIFF (2D split, 3D stacks), Zarr (segmentation masks)
- **Remote sync target:** https://syncandshare.desy.de/index.php/s/get4QQrB7rHZFwq

#### Subfolder structure

TBD

### Sample data for feature extraction

`data/sample_data/` is a small, self-contained subset of brightfield images used
to run and validate the scPortrait ConvNeXt feature extractor (mask-free — it
runs its own segmentation). It is **generated**, not tracked; rebuild it with
`sbatch slurm/sample_data.sbatch` (idempotent).

The sample copies (via `cp -L`, dereferencing the source symlinks into real
~2 MB TIFs) a few wells from three experiments' `split_data/` folders, keeping
all timepoints and all z-slices:

| Short name | Source `split_data/` folder | Wells | BF files |
|---|---|---|---|
| `HD1883` | `HD1883 MF5V1 0-72h 20-03-26` | E07, F08 | 1512 |
| `Ew2-1` | `Ew2-1 MF5V1 0-72h 06-03-26` | C09, D07 | 1512 |
| `Ew2-2` | `Ew2-2 MF5V1 072h 17-04-26` | E07, E10 | 1512 |

Each well = 36 timepoints × 21 z-slices = 756 BF files; total ≈ 4,536 files
(~8.4 GB). Files keep their source names (`pMF5V1_<well>_t<n>_z<n>_BF.tif`), which
do **not** encode the experiment — hence the per-experiment subdirs, which
namespace both the inputs and the per-image output CSVs.

Run feature extraction over it with `sbatch slurm/gpu_feature_scportrait.sbatch`
(configured to run one `--output-dir` per experiment; ships with a z10-only
smoke pass and a commented full all-z pass). To change the selection, edit the
`EXP_WELLS` map in `slurm/sample_data.sbatch` and resubmit.

---

## Downloading the raw dataset

Use `data/download_data.py`, or the SLURM wrapper `slurm/download_data.sbatch`
for anything large (a full share is ~88 GiB and takes ~20–40 min per experiment).

### Why a plain download is not enough

`hub.dkfz.de` **disables public WebDAV**, so a share folder cannot be listed —
`PROPFIND` returns `401` for every auth form, and the OCS API needs a login.
Instead, `--discover` *probes* the naming grid with one-byte range requests: an
existing file answers `206` with `Content-Range: bytes 0-0/<size>` (existence and
size for almost no traffic), a missing one `404`.

Two properties of this dataset make the probing non-obvious, and both are handled:

1. **The timepoint axis is not independent of the channel axis.** One channel is
   dense and the others are sampled every 10th frame, so timepoints are probed
   per `(well, channel)` pair. Probing `t` once and crossing it with the channel
   list would fabricate ~13,600 non-existent names per experiment.
2. **Projection folder names drift from their parent.** The embedded date often
   differs (`SA110 … 13-02-26 T40` → `SA110 … 13-02-25_Projection`), `Ew2-2` uses
   `072h` in the parent but `0-72h` in the projection, and **Ew2-1's projection
   carries an extra token** (`Ew2-1 PMU421 MF5V1 0-72h 06-03-26_Projection`) that
   no date sweep can reach — it must be passed explicitly as its own link.

### Download the whole share

Share links live one per line in a text file; address-bar URLs (`?path=%2F…`) are
accepted as-is — there is no need to convert them to `/download?…` links.

```bash
sbatch --export=ALL,\
URL_FILE="$HOME/projects/single_cell/slurm/mf5v1_t1-40_shares.txt",\
OUTPUT_DIR="data/MF5V1 Timepoints 1-40",\
MANIFEST_DIR="$HOME/projects/single_cell/data",\
DISCOVER=1,NUM_WORKERS=8 \
  slurm/download_data.sbatch
```

Provided link lists:

| File | Contents |
|---|---|
| `slurm/mf5v1_t1-40_shares.txt` | all five experiment folders |
| `slurm/mf5v1_t1-40_shares_dense.txt` | HD1509 + SA110 only |
| `slurm/mf5v1_t1-40_shares_sparse3.txt` | Ew2-1, Ew2-2, HD1883 (+ Ew2-1's explicit projection link) |

Key options (`sbatch` variable → script flag):

| Variable | Flag | Effect |
|---|---|---|
| `URL_FILE` | `--url-file` | One share link per line (or use `URL` for a single link) |
| `DISCOVER=1` | `--discover --include-subfolders` | Probe the naming grid; also pull nested `*_Projection` folders |
| `MANIFEST_DIR` | `--manifest-dir` | Write `<folder>.filelist.txt` per folder — the record of what the share holds |
| `NUM_WORKERS` | `--workers` | Parallel download streams (8 is comfortable) |

Run it directly to inspect before transferring — `--dry-run` prints the
per-folder inventory and an exact byte total:

```bash
uv run python data/download_data.py \
    --url-file slurm/mf5v1_t1-40_shares.txt \
    --discover --include-subfolders --dry-run \
    --dest "data/MF5V1 Timepoints 1-40" --manifest-dir data
```

**Resumability:** files stream to `*.part` and are renamed only on completion,
and existing non-empty files are skipped. A cancelled or timed-out job can be
re-submitted unchanged and it continues where it stopped. Each folder is
mirrored into its own subdirectory (`--mirror`, the default when more than one
folder share is given), with projections nested under their parent.

---

## Generating the inventory list

`scripts/summarize_raw_dataset.py` reports, per experiment, what is on disk
against two baselines — and writes a `raw_summary/` folder beside each
experiment, mirroring the role of `processed_summary/` for pipeline outputs.

```bash
uv run python scripts/summarize_raw_dataset.py \
    --dataset-root "data/MF5V1 Timepoints 1-40" \
    --manifest-dir data
```

The two baselines answer different questions:

| Column | Baseline | Question |
|---|---|---|
| `dl gap` | the `*.filelist.txt` manifest | Did every file the share holds actually land on disk? |
| `wN miss` | full dense grid (`t1`–`t40` × wells × `z1`–`z20`) | How far is each wavelength from a complete acquisition? |

So a healthy, fully-downloaded experiment shows `dl gap = 0` with a non-zero
`miss` on the two sparse fluorescence channels — the sparsity is a property of
the acquisition, not a download failure. Example (HD1509):

```
folder                                     on disk  dl gap  w1 miss  w2 miss  w3 miss
HD1509 MF5V1 0-72h 23-02-26 T40               8640       0        0     6480     6480
HD1509 MF5V1 0-72h 23-02-25_Projection         432       0        0      324      324
```

Each `raw_summary/` contains:

| File | Contents |
|---|---|
| `raw_summary.json` | Per-folder `{expected, found, missing, missing_entries}` blocks — `download` vs manifest, `acquisition` per wavelength — in the same shape as `processed_summary.json` |
| `raw_inventory.csv` | One row per folder: files and bytes on disk, well count, `download_missing` (gap vs manifest; empty without a manifest), per-wavelength missing counts |
| `raw_missing.csv` | One row per missing frame, with a `reason` column |

Summaries are keyed on **wavelength index** (`w1`/`w2`/`w3`) rather than channel
name deliberately, so the report never depends on the channel-mapping table; the
console header prints the measured brightfield slot per experiment. Options:
`--expected-timepoints N` changes the dense-grid depth (default 40),
`--out-dirname` renames the output folder.

Reusable logic lives in `src/dataset_analysis/raw_share_summary.py`
(`build_raw_summary`, `write_raw_summary`, `scan_frames`).

## Analysis Channels

| Channel | Label | Purpose | Status |
|---|---|---|---|
| w1 (HD1509, SA110) or w3 (Ew2-1, Ew2-2, HD1883) | BF | Cell segmentation (Cellpose-SAM) | In progress (3 of 5 — SA110 pending). **Dense `t1`–`t40` now available for all five** in `data/MF5V1 Timepoints 1-40/` |
| w2 | mCherry | Cell activity labelling (Otsu / Percentile / Manual threshold) | Partial — HD1509 and HD1883 done. **Sampling is capped at 100 min per well** (acquisition limit, not an export gap), so the 2026-08 re-export adds no mCherry timepoints |
| w3 (HD1509, SA110) or w1 (Ew2-1, Ew2-2, HD1883) | FlipGFP | GFP-based cell death reporter | Out of scope for current voucher |
