# Dataset

This document covers the dataset used with this pipeline: its experimental design, file naming convention, plate layout, and directory structure.

---

## Overview

72-hour drug-response time-lapse screen of Ewing sarcoma cell lines against a panel of targeted therapies, acquired on a 384-well plate (pMF5V1) with three fluorescence channels.

| Property | Value |
|---|---|
| Plate ID | pMF5V1 |
| Plate format | 384-well |
| Duration | 0–72 h |
| Channels | BF, mCherry, FlipGFP (see channel mapping below) |
| Z-slices per well per timepoint | 20 core (z1–z20) + 1 projection (z0) |

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

The assignment of channels to wavelength slots (`w1`, `w2`, `w3`) differs between experiment groups. mCherry is always `w2`; BF and FlipGFP swap:

| Experiment group | w1 | w2 | w3 |
|---|---|---|---|
| HD1509, HD1883, SA110 | BF | mCherry | FlipGFP |
| Ew2-1 PMU421 | FlipGFP | mCherry | BF |

In code, these are the named constants `WAVELENGTH_MAPPINGS_HD_SA` and `WAVELENGTH_MAPPINGS_EW2` in `src/utils/file_utils.py`. All pipeline entry points accept an `--experiment-name` flag that resolves the correct mapping automatically.

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

## Data Format

### Raw data

- **Format:** 2D TIFF, one file per z-slice per timepoint per well per channel
- **Location:** `data/MF5V1 Timelapse samples 19.03.2024/`
- **Remote source:** https://hub.dkfz.de/s/HZLeBtBcwsezKKB

### Processed data

- **Location:** `data/MF5V1_processed Timelapse samples 19.03.2024/`
- **Formats:** TIFF (2D split, 3D stacks), Zarr (segmentation masks)
- **Remote sync target:** https://syncandshare.desy.de/index.php/s/get4QQrB7rHZFwq

#### Subfolder structure

TBD

---

## Analysis Channels

| Channel | Label | Purpose | Status |
|---|---|---|---|
| w1 or w3 (experiment-dependent) | BF | Cell segmentation (Cellpose-SAM) | In progress (3 of 5 experiments) |
| w2 | mCherry | Cell activity labelling (Otsu / Percentile / Manual threshold) | Partial — HD1509 and HD1883 done, needs rerun once full timepoints available |
| w1 or w3 (experiment-dependent) | FlipGFP | GFP-based cell death reporter | Out of scope for current voucher |
