# scPortrait BF-only single-cell pipeline

Segments cells in brightfield microscopy images, extracts single-cell crops, and featurizes each cell using a deep encoder.

## Project structure

```
my_project/
├── config.yml         # scPortrait configuration
├── run_pipeline.py    # main pipeline script
└── README.md
```

## Requirements

```
pip install scportrait
```

Cellpose's `cyto3` model is downloaded automatically on first run.

## Usage

```bash
python run_pipeline.py          # run on all *_BF.tif files in INPUT_DIR
python run_pipeline.py --debug  # verbose output and intermediate saves
```

## Configuration

Edit the two path constants at the top of `run_pipeline.py` to point at your data:

| Constant | Description |
|---|---|
| `INPUT_DIR` | Directory containing `*_BF.tif` images |
| `PROJECT_DIR` | Root output directory (created automatically) |

The pipeline skips `*_Cells.tif` and `*_w2.tif` files; only brightfield images
are used.

## Output

For each input image `<stem>_BF.tif`, the pipeline writes:

```
tmp/scportrait_bf_pipeline/<stem>/
├── plots/
│   ├── <stem>_segmentation.png   # BF image + cytosol mask overlay
│   ├── <stem>_extraction.png     # grid of 128-px single-cell crops
│   └── <stem>_featurization.png  # encoder activation distribution
└── ...                           # scPortrait project files (HDF5, masks)
```

## Pipeline steps

1. **Segmentation** – `CytosolOnlySegmentationCellpose` runs Cellpose `cyto3`
   on the full-resolution BF image to detect cytosol boundaries.
   `filter_masks_size: false` retains small cells.

2. **Extraction** – `HDF5CellExtraction` crops each detected cell to 128 × 128 px
   and stores the crops in a compressed HDF5 archive.

3. **Featurization** – `MLClusterClassifier` passes each crop through a deep
   encoder (`channel_selection: 0` selects the brightfield channel). The
   resulting latent embedding per cell is stored in the project's SpatialData
   object and can be retrieved via `project.sdata.tables`.

## Retrieving features programmatically

```python
from scportrait.pipeline.project import Project

project = Project("tmp/scportrait_bf_pipeline/<sample>", ...)
table_key = [k for k in project.sdata.tables if "MLClusterClassifier" in k][0]
features_df = project.sdata.tables[table_key].to_df()
```

## Note

The file is outdated, and should be updated 