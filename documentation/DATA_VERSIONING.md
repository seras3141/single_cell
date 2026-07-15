# Data versioning (DVC)

How pipeline data is version-tracked, how to track a new experiment, and how to extend the
setup when the pipeline gains a model or a step. This is the canonical, committed reference;
per-phase design/dev notes live under `docs/data_versioning/` (not git-tracked).

---

## What it does

The data lives on Lustre (reached from the repo via the `data/` symlink), so **large data is
never committed to git** — only small hashes/pointers are. DVC provides reproducibility and
change-detection on top of the existing pipeline; it does **not** run the pipeline.

| Data | Mechanism | Bytes stored? |
|---|---|---|
| **Raw** TIFF trees | per-file `md5` manifest `data_versioning/raw/<exp>.md5` + folder hash in `raw_manifest_index.json` | no — hash only |
| **Large processed** (`split_data`, `3d_data`, `blur_heatmaps`, `inference`, `inference_tracked`, `inference_scportrait`) | one folder-state hash per subfolder (+ per model) in `data_versioning/processed_hashes/processed_hash_index.json` | no — hash only |
| **Small stats** (`processed_summary/`, `mcherry_metrics/`, `manifest.json`, CSVs) | DVC-stored to the local `store` remote (same Lustre mount, hardlinked) | yes — recoverable |
| **Per-run provenance** | `manifest.json` per experiment: per-stage status + DVC hash/path + `config_git_commit` (written automatically by the `run_*` scripts) | n/a |

Folder hashes are Merkle-style (md5 of the recursive per-file md5 listing), so a folder's hash
changes on any nested file change or an added/removed file/subfolder (e.g. a new model). This
gives version **tracking** and change-detection; large outputs are not byte-recoverable but are
regenerable via `dvc repro` from raw + pinned `config.yaml` + code. Small stats **are**
byte-recoverable from the `store`.

There is also a `dvc.yaml` reproducibility DAG (`prepare → segment-2d → track → extract`,
`cache: false` outputs) that coexists with `scripts/run_pipeline.py`.

---

## Everyday workflow: run, then track

**DVC does not run the pipeline — you do, however you like (step-wise, CPU/GPU, any order).**
Then one job records the version.

1. **Run the pipeline** (your usual step jobs). For a new/updated experiment, give `prepare`
   the renaming inputs: `--experiment-name <X>` (wavelength mapping) and `--plate MF5V1`.
   Each `run_*` step updates that experiment's `manifest.json` automatically.
2. **Track it:**
   ```bash
   sbatch slurm/track_experiment.sbatch "SA110 MF5V1 0-72h 13-02-26"
   ```
   Refreshes the raw manifest, the Tier-2 folder hashes (merging only that experiment), and
   byte-versions the small stats to the local `store`; then prints the exact `git` commands.
3. **Commit** the small text (hashes + `.dvc` pointers):
   ```bash
   git add "data_versioning/raw/SA110 MF5V1 0-72h 13-02-26.md5" \
           data_versioning/raw/raw_manifest_index.json \
           data_versioning/processed_hashes/processed_hash_index.json \
           "data_versioning/processed/SA110 MF5V1 0-72h 13-02-26" .gitignore
   git commit -m "Track SA110 MF5V1 0-72h 13-02-26"
   git push
   ```

No manual hashing; no large-file sync (only the small stats copy to a local store).
Partial runs are fine — the tracker hashes whatever subfolders exist.

### Checking / recovering
- **Raw integrity:** `cd <resolved raw dir> && md5sum -c "data_versioning/raw/<exp>.md5"`.
- **Processed drift:** re-run the tracker and `git diff data_versioning/processed_hashes/processed_hash_index.json` — a changed `folder_md5` (e.g. under `inference/<model>`) shows what changed; a new model appears as a new key.
- **Stats recovery:** `VIRTUAL_ENV="" uv run dvc pull -r store <unit>.dvc`.
- **Large outputs:** rebuild from the `config_git_commit` recorded in `manifest.json` (re-run the pipeline / `dvc repro`); the folder hash confirms the rebuild matches.

---

## Maintenance & extending the pipeline

Three lists must stay in sync with the real pipeline; keep them aligned and the routine loop
(run → track → commit) never changes:

| Concern | Source of truth |
|---|---|
| Provenance (stages tracked per run) | `STAGE_ORDER` in `src/dataset_analysis/run_manifest.py` |
| Tracking (folders hashed) | `DEFAULT_SUBFOLDERS` (+ `MODEL_LEVEL`) in `scripts/hash_processed_tier2.py` |
| Reproducibility (stages + params) | `dvc.yaml`, sourced from `config/config.yaml` |

### A new experiment — no code changes
Run the pipeline, then `sbatch slurm/track_experiment.sbatch "<experiment name>"` and commit
(see the everyday workflow). The wrapper handles a new name via `--experiments "<name>"
--merge`, leaving the other experiments' entries untouched. Optionally add it to
`DEFAULT_EXPERIMENTS` in `hash_processed_tier2.py` (and the `EXPS` list in the bulk sbatch) so
a full re-hash includes it by default.

### A new model (e.g. beyond `cellpose_sam`) — mostly automatic
- **Tracking captures it for free:** `inference/`/`inference_tracked/` are in `MODEL_LEVEL`, so
  the next tracking run records the new model's own per-model hash.
- **To run it through the pipeline/DAG:** set `segmentation.cellpose.model_type` in
  `config/config.yaml` (DAG paths derive from it). A different *framework* (not cellpose) needs
  `run_inference` code support (pipeline code, not DVC).
- **New top-level output dir** (e.g. `inference_<framework>/`)? Add its name to
  `DEFAULT_SUBFOLDERS` **and** `MODEL_LEVEL` (`inference_scportrait` is the precedent).
- To compare several models at once in the DAG, add a `foreach` over models.

### A new pipeline step `foo` — ~5-file checklist
1. `src/dataset_analysis/run_manifest.py` — add `"foo"` to `STAGE_ORDER`.
2. `scripts/run_pipeline.py` — add the step logic + `complete_stage_with_dvc(manifest, "foo", output_dir=…)`; add `"foo"` to the `--steps` argparse `choices`.
3. `config/config.yaml` — add a `foo:` params section (+ any new output-folder key).
4. `dvc.yaml` — add a `foo:` stage: `cmd` wrapping the `run_foo` entry point; `deps` (upstream stage outs + the script + its `src/` package); `params` (the `foo` config section); `outs` with `cache: false` (not nested inside another stage's out).
5. `scripts/hash_processed_tier2.py` — add `foo`'s output subfolder to `DEFAULT_SUBFOLDERS` (+ `MODEL_LEVEL` if per-model).
6. Extend `tests/dataset_analysis/test_processed_hash.py` / `tests/utils/test_run_manifest.py` if the schema/subfolders changed; re-run the tracker; commit.

---

## Reference

| Path | Role |
|---|---|
| `dvc.yaml` | reproducibility DAG (`prepare→segment-2d→track→extract`, `cache: false`) |
| `.dvc/config` | DVC config; default remote `store` = local dir on Lustre (no personal/WebDAV remote — put any per-user remote in the gitignored `.dvc/config.local`) |
| `scripts/hash_processed_tier2.py` | Tier-2 hash-only folder hashing (per-subfolder + per-model) |
| `slurm/track_experiment.sbatch` | per-experiment tracking wrapper (raw manifest + Tier-2 hashes + stats store) |
| `src/dataset_analysis/run_manifest.py` | `manifest.json` schema + DVC/git provenance helpers |
| `data_versioning/` | committed hashes/pointers (raw `.md5`, `processed_hashes/…json`, stats `.dvc`) |
| `docs/data_versioning/` | per-phase design/dev notes (not git-tracked) |
