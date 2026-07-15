"""Browse BF/mCherry/mask triples across wells and timepoints for a plate."""

from pathlib import Path
from typing import Dict, Tuple, Union

from src.inference.output_manager import OutputManager
from src.utils.file_utils import ConfigurableFileHandler, get_groups_from_filenames
from src.visualize.paths import resolve_related_paths


def build_index(
    raw_data_dir: Union[str, Path],
    file_handler: ConfigurableFileHandler,
    bf_pattern: str = "*_BF.tif",
) -> dict:
    """Index BF files under raw_data_dir by well and timepoint.

    Returns {"wells": [...], "timepoints": [...], "by_well_time": {(well, timepoint):
    bf_path}, "groups": {unique_id: [bf_path, ...]}}.
    """
    bf_paths = sorted(Path(raw_data_dir).glob(bf_pattern))

    by_well_time: Dict[Tuple[str, str], Path] = {}
    wells = set()
    timepoints = set()
    for bf_path in bf_paths:
        well = file_handler.extract_sample_id(bf_path.name)
        timepoint = file_handler.extract_time_point(bf_path.name)
        if well is None or timepoint == "unknown":
            continue
        by_well_time[(well, timepoint)] = bf_path
        wells.add(well)
        timepoints.add(timepoint)

    file_map = {str(p): p.name for p in bf_paths}
    groups = get_groups_from_filenames(file_map, file_handler)

    return {
        "wells": sorted(wells),
        "timepoints": sorted(timepoints, key=lambda t: int(t) if t.isdigit() else t),
        "by_well_time": by_well_time,
        "groups": groups,
    }


def resolve_selection(
    index: dict, output_manager: OutputManager, well: str, timepoint: str
) -> dict:
    """Resolve BF/mCherry/mask paths for a given (well, timepoint) selection.

    Returns {"bf": None, "mcherry": None, "mask": None} if there's no BF file indexed
    for that combination, rather than raising.
    """
    bf_path = index["by_well_time"].get((well, timepoint))
    if bf_path is None:
        return {"bf": None, "mcherry": None, "mask": None}
    return resolve_related_paths(bf_path, output_manager)


def build_browser_widget(
    index: dict,
    output_manager: OutputManager,
    mask_alpha: float = 0.5,
    mcherry_alpha: float = 0.5,
    mcherry_cmap: str = "inferno",
):
    """Dropdowns for well/timepoint that re-render the layered view on change."""
    import tifffile
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    from src.visualize.headless_layers import build_layer_controls, render_layers

    well_dropdown = widgets.Dropdown(options=index["wells"], description="Well")
    timepoint_dropdown = widgets.Dropdown(
        options=index["timepoints"], description="Timepoint"
    )
    output = widgets.Output()

    def _redraw(*_change):
        paths = resolve_selection(
            index, output_manager, well_dropdown.value, timepoint_dropdown.value
        )
        bf = tifffile.imread(paths["bf"]) if paths["bf"] is not None else None
        mcherry = (
            tifffile.imread(paths["mcherry"]) if paths["mcherry"] is not None else None
        )
        mask = (
            OutputManager.load_masks(paths["mask"])
            if paths["mask"] is not None
            else None
        )

        with output:
            clear_output(wait=True)
            if bf is None:
                print(
                    f"No BF file for well={well_dropdown.value}, "
                    f"timepoint={timepoint_dropdown.value}"
                )
                return
            fig, layer_artists = render_layers(
                bf,
                mask=mask,
                mcherry=mcherry,
                mask_alpha=mask_alpha,
                mcherry_alpha=mcherry_alpha,
                mcherry_cmap=mcherry_cmap,
            )
            display(fig)
            display(build_layer_controls(layer_artists, fig))

    well_dropdown.observe(_redraw, names="value")
    timepoint_dropdown.observe(_redraw, names="value")

    _redraw()

    return widgets.VBox([well_dropdown, timepoint_dropdown, output])
