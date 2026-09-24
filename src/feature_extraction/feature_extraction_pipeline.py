from typing import List, Tuple, Dict, Any, Optional
import logging
import os
import re
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from src.feature_extraction.feature_extractor_incarta import (
    extract_all_instance_features,
)

try:
    from src.feature_extraction.feature_extractor_scportrait import (
        get_scportrait_features,
    )
except ImportError:
    get_scportrait_features = None
from src.feature_extraction.feature_extractor_regionprops import get_region_properties
from src.utils.config_schemas import (
    FeatureExtractionConfig,
    check_feature_method_available,
)
from src.utils.data_exclusions import DataExclusions, load_data_exclusions
from src.utils.file_utils import ConfigurableFileHandler
from src.utils.image_utils import load_image, load_labels

# Default filename patterns, from the same schema defaults the CLI config uses
# (MF5V1 layout: ``<stem>_BF.tif`` images, ``<stem>_pred_mask.tif`` masks).
DEFAULT_IMAGE_PATTERN = FeatureExtractionConfig.image_pattern
DEFAULT_MASK_PATTERN = FeatureExtractionConfig.mask_pattern

# Exceptions that signal a code defect or missing dependency rather than a bad
# input file. The per-file handler in ``extract_features_from_path`` re-raises
# these instead of recording them: they would recur on every file, so fail on
# the first. TypeError/AttributeError are deliberately NOT here: numeric
# libraries raise them for bad data too (e.g. skimage on a float label image).
_PROGRAMMING_ERRORS = (NotImplementedError, NameError, ImportError)


class FeatureExtractionError(RuntimeError):
    """A run finished but one or more input files failed."""


def _pattern_has_channel(pattern: str, channel: str) -> bool:
    """True if an image glob selects ``channel``, e.g. ``"*_BF_3d.tif"`` and ``"BF"``."""
    return re.search(rf"_{re.escape(channel)}(?:[_.]|$)", pattern) is not None


# scPortrait label-mask export layout. The exported mask mirrors the Cellpose
# tracked-mask tree but lives under a sibling ``inference_scportrait/`` dir:
#   <sample>/inference_scportrait/scportrait/test/final_2d/<stem>_pred_mask.tif
SCPORTRAIT_MASK_ROOT_NAME = "inference_scportrait"
SCPORTRAIT_MASK_SUBDIRS = ("scportrait", "test", "final_2d")

# Milestone 2 (mask injection): exported LABEL MASKS from an injected run go
# under a SEPARATE sibling tree so they never collide with the native-scPortrait
# masks above. This covers the mask export only -- the per-image and combined
# feature CSVs follow ``output.output_dir`` like every other method, so an
# injected run must be pointed at its own output directory or it will overwrite
# a native run's CSVs (``slurm/gpu_feature_scportrait_injected.sbatch`` passes
# ``--output-dir <sample>/inference_scportrait_injected/features`` for this
# reason). ``_warn_on_existing_output`` flags the collision at run time.
SCPORTRAIT_INJECTED_ROOT_NAME = "inference_scportrait_injected"

# Child names whose presence marks a processed-experiment ("sample") folder,
# e.g. ``.../HD1509 MF5V1 0-72h 23-02-26/``. No single marker is present in
# every experiment (SA110 lacks manifest.json; HD1883 lacks inference_tracked),
# so any one match identifies the sample dir.
_SAMPLE_DIR_MARKERS = frozenset(
    {
        "split_data",
        "inference",
        "inference_tracked",
        "3d_data",
        "processed_summary",
        "manifest.json",
        "blur_heatmaps",
    }
)


def derive_sample_dir(image_path: Path) -> Optional[Path]:
    """Walk up from an input image to its processed-experiment ("sample") dir.

    Input BF images live at ``<sample>/split_data/<stem>_BF.tif``; the sample
    dir is the nearest ancestor that contains recognizable pipeline-output
    children (``inference_tracked``, ``split_data``, ``manifest.json``, ...).

    Resolves the containing *directory* rather than the image file: in the
    processed datasets the ``split_data`` BF images are symlinks into the raw
    dataset tree, so ``image_path.resolve()`` would jump out of the processed
    experiment folder (whose markers we need) and into the marker-less raw
    folder. Resolving ``image_path.parent`` keeps us on the processed side.

    Returns None if no such ancestor is found (e.g. ad-hoc input layouts such
    as the raw ``data_old/Plate 2426/BF Images/`` validation inputs).
    """
    start = Path(image_path).parent.resolve()
    for ancestor in (start, *start.parents):
        try:
            child_names = {child.name for child in ancestor.iterdir()}
        except (OSError, PermissionError):
            continue
        if child_names & _SAMPLE_DIR_MARKERS:
            return ancestor
    return None


def resolve_cellpose_mask(
    bf_path: Path,
    mask_root: Path,
    mask_pattern: str = "{stem}_pred_mask.tif",
) -> Path:
    """Resolve the cellpose_sam mask paired with a brightfield image (M2).

    The BF stem ``<...>_BF`` maps to mask ``<...>_pred_mask.tif`` — the same
    1:1-by-stem pairing ``mcherry_metrics`` uses. ``mask_pattern`` may use a
    ``{stem}`` placeholder, where ``stem`` is the BF stem with a trailing
    ``_BF`` removed. The direct ``mask_root/<filename>`` is checked first (the
    common case: a ``.../cellpose_sam/final_2d`` dir); otherwise ``mask_root``
    is searched recursively.

    **Precedence is explicit:** a file sitting directly in ``mask_root`` wins
    over a same-named file in a nested subdirectory. The direct hit is
    unambiguous by construction (one exact path), and taking it avoids an
    ``rglob`` over a ``final_2d`` tree of thousands of masks for every image.
    Only when there is no direct hit does the recursive search apply, and that
    path raises ``FileNotFoundError`` unless exactly one mask matches — it
    never guesses among nested candidates.
    """
    stem = Path(bf_path).stem
    if stem.endswith("_BF"):
        stem = stem[: -len("_BF")]
    filename = mask_pattern.format(stem=stem)

    direct = Path(mask_root) / filename
    if direct.exists():
        return direct

    matches = sorted(Path(mask_root).rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one cellpose mask '{filename}' for "
            f"'{Path(bf_path).name}' under {mask_root}, found {len(matches)}"
            + (f": {[str(m) for m in matches[:5]]}" if matches else "")
        )
    return matches[0]


def _extract_one_pair(
    pipeline: "FeatureExtractionPipeline",
    image_path: Path,
    mask_path: Path | None,
    save_individual: bool,
) -> Tuple[Optional[pd.DataFrame], List[Tuple[str, str]]]:
    """Module-level worker for parallel file processing.

    Must live at module scope (not a closure/bound method) so it is picklable by
    the joblib ``loky`` backend. Runs one image/mask through ``pipeline`` with
    inner per-cell parallelism disabled, optionally saves the per-image CSV, and
    returns ``(features_df, new_error_records)`` — the error records are returned
    (rather than left on ``pipeline.error_files``) because worker mutations of
    the pickled ``pipeline`` copy do not propagate back to the parent process.
    """
    err_before = len(pipeline.error_files)
    features_df = pipeline.extract_features_from_path(
        image_path, mask_path, inner_n_jobs=1
    )
    new_errors = list(pipeline.error_files[err_before:])
    if features_df is not None and save_individual:
        pipeline.save_image_features(features_df, Path(image_path))
    return features_df, new_errors


class FeatureExtractionPipeline:
    """Pipeline for extracting features from datasets of segmented cells."""

    def __init__(
        self,
        config: Dict[str, Any] = {},
        method: str | None = None,
        output_dir: str | None = None,
        log_config: Dict[str, Any] = {},
        exclusions: DataExclusions | None = None,
    ):
        """Initialize feature extraction pipeline.

        Args:
            config: Feature configuration dictionary
            method: Feature extraction method (overrides config if provided)
            output_dir: Output directory (overrides config if provided)
            log_config: Logging configuration dictionary
            exclusions: Known-missing/excluded-stack registry. ``None`` loads the
                tracked ``config/data_exclusions.yaml``; pass
                ``DataExclusions.empty()`` to disable it.
        """

        self.feature_config = config

        # Extract configuration sections
        # self.paths_config = feature_config.get('paths', {})
        self.method = method or self.feature_config.get("method", "incarta")
        self.output_config = self.feature_config.get("output", {})
        self.processing_config = self.feature_config.get("processing", {})

        # Validate method. Fail here, not per file: an unavailable method would
        # otherwise fail once per image.
        check_feature_method_available(self.method)
        self.exclusions = (
            exclusions if exclusions is not None else load_data_exclusions()
        )

        # Setup output directory first
        self._setup_output(output_dir, self.output_config)

        self.log_config = log_config
        # Library code must not configure logging; just obtain a module logger
        # and rely on the entry point (scripts/run_feature_extraction.py) having
        # called setup_logging(). Records propagate to the root logger.
        self.logger = logging.getLogger(__name__)

        # Initialize counters and results
        self.skipped_files = 0
        self.processed_files = 0
        self.error_files: List[Tuple[str, str]] = []
        self.expected_unpaired: List[str] = []
        self.all_features: List[pd.DataFrame] = []

    def _setup_output(
        self, output_dir: str | None = None, output_config: Dict[str, Any] | None = None
    ):
        """Setup output directory structure."""
        if output_dir:
            self.output_dir = Path(output_dir)
        elif output_config:
            self.output_dir = Path(output_config.get("output_dir", "output/features"))
        else:
            raise NotImplementedError("output_dir or output_config must be set")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FeatureExtractionPipeline":
        """Create pipeline instance from configuration dictionary."""

        feature_config = config.get("feature_extraction", {})
        log_config = config.get("logging", {})
        output_dir = config.get("paths", {}).get("output_dir")

        return cls(config=feature_config, output_dir=output_dir, log_config=log_config)

    def find_image_mask_pairs(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_patterns: List[str] | None = None,
        mask_patterns: List[str] | None = None,
    ) -> List[Tuple[Path, Path]]:
        """Find matching image and mask file pairs.

        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks
            image_patterns: List of glob patterns for images
            mask_patterns: List of glob patterns for masks

        Returns:
            List of (image_path, mask_path) tuples
        """

        pairs = []

        # Get file patterns
        image_patterns = image_patterns or [DEFAULT_IMAGE_PATTERN]
        mask_patterns = mask_patterns or [DEFAULT_MASK_PATTERN]

        self.logger.debug(f"Searching for image patterns: {image_patterns}")
        self.logger.debug(f"Searching for mask patterns: {mask_patterns}")

        # Find all image and mask files
        image_files = []
        mask_files = []

        for pattern in image_patterns:
            image_files.extend(image_dir.rglob(pattern))

        for pattern in mask_patterns:
            mask_files.extend(mask_dir.rglob(pattern))

        self.logger.info(
            f"Found {len(image_files)} potential image files and {len(mask_files)} mask files"
        )

        # Match files based on configuration
        pairs = self.match_files(
            image_files,
            mask_files,
            mask_patterns=mask_patterns,
            image_patterns=image_patterns,
        )

        self.logger.info(f"Successfully paired {len(pairs)} image-mask pairs")
        return pairs

    @staticmethod
    def _pair_key(name: str, pattern: str) -> Optional[str]:
        """Return the identifier captured by ``*`` in ``pattern`` for ``name``.

        The glob-style ``pattern`` is translated to an *anchored* regex in which
        each ``*`` becomes a capture group and every other character is matched
        literally (so ``.`` matches only a literal dot, not any char). The
        concatenation of the captured groups is the pairing key: an image and a
        mask pair iff their keys are **exactly equal**. Returns None if ``name``
        does not match ``pattern``.

        This replaces the old ``startswith(prefix)`` matching, which paired by
        prefix and so mismatched neighbouring identifiers — e.g. mask timepoint
        ``t21`` matched image ``t211`` because ``"t211...".startswith("t21")``,
        assigning one image to two masks. Exact-key equality is boundary-safe.
        """
        regex = "(.*)".join(re.escape(segment) for segment in pattern.split("*"))
        match = re.fullmatch(regex, name)
        if match is None:
            return None
        return "".join(match.groups())

    @classmethod
    def _first_key(cls, name: str, patterns: List[str]) -> Optional[str]:
        """Pairing key from the first ``patterns`` entry that matches ``name``."""
        for pattern in patterns:
            key = cls._pair_key(name, pattern)
            if key is not None:
                return key
        return None

    def find_image_given_mask(
        self,
        mask_path: Path,
        image_files: List[Path],
        mask_patterns: List[str] | None = None,
        image_patterns: List[str] | None = None,
    ) -> Optional[Path]:
        """Find the image whose pairing key equals this mask's pairing key.

        Args:
            mask_path: Path to the mask file
            image_files: List of available image files
            mask_patterns: Glob patterns for masks (default ``['*_Cells.tif']``)
            image_patterns: Glob patterns for images (default ``['*_BF.tif']``)

        Returns:
            Path to the matching image file, or None if not found
        """
        mask_patterns = mask_patterns or [DEFAULT_MASK_PATTERN]
        image_patterns = image_patterns or [DEFAULT_IMAGE_PATTERN]

        mask_key = self._first_key(mask_path.name, mask_patterns)
        if mask_key is None:
            self.logger.warning(f"Mask matches no pattern: {mask_path.name}")
            return None

        for image in image_files:
            if self._first_key(image.name, image_patterns) == mask_key:
                self.logger.debug(
                    f"Matched {image.name} with {mask_path.name} on key {mask_key}"
                )
                return image

        self.logger.warning(f"No matching image found for mask: {mask_path.name}")
        return None

    def match_files(
        self,
        image_files: List[Path],
        mask_files: List[Path],
        mask_patterns: List[str] | None = None,
        image_patterns: List[str] | None = None,
    ) -> List[Tuple[Path, Path]]:
        """Match image files with masks by exact pairing key (see ``_pair_key``).

        Images are indexed once by key, then each mask joins to the image with
        the same key. Because keys are the full wildcard-captured identifier
        (not a prefix), a mask pairs with exactly one image and neighbouring
        identifiers (``t21`` vs ``t211``) no longer collide.

        Args:
            image_files: List of image file paths
            mask_files: List of mask file paths
            mask_patterns: Glob patterns for masks (default ``['*_Cells.tif']``)
            image_patterns: Glob patterns for images (default ``['*_BF.tif']``)

        Returns:
            List of matched (image_path, mask_path) tuples
        """
        mask_patterns = mask_patterns or [DEFAULT_MASK_PATTERN]
        image_patterns = image_patterns or [DEFAULT_IMAGE_PATTERN]

        # Index images by pairing key; first occurrence wins, warn on collisions.
        image_by_key: Dict[str, Path] = {}
        for image in image_files:
            key = self._first_key(image.name, image_patterns)
            if key is None:
                continue
            if key in image_by_key:
                self.logger.warning(
                    f"Multiple images share pairing key '{key}': keeping "
                    f"{image_by_key[key].name}, ignoring {image.name}"
                )
                continue
            image_by_key[key] = image

        pairs = []
        paired_keys = set()
        for mask in mask_files:
            key = self._first_key(mask.name, mask_patterns)
            if key is None:
                self.logger.error(f"Mask matches no pattern: {mask.name}")
                self.error_files.append((str(mask), "Mask matches no pattern"))
                continue
            image = image_by_key.get(key)
            if image is None:
                self._record_unpaired_mask(mask, image_patterns)
                continue
            pairs.append((image, mask))
            paired_keys.add(key)

        n_unmasked = len(set(image_by_key) - paired_keys)
        if n_unmasked:
            # Expected for z0 projections, which are never segmented.
            self.logger.info(f"{n_unmasked} images have no mask and were not processed")

        return pairs

    def _record_unpaired_mask(self, mask: Path, image_patterns: List[str]) -> None:
        """Record a mask with no image: an error unless registered as known-missing.

        Known-missing only applies for a channel the image patterns select.
        """
        handler = self._get_file_handler()
        experiment = self.exclusions.experiment_of(mask)
        sample = handler.extract_sample_id(mask.name)
        timepoint = handler.extract_time_point(mask.name)
        z_index = handler.extract_z_index(mask.name)
        known = (
            experiment is not None
            and sample is not None
            and str(timepoint).isdigit()
            and z_index is not None
            and any(
                _pattern_has_channel(pattern, entry.channel)
                and self.exclusions.is_known_missing(
                    experiment, sample, int(timepoint), int(z_index), entry.channel
                )
                for entry in self.exclusions.known_missing
                for pattern in image_patterns
            )
        )
        if known:
            self.logger.info(
                f"No image for mask {mask.name}: registered as known-missing"
            )
            self.expected_unpaired.append(str(mask))
            return
        self.logger.error(f"No matching image found for mask: {mask.name}")
        self.error_files.append((str(mask), "No matching image"))

    @staticmethod
    def _load_pair(image_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load a BF image and its label mask; both must be 2D and the same shape.

        The mask goes through ``load_labels`` (tif/zarr/h5, any label dtype incl.
        uint32) and the image through ``load_image``, the reader inference used
        for the BF that produced the mask. Raises ``ValueError`` otherwise, which
        the per-file handler records as an error.
        """
        image = np.asarray(load_image(image_path))
        mask = np.asarray(load_labels(mask_path))
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f"Mask must have an integer label dtype, got {mask.dtype}")
        if image.ndim != 2 or mask.ndim != 2:
            raise ValueError(
                f"Expected a 2D image and mask, got image {image.shape} and "
                f"mask {mask.shape}"
            )
        if image.shape != mask.shape:
            raise ValueError(
                f"Image and mask shapes differ: {image.shape} vs {mask.shape}"
            )
        return image, mask

    '''
    def validate_mask(self, mask: np.ndarray, image_path: Path) -> bool:
        """Validate mask meets quality criteria.
        
        Args:
            mask: Mask array
            image_path: Path to corresponding image (for logging)
            
        Returns:
            True if mask is valid, False otherwise
        """
        validation_config = self.processing_config.get('validation', {})
        
        # Count instances
        unique_labels = np.unique(mask)
        n_instances = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
        
        min_instances = validation_config.get('min_instances_per_image', 1)
        max_instances = validation_config.get('max_instances_per_image', 10000)
        
        if n_instances < min_instances:
            logger.warning(f"Skipping {image_path.name}: too few instances ({n_instances})")
            return False
        
        if n_instances > max_instances:
            logger.warning(f"Skipping {image_path.name}: too many instances ({n_instances})")
            return False
        
        return True
    '''

    def _scportrait_mask_export_path(
        self,
        image_path: Path,
        sc_cfg: Dict[str, Any],
        injected: bool = False,
    ) -> Optional[Path]:
        """Build the destination TIFF path for the exported scPortrait mask.

        The export root is derived from the input image's sample folder (see
        ``derive_sample_dir``) so the mask lands alongside the Cellpose outputs
        under ``<sample>/inference_scportrait/`` (native) or, when ``injected``
        is set (Milestone 2), under ``<sample>/inference_scportrait_injected/``
        so the injected run never overwrites the native one. An explicit
        ``scportrait.mask_export_root`` (native) /
        ``scportrait.injected_mask_export_root`` (injected) config value
        overrides the derivation.

        Returns None (export skipped) when no sample folder can be derived and
        no override is configured.
        """
        root = sc_cfg.get(
            "injected_mask_export_root" if injected else "mask_export_root"
        )
        if root:
            export_root = Path(root)
        else:
            sample_dir = derive_sample_dir(image_path)
            if sample_dir is None:
                self.logger.warning(
                    "Could not derive a sample folder for %s; skipping "
                    "scPortrait mask export. Set scportrait.%s to export "
                    "anyway.",
                    image_path,
                    "injected_mask_export_root" if injected else "mask_export_root",
                )
                return None
            root_name = (
                SCPORTRAIT_INJECTED_ROOT_NAME if injected else SCPORTRAIT_MASK_ROOT_NAME
            )
            export_root = sample_dir / root_name

        stem = image_path.stem
        if stem.endswith("_BF"):
            stem = stem[: -len("_BF")]
        return export_root.joinpath(*SCPORTRAIT_MASK_SUBDIRS, f"{stem}_pred_mask.tif")

    def _get_file_handler(self) -> ConfigurableFileHandler:
        """Return a cached filename handler for per-cell metadata extraction.

        Built lazily (and per-process, so it survives loky worker pickling of
        ``self``) since it is only needed when ``include_metadata`` is set.
        """
        handler = getattr(self, "_file_handler", None)
        if handler is None:
            handler = ConfigurableFileHandler()
            self._file_handler = handler
        return handler

    def extract_features_from_path(
        self,
        image_path: Path | str,
        mask_path: Path | str | None = None,
        inner_n_jobs: int | None = None,
    ) -> Optional[pd.DataFrame]:
        """Extract features from a single image (+ mask for mask-based methods).

        Args:
            image_path: Path to image file
            mask_path: Path to mask file. Required for every method except
                'scportrait'. For 'scportrait' it selects the mode: ``None``
                runs scPortrait's own segmentation (native), while a supplied
                path is injected as an external mask and must exist, skipping
                scPortrait's internal Cellpose (Milestone 2).
            inner_n_jobs: Number of jobs for *inner* (per-cell) parallelism.
                When the outer file loop is parallelized (see ``process_batch``),
                the caller passes ``1`` so that N file workers do not each spawn
                cores' worth of cell workers (N*cores oversubscription). When
                None (default), the value falls back to ``feature_config.n_jobs``
                — the sequential-loop behavior.

        Returns:
            DataFrame with extracted features, or None if extraction fails
        """

        image_path = Path(image_path)
        mask_path = Path(mask_path) if mask_path is not None else None

        if not image_path.exists():
            self.logger.error(f"Image file does not exist: {image_path}")
            self.error_files.append((str(image_path), "File not found"))
            return None

        # scPortrait runs its own segmentation and normally needs no mask; every
        # other method requires an existing mask. Milestone 2: when a mask IS
        # supplied for scPortrait, it is injected (external cellpose_sam mask,
        # skipping scPortrait's internal Cellpose) and must exist.
        injecting = self.method == "scportrait" and mask_path is not None
        if self.method != "scportrait":
            if mask_path is None or not mask_path.exists():
                self.logger.error(f"Mask file does not exist: {mask_path}")
                self.error_files.append((str(mask_path), "File not found"))
                return None
        elif injecting and not mask_path.exists():
            self.logger.error(f"Injection mask does not exist: {mask_path}")
            self.error_files.append((str(mask_path), "Injection mask not found"))
            return None

        try:
            if self.method == "scportrait":
                # scPortrait takes file paths (not loaded arrays) and runs its own
                # segmentation/extraction/featurization, so handle it before loading.
                if get_scportrait_features is None:
                    raise RuntimeError(
                        "scportrait is not installed. Install it with 'pip install scportrait' to use this method."
                    )
                sc_cfg = self.feature_config.get("scportrait", {})
                # Injected runs get their own project subtree so native and
                # injected scPortrait projects (keyed by image stem) never clash.
                project_root = Path(
                    sc_cfg.get("project_location", "tmp/scportrait_projects")
                )
                if injecting:
                    project_root = project_root / "injected"
                project_location = str(project_root / image_path.stem)
                mask_export_path = self._scportrait_mask_export_path(
                    image_path, sc_cfg, injected=injecting
                )
                if injecting:
                    self.logger.info(
                        "scPortrait mask injection: %s <- %s",
                        image_path.name,
                        mask_path,
                    )
                features_df = get_scportrait_features(
                    image_paths=[str(image_path), str(image_path)],
                    channel_names=sc_cfg.get(
                        "channel_names", ["brightfield", "brightfield_ch1"]
                    ),
                    config_path=sc_cfg.get(
                        "config_path",
                        "src/feature_extraction/scportrait_project/config.yml",
                    ),
                    project_location=project_location,
                    overwrite=sc_cfg.get("overwrite", True),
                    debug=sc_cfg.get("debug", False),
                    mask_path=str(mask_path) if injecting else None,
                    plots_dir=(
                        str(Path(project_location) / "plots")
                        if sc_cfg.get("save_plots", True)
                        else None
                    ),
                    scportrait_mask_export_path=(
                        str(mask_export_path) if mask_export_path is not None else None
                    ),
                )
            else:
                assert mask_path is not None  # checked above for non-scportrait
                image, mask = self._load_pair(image_path, mask_path)

                # Extract features using the main function. ``inner_n_jobs``
                # controls per-cell parallelism; when the outer file loop is
                # parallel the caller passes 1 (see ``process_batch``).
                n_jobs = (
                    inner_n_jobs
                    if inner_n_jobs is not None
                    else self.feature_config.get("n_jobs", -1)
                )

                if self.method == "incarta":
                    features_df = extract_all_instance_features(
                        mask, image, n_jobs=n_jobs
                    )
                else:  # regionprops; the constructor rejected everything else
                    features_df = get_region_properties(mask, intensity_image=image)

            if features_df.empty:
                self.logger.warning(f"No features extracted from {image_path.name}")
                return None

            # Per-cell key columns parsed from the image filename
            # (e.g. ``pMF5V1_C09_t11_z10_BF.tif``). These are written
            # **unconditionally** — together with ``cell_id`` they form the join
            # key with the mcherry_metrics CSV
            # ``(sample_id, timepoint, z_index, cell_id)``, so they are data, not
            # the optional ``include_metadata`` provenance below. Mirrors
            # ``mcherry_metrics.io.loaders.extract_image_metadata``.
            handler = self._get_file_handler()
            name = image_path.name
            sample_id = handler.extract_sample_id(name)
            z_index = handler.extract_z_index(name)
            timepoint = handler.extract_time_point(name)
            features_df["sample_id"] = sample_id if sample_id is not None else ""
            features_df["timepoint"] = "" if timepoint == "unknown" else str(timepoint)
            features_df["z_index"] = -1 if z_index is None else int(z_index)

            # Optional provenance columns (filenames, dataset), gated by config.
            if self.output_config.get("include_metadata", True):
                features_df["image_filename"] = image_path.name
                if mask_path is not None:
                    features_df["mask_filename"] = mask_path.name
                # features_df['processing_timestamp'] = datetime.now().isoformat()
                # features_df['feature_extraction_version'] = '1.0'
                features_df["dataset_name"] = image_path.parent.name

            self.logger.debug(
                f"Extracted {len(features_df)} instances from {image_path.name}"
            )

            return features_df

        except _PROGRAMMING_ERRORS:
            raise
        except Exception as e:
            self.logger.error(f"Error extracting features from {image_path}: {str(e)}")
            self.error_files.append((str(image_path), str(e)))
            return None

    def save_image_features(self, features_df: pd.DataFrame, image_path: Path):
        """Save features for individual image to CSV file.

        Args:
            features_df: Features DataFrame
            image_path: Original image path (for naming output file)
        """

        # Create output filename
        output_format = self.output_config.get(
            "individual_format", "{image_name}_features.csv"
        )
        output_name = output_format.format(image_name=image_path.stem)

        # Create subdirectory if configured
        output_path = self.output_dir
        if self.output_config.get("create_subdirs", True):
            subdir = image_path.parent.name
            output_path = output_path / subdir
            output_path.mkdir(parents=True, exist_ok=True)

        # Save file
        output_file = output_path / output_name
        features_df.to_csv(output_file, index=False)
        self.logger.debug(f"Saved individual features to {output_file}")

    def process_batch(
        self,
        image_dir: Path | str,
        mask_dir: Path | str,
        image_patterns: List[str] | None = None,
        mask_patterns: List[str] | None = None,
    ) -> pd.DataFrame:
        """Process single dir containing images and masks, and extract features.

        Args:
            image_dir: Directory containing images (if None, uses config)
            mask_dir: Directory containing masks (if None, uses config)
            image_patterns: List of glob patterns for images
            mask_patterns: List of glob patterns for masks

        Returns:
            Combined DataFrame with all features
        """

        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)

        self.logger.info(f"Processing dataset: {mask_dir} with images from {image_dir}")
        errors_before = len(self.error_files)
        expected_before = len(self.expected_unpaired)

        # Find image-mask pairs
        pairs = self.find_image_mask_pairs(
            image_dir,
            mask_dir,
            image_patterns=image_patterns,
            mask_patterns=mask_patterns,
        )
        if not pairs:
            only_known_missing = (
                len(self.expected_unpaired) > expected_before
                and len(self.error_files) == errors_before
            )
            if only_known_missing:
                # Every mask in the batch is a registered known-missing input.
                self.logger.info(
                    f"No image-mask pairs in {image_dir}: every unpaired mask is "
                    "registered as known-missing"
                )
                return pd.DataFrame()
            self.logger.error(f"No valid image-mask pairs found in {image_dir}")
            self.error_files.append((str(mask_dir), "No valid image-mask pairs"))
            return pd.DataFrame()

        # Process pairs. scPortrait runs its own GPU inference and is kept
        # sequential (one image per GPU); CPU methods fan out across file
        # workers driven by ``n_jobs``.
        n_workers = self._resolve_file_workers()
        if self.method == "scportrait" or n_workers <= 1:
            all_features, processed_files = self._process_pairs_sequential(pairs)
        else:
            self.logger.info(
                f"Parallelizing feature extraction across {n_workers} file workers"
            )
            all_features, processed_files = self._process_pairs_parallel(
                pairs, n_workers
            )

        self.processed_files += processed_files
        self._report_errors(errors_before, len(pairs))

        # Combine all features
        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            self.logger.info(
                f"Total features extracted: {len(combined_df)} instances "
                f"from {processed_files} images"
            )
        else:
            combined_df = pd.DataFrame()
            self.logger.warning("No features extracted from any files")

        return combined_df

    def _report_errors(self, errors_before: int, n_inputs: int) -> None:
        """Log this batch's error count so it reaches the run log, not only the summary."""
        batch_errors = self.error_files[errors_before:]
        if not batch_errors:
            return
        preview = "; ".join(f"{path}: {msg}" for path, msg in batch_errors[:5])
        self.logger.warning(
            f"{len(batch_errors)} file error(s) across {n_inputs} inputs. "
            f"First: {preview}"
        )

    def _resolve_file_workers(self) -> int:
        """Resolve the number of concurrent file workers from ``n_jobs``.

        ``--n-jobs`` (``feature_config['n_jobs']``) is repurposed as the number
        of images processed *concurrently* by the outer file loop. ``-1`` means
        "all available cores"; ``0`` or ``None`` means sequential (1).
        """
        raw = self.feature_config.get("n_jobs", -1)
        if raw in (None, 0):
            return 1
        if raw < 0:
            return os.cpu_count() or 1
        return int(raw)

    def _process_pairs_sequential(
        self,
        pairs: List[Tuple[Path, Path]],
    ) -> Tuple[List[pd.DataFrame], int]:
        """Process (image, mask) pairs one at a time (original behavior)."""
        processed_files = 0
        all_features: List[pd.DataFrame] = []
        save_individual = self.output_config.get("save_individual_files", True)

        for image_path, mask_path in tqdm(pairs, desc="Processing files"):
            features_df = self.extract_features_from_path(image_path, mask_path)

            if features_df is not None:
                all_features.append(features_df)
                processed_files += 1

                # Save individual file if configured
                if save_individual:
                    self.save_image_features(features_df, image_path)

        return all_features, processed_files

    def _process_pairs_parallel(
        self,
        pairs: List[Tuple[Path, Path]],
        n_workers: int,
    ) -> Tuple[List[pd.DataFrame], int]:
        """Process (image, mask) pairs concurrently across ``n_workers`` procs.

        Uses a joblib ``loky`` process pool. Each worker runs one image with
        inner (per-cell) parallelism disabled (``inner_n_jobs=1``) to avoid
        N*cores oversubscription, saves its own per-image CSV, and returns
        ``(features_df, error_records)``. Results are yielded in submission
        order, so the combined table is identical to the sequential path. Error
        records are aggregated back into ``self.error_files`` here, since worker
        mutations of ``self`` do not cross process boundaries.

        Note: for near-linear speedup, set ``OMP_NUM_THREADS=1`` in the launch
        environment (see ``slurm/feature_extraction.sbatch``) so BLAS/OpenMP
        threads in each worker do not oversubscribe the cores.
        """
        save_individual = self.output_config.get("save_individual_files", True)
        processed_files = 0
        all_features: List[pd.DataFrame] = []

        results = Parallel(n_jobs=n_workers, backend="loky", return_as="generator")(
            delayed(_extract_one_pair)(self, image_path, mask_path, save_individual)
            for image_path, mask_path in pairs
        )
        for features_df, new_errors in tqdm(
            results, total=len(pairs), desc="Processing files"
        ):
            if new_errors:
                # Worker processes log to their own stderr, not the run's log
                # file, so re-log here in the parent.
                for path, msg in new_errors:
                    self.logger.error(f"Error in file worker for {path}: {msg}")
                self.error_files.extend(new_errors)
            if features_df is not None:
                all_features.append(features_df)
                processed_files += 1

        return all_features, processed_files

    def find_images(
        self,
        image_dir: Path | str,
        image_patterns: List[str] | None = None,
    ) -> List[Path]:
        """Find image files in a directory (no mask pairing).

        Used by segmentation-free methods (scPortrait) that only need input
        images.

        Args:
            image_dir: Directory containing images
            image_patterns: List of glob patterns for images (default ['*_BF.tif'])

        Returns:
            Sorted, de-duplicated list of image paths
        """
        image_dir = Path(image_dir)
        image_patterns = image_patterns or [DEFAULT_IMAGE_PATTERN]
        self.logger.debug(f"Searching for image patterns: {image_patterns}")

        image_files: List[Path] = []
        for pattern in image_patterns:
            image_files.extend(image_dir.rglob(pattern))

        image_files = sorted(set(image_files))
        self.logger.info(f"Found {len(image_files)} image files in {image_dir}")
        return image_files

    def process_batch_scportrait(
        self,
        image_dir: Path | str,
        image_patterns: List[str] | None = None,
        mask_dir: Path | str | None = None,
        mask_pattern: str | None = None,
    ) -> pd.DataFrame:
        """Extract scPortrait features from every image in a directory.

        Two modes:
        * **Native (default, ``mask_dir=None``):** scPortrait runs its own
          segmentation; images are discovered directly (no mask pairing).
        * **Injection (Milestone 2, ``mask_dir`` set):** each BF image's
          cellpose_sam mask is resolved by stem and injected, skipping
          scPortrait's internal Cellpose. Images with no resolvable mask are
          skipped (recorded as errors), not silently dropped.

        Args:
            image_dir: Directory containing input images
            image_patterns: List of glob patterns for images
            mask_dir: Directory of external masks to inject (None => native mode)
            mask_pattern: Filename pattern for masks (``{stem}`` placeholder);
                defaults to ``{stem}_pred_mask.tif``

        Returns:
            Combined DataFrame with features from all images
        """
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir) if mask_dir else None
        mask_pat = mask_pattern or "{stem}_pred_mask.tif"
        mode = f"injection (masks from {mask_dir})" if mask_dir else "native"
        self.logger.info(
            f"Processing scPortrait batch [{mode}] from images in {image_dir}"
        )
        if mask_dir is not None:
            if "{stem}" not in mask_pat:
                # A glob such as "*_pred_mask.tif" survives .format() unchanged
                # and then matches every mask in the tree, so each lookup raises
                # and the run finishes empty with exit 0. Fall back loudly rather
                # than fail quietly. This is the single validation point: callers
                # forward whatever pattern they were configured with.
                self.logger.warning(
                    "Ignoring mask_pattern %r for scPortrait injection: it has "
                    "no '{stem}' placeholder. Using the default template instead.",
                    mask_pat,
                )
                mask_pat = "{stem}_pred_mask.tif"
            self._warn_on_existing_output()

        images = self.find_images(image_dir, image_patterns=image_patterns)
        if not images:
            self.logger.error(f"No images found in {image_dir}")
            self.error_files.append((str(image_dir), "No images found"))
            return pd.DataFrame()

        processed_files = 0
        all_features = []
        errors_before = len(self.error_files)
        for image_path in tqdm(images, desc="Processing images"):
            inject_mask: Path | None = None
            if mask_dir is not None:
                try:
                    inject_mask = resolve_cellpose_mask(image_path, mask_dir, mask_pat)
                except FileNotFoundError as exc:
                    self.logger.warning("Skipping %s: %s", image_path.name, exc)
                    self.error_files.append((str(image_path), str(exc)))
                    continue
            features_df = self.extract_features_from_path(
                image_path, mask_path=inject_mask
            )
            if features_df is not None:
                all_features.append(features_df)
                processed_files += 1
                if self.output_config.get("save_individual_files", True):
                    self.save_image_features(features_df, image_path)

        self.processed_files += processed_files
        self._report_errors(errors_before, len(images))

        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            self.logger.info(
                f"Total features extracted: {len(combined_df)} instances from {processed_files} images"
            )
        else:
            combined_df = pd.DataFrame()
            self.logger.warning("No features extracted from any images")

        return combined_df

    def process_single_image(
        self,
        image_path: Path | str,
        mask_path: Path | str | None = None,
    ) -> Optional[pd.DataFrame]:
        """Extract features from a single image and save the results.

        For scPortrait, ``mask_path`` selects the mode: omit it for native
        segmentation, or pass a mask to inject it and skip scPortrait's internal
        Cellpose. For mask-based methods it is required.

        Args:
            image_path: Path to the input image
            mask_path: Path to the mask file (required for non-scportrait methods)

        Returns:
            Features DataFrame, or None if extraction failed / no features
        """
        image_path = Path(image_path)
        self.logger.info(f"Processing single image: {image_path}")

        features_df = self.extract_features_from_path(image_path, mask_path)
        if features_df is None or features_df.empty:
            self.logger.warning(f"No features extracted from {image_path.name}")
            return features_df
        self.processed_files += 1

        if self.output_config.get("save_individual_files", True):
            self.save_image_features(features_df, image_path)
        self.save_combined_features(features_df)
        return features_df

    def _warn_on_existing_output(self) -> None:
        """Warn when an injected run is about to overwrite existing CSVs.

        Only the exported masks are namespaced by ``SCPORTRAIT_INJECTED_ROOT_NAME``;
        the feature CSVs go to ``self.output_dir``. Pointing an injected run at a
        native run's output directory silently replaces its output, so say so
        rather than overwrite quietly.

        Both output shapes are checked. The shipped config has
        ``save_combined_file: false`` with ``save_individual_files: true``, so a
        native run commonly leaves *only* per-image CSVs and no combined file --
        looking for the combined file alone would miss the usual case. Per-image
        files may sit in a per-source subdirectory (``create_subdirs``), hence
        the recursive search.
        """
        existing = None
        combined = self.output_dir / self.output_config.get(
            "combined_filename", "all_features.csv"
        )
        if combined.exists():
            existing = combined
        else:
            individual = self.output_config.get(
                "individual_format", "{image_name}_features.csv"
            ).format(image_name="*")
            existing = next(iter(sorted(self.output_dir.rglob(individual))), None)

        if existing is not None:
            self.logger.warning(
                "Injected scPortrait run will overwrite existing feature output "
                "in %s (e.g. %s). Injected masks are namespaced, feature CSVs are "
                "NOT -- pass a separate --output-dir to keep a native run's CSVs.",
                self.output_dir,
                existing.name,
            )

    def save_combined_features(self, features_df: pd.DataFrame):
        """Save combined features to CSV file.

        Args:
            features_df: Combined features DataFrame
        """
        if not self.output_config.get("save_combined_file", True):
            return

        if features_df.empty:
            self.logger.warning("No features to save")
            return

        # Save combined file
        combined_filename = self.output_config.get(
            "combined_filename", "all_features.csv"
        )
        output_file = self.output_dir / combined_filename

        features_df.to_csv(output_file, index=False)
        self.logger.info(f"Saved combined features to {output_file}")

    def save_summary(self, features_df: pd.DataFrame) -> Path:
        """Write the run summary, including every per-file error.

        Written regardless of ``save_combined_file``: it is the durable record
        of which inputs failed.

        Returns:
            Path of the summary file.
        """
        summary_file = self.output_dir / "feature_extraction_summary.txt"
        with open(summary_file, "w") as f:
            f.write("Feature Extraction Summary\n")
            f.write("========================\n\n")
            f.write(f"Processing completed: {datetime.now()}\n")
            f.write(f"Method: {self.method}\n")
            f.write(f"Files processed: {self.processed_files}\n")
            f.write(f"Files skipped: {self.skipped_files}\n")
            f.write(f"Files with errors: {len(self.error_files)}\n")
            f.write(f"Expected unpaired masks: {len(self.expected_unpaired)}\n")
            f.write(f"Total instances: {len(features_df)}\n")
            f.write(f"Total columns per instance: {len(features_df.columns)}\n\n")

            if self.error_files:
                f.write("Error Files:\n")
                for filepath, error in self.error_files:
                    f.write(f"  {filepath}: {error}\n")

            if self.expected_unpaired:
                f.write("\nExpected unpaired masks (known-missing input):\n")
                for filepath in self.expected_unpaired:
                    f.write(f"  {filepath}\n")

            f.write("\nFeature Columns:\n")
            for col in features_df.columns:
                f.write(f"  {col}\n")

        self.logger.info(f"Saved processing summary to {summary_file}")
        return summary_file

    def raise_if_errors(self, summary_file: Path | None = None) -> None:
        """Raise :class:`FeatureExtractionError` if any input file failed."""
        if not self.error_files:
            return
        where = f"; see {summary_file}" if summary_file is not None else ""
        raise FeatureExtractionError(
            f"{len(self.error_files)} input file(s) failed during feature "
            f"extraction{where}"
        )

    def run(
        self, image_dirs: List[Path] = [], mask_dirs: List[Path] = []
    ) -> pd.DataFrame:
        """Run the complete feature extraction pipeline.

        Args:
            image_dirs: List of image directories (if None, uses config)
            mask_dirs: List of mask directories (if None, uses config)

        Returns:
            Combined features DataFrame
        """
        start_time = time.time()
        self.logger.info("Starting feature extraction pipeline")

        all_datasets_features = []

        if image_dirs and mask_dirs and len(image_dirs) != len(mask_dirs):
            self.logger.error(
                "Number of image directories must match number of mask directories"
            )
            return pd.DataFrame()

        # Process each directory
        try:
            for image_dir, mask_dir in zip(image_dirs, mask_dirs):
                self.logger.info(f"Processing directory: {image_dir}")
                features_df = self.process_batch(
                    image_dir,
                    mask_dir,
                    image_patterns=[
                        self.feature_config.get("image_pattern")
                        or DEFAULT_IMAGE_PATTERN
                    ],
                    mask_patterns=[
                        self.feature_config.get("mask_pattern") or DEFAULT_MASK_PATTERN
                    ],
                )

                if not features_df.empty:
                    all_datasets_features.append(features_df)
        except BaseException:
            # A code defect re-raised mid-run: still leave a summary of what was done.
            self.save_summary(pd.DataFrame())
            raise

        # Combine all datasets
        if all_datasets_features:
            final_features = pd.concat(all_datasets_features, ignore_index=True)
        else:
            final_features = pd.DataFrame()

        # Save results; any per-file error then fails the run.
        self.save_combined_features(final_features)
        summary_file = self.save_summary(final_features)

        # Log completion
        elapsed_time = time.time() - start_time
        self.logger.info(f"Feature extraction completed in {elapsed_time:.2f} seconds")
        self.logger.info(
            f"Final results: {len(final_features)} instances from "
            f"{self.processed_files} images"
        )
        self.raise_if_errors(summary_file)

        return final_features
