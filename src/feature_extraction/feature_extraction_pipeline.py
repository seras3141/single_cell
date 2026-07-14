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
import cv2

from src.feature_extraction.feature_extractor_incarta import (
    extract_all_instance_features,
)

try:
    from src.feature_extraction.feature_extractor_pyradiomics import (
        get_radiomics_features,
    )
except ImportError:
    get_radiomics_features = None
try:
    from src.feature_extraction.feature_extractor_scportrait import (
        get_scportrait_features,
    )
except ImportError:
    get_scportrait_features = None
from src.feature_extraction.feature_extractor_regionprops import get_region_properties
from src.utils.file_utils import ConfigurableFileHandler

# scPortrait label-mask export layout. The exported mask mirrors the Cellpose
# tracked-mask tree but lives under a sibling ``inference_scportrait/`` dir:
#   <sample>/inference_scportrait/scportrait/test/final_2d/<stem>_pred_mask.tif
SCPORTRAIT_MASK_ROOT_NAME = "inference_scportrait"
SCPORTRAIT_MASK_SUBDIRS = ("scportrait", "test", "final_2d")

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
    ):
        """Initialize feature extraction pipeline.

        Args:
            config: Feature configuration dictionary
            method: Feature extraction method (overrides config if provided)
            output_dir: Output directory (overrides config if provided)
            log_config: Logging configuration dictionary
        """

        self.feature_config = config

        # Extract configuration sections
        # self.paths_config = feature_config.get('paths', {})
        self.method = method or self.feature_config.get("method", "incarta")
        self.output_config = self.feature_config.get("output", {})
        self.processing_config = self.feature_config.get("processing", {})

        # Validate method
        if self.method not in ["incarta", "regionprops", "pyradiomics", "scportrait"]:
            raise ValueError(f"Unsupported feature extraction method: {self.method}")

        # Setup output directory first
        self._setup_output(output_dir, self.output_config)

        self.log_config = log_config
        # Library code must not configure logging; just obtain a module logger
        # and rely on the entry point (scripts/run_feature_extraction.py) having
        # called setup_logging(). Records propagate to the root logger.
        self.logger = logging.getLogger(__name__)

        # Initialize counters and results
        self.skipped_files = 0
        self.error_files = []
        self.all_features = []

    def _setup_output(
        self, output_dir: str | None = None, output_config: Dict[str, Any] | None = None
    ):
        """Setup output directory structure."""
        if output_dir:
            self.output_dir = Path(output_dir)
        elif output_config:
            self.output_dir = output_config.get("output_dir", "output/features")
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
        image_patterns = image_patterns or ["*_BF.tif"]
        mask_patterns = mask_patterns or ["*_Cells.tif"]

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
        mask_patterns = mask_patterns or ["*_Cells.tif"]
        image_patterns = image_patterns or ["*_BF.tif"]

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
        mask_patterns = mask_patterns or ["*_Cells.tif"]
        image_patterns = image_patterns or ["*_BF.tif"]

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
        for mask in mask_files:
            key = self._first_key(mask.name, mask_patterns)
            if key is None:
                self.logger.warning(f"Mask matches no pattern: {mask.name}")
                continue
            image = image_by_key.get(key)
            if image is None:
                self.logger.warning(f"No matching image found for mask: {mask.name}")
                continue
            pairs.append((image, mask))

        return pairs

    # Why use custom function and preprocessing here instead of inside extract_all_instance_features?
    def load_image_and_mask(
        self, image_path: Path, mask_path: Path
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load image and mask files.

        Args:
            image_path: Path to image file
            mask_path: Path to mask file

        Returns:
            Tuple of (image, mask) arrays, or (None, None) if loading fails
        """

        try:
            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None, None

            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                self.logger.error(f"Failed to load mask: {mask_path}")
                return None, None

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # Validate dimensions match
            if image.shape != mask.shape:
                self.logger.error(
                    f"Image and mask dimensions don't match: {image.shape} vs {mask.shape}"
                )
                return None, None

            # Apply preprocessing if configured
            preprocessing = self.feature_config.get("preprocessing", {})
            if preprocessing.get("normalize_intensity", False):
                image = image.astype(np.float32) / 255.0

            clip_percentiles = preprocessing.get("clip_percentiles")
            if clip_percentiles:
                lower, upper = clip_percentiles
                p_low, p_high = np.percentile(image, [lower, upper])  # type: ignore
                image = np.clip(image, p_low, p_high)

            return image, mask

        except Exception as e:
            self.logger.error(f"Error loading {image_path} and {mask_path}: {str(e)}")
            return None, None

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
    ) -> Optional[Path]:
        """Build the destination TIFF path for the exported scPortrait mask.

        The export root is derived from the input image's sample folder (see
        ``derive_sample_dir``) so the mask lands alongside the Cellpose outputs
        under ``<sample>/inference_scportrait/``. An explicit
        ``scportrait.mask_export_root`` config value overrides the derivation.

        Returns None (export skipped) when no sample folder can be derived and
        no override is configured.
        """
        root = sc_cfg.get("mask_export_root")
        if root:
            export_root = Path(root)
        else:
            sample_dir = derive_sample_dir(image_path)
            if sample_dir is None:
                self.logger.warning(
                    "Could not derive a sample folder for %s; skipping "
                    "scPortrait mask export. Set scportrait.mask_export_root "
                    "to export anyway.",
                    image_path,
                )
                return None
            export_root = sample_dir / SCPORTRAIT_MASK_ROOT_NAME

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
            mask_path: Path to mask file. Optional and ignored for the
                'scportrait' method (which runs its own segmentation); required
                for all other methods.
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

        # scPortrait runs its own segmentation and needs no mask; every other
        # method requires an existing mask.
        if self.method != "scportrait":
            if mask_path is None or not mask_path.exists():
                self.logger.error(f"Mask file does not exist: {mask_path}")
                self.error_files.append((str(mask_path), "File not found"))
                return None

        try:
            if self.method == "scportrait":
                # scPortrait takes file paths (not loaded arrays) and runs its own
                # segmentation/extraction/featurization, so handle it before cv2.imread.
                if get_scportrait_features is None:
                    raise RuntimeError(
                        "scportrait is not installed. Install it with 'pip install scportrait' to use this method."
                    )
                sc_cfg = self.feature_config.get("scportrait", {})
                project_location = str(
                    Path(sc_cfg.get("project_location", "tmp/scportrait_projects"))
                    / image_path.stem
                )
                mask_export_path = self._scportrait_mask_export_path(image_path, sc_cfg)
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
                # Load image and mask
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if image is None or mask is None:
                    self.logger.error(
                        f"Failed to load image: {image_path} or mask: {mask_path}"
                    )
                    return None

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
                elif self.method == "regionprops":
                    features_df = get_region_properties(mask, intensity_image=image)
                elif self.method == "pyradiomics":
                    # DISCONTINUED: the pyradiomics backend is being reimplemented
                    # in a future version (separate branch) and is not supported
                    # here. It is intentionally excluded from file-level
                    # parallelization until then.
                    raise NotImplementedError(
                        "The 'pyradiomics' feature-extraction method is discontinued "
                        "and will be reimplemented in a future version. Use 'incarta' "
                        "or 'regionprops' instead."
                    )
                else:
                    raise ValueError(
                        f"Unknown feature extraction method: {self.method}"
                    )

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

        # Find image-mask pairs
        pairs = self.find_image_mask_pairs(
            image_dir,
            mask_dir,
            image_patterns=image_patterns,
            mask_patterns=mask_patterns,
        )
        if not pairs:
            self.logger.error(f"No valid image-mask pairs found in {image_dir}")
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
        image_patterns = image_patterns or ["*_BF.tif"]
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
    ) -> pd.DataFrame:
        """Extract scPortrait features from every image in a directory.

        Mask-free counterpart to ``process_batch``: scPortrait runs its own
        segmentation, so images are discovered directly (no mask pairing).

        Args:
            image_dir: Directory containing input images
            image_patterns: List of glob patterns for images

        Returns:
            Combined DataFrame with features from all images
        """
        image_dir = Path(image_dir)
        self.logger.info(f"Processing scPortrait batch from images in {image_dir}")

        images = self.find_images(image_dir, image_patterns=image_patterns)
        if not images:
            self.logger.error(f"No images found in {image_dir}")
            return pd.DataFrame()

        processed_files = 0
        all_features = []
        for image_path in tqdm(images, desc="Processing images"):
            features_df = self.extract_features_from_path(image_path, mask_path=None)
            if features_df is not None:
                all_features.append(features_df)
                processed_files += 1
                if self.output_config.get("save_individual_files", True):
                    self.save_image_features(features_df, image_path)

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

        For scPortrait, ``mask_path`` is optional (segmentation is internal);
        for mask-based methods it is required.

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

        if self.output_config.get("save_individual_files", True):
            self.save_image_features(features_df, image_path)
        self.save_combined_features(features_df)
        return features_df

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

        # Save summary statistics
        summary_file = self.output_dir / "feature_extraction_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Feature Extraction Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Processing completed: {datetime.now()}\n")
            f.write(f"Total files processed: {len(features_df)}\n")
            f.write(f"Files skipped: {self.skipped_files}\n")
            f.write(f"Files with errors: {len(self.error_files)}\n")
            f.write(f"Total instances: {len(features_df)}\n")
            f.write(f"Total features per instance: {len(features_df.columns)}\n\n")

            if self.error_files:
                f.write("Error Files:\n")
                for filepath, error in self.error_files:
                    f.write(f"  {filepath}: {error}\n")

            f.write(f"\nFeature Columns:\n")
            for col in features_df.columns:
                f.write(f"  {col}\n")

        self.logger.info(f"Saved processing summary to {summary_file}")

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
        for image_dir, mask_dir in zip(image_dirs, mask_dirs):
            self.logger.info(f"Processing directory: {image_dir}")
            features_df = self.process_batch(image_dir, mask_dir)

            if not features_df.empty:
                all_datasets_features.append(features_df)

        # Combine all datasets
        if all_datasets_features:
            final_features = pd.concat(all_datasets_features, ignore_index=True)
        else:
            final_features = pd.DataFrame()

        # Save results
        self.save_combined_features(final_features)

        # Log completion
        elapsed_time = time.time() - start_time
        self.logger.info(f"Feature extraction completed in {elapsed_time:.2f} seconds")
        self.logger.info(
            f"Final results: {len(final_features)} instances from xxx images"
        )

        return final_features
