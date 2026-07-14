import tifffile as tiff
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union

try:
    import zarr
    _ZARR_AVAILABLE = True
    _ZARR_MAJOR = int(zarr.__version__.split(".")[0])
    if _ZARR_MAJOR >= 3:
        from zarr.codecs import BloscCodec as _BloscCodec  # type: ignore[import-untyped]
    else:
        from numcodecs import Blosc as _Blosc  # type: ignore[import-untyped]
except ImportError:
    _ZARR_AVAILABLE = False
    _ZARR_MAJOR = 0

import importlib.util as _importlib_util
_H5PY_AVAILABLE = _importlib_util.find_spec("h5py") is not None

#: Supported label formats → file extensions.
LABEL_FORMATS: Dict[str, str] = {"tif": ".tif", "zarr": ".zarr", "hdf5": ".h5"}


def load_image(file_path: Union[str, Path]) -> np.ndarray:
    file_path = Path(file_path)

    if file_path.suffix.lower() in ['.tif', '.tiff']:
        image = tiff.imread(str(file_path))
    else:
        from PIL import Image
        image = np.array(Image.open(file_path))

    return image


def _optimal_label_dtype(arr: np.ndarray) -> type:
    max_val = int(arr.max())
    if max_val <= np.iinfo(np.uint8).max:
        return np.uint8
    elif max_val <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def _normalize_label_dtype(masks: np.ndarray) -> np.ndarray:
    """Cast instance-label array to uint8, uint16, or uint32 as needed."""
    return masks.astype(_optimal_label_dtype(masks))


def save_labels(masks: np.ndarray, output_path: Union[str, Path], chunks: Tuple | None = None) -> None:
    """
    Save an instance-label array; output format is determined by the file extension:

    * ``.tif`` / ``.tiff``  LZW-compressed TIFF via tifffile
    * ``.zarr``             Zarr directory store with Blosc/zstd compression
    * ``.h5``               HDF5 file with gzip compression via h5py

    The array dtype is normalised to uint8/uint16/uint32 before writing.
    """
    output_path = Path(output_path)
    masks = _normalize_label_dtype(masks)
    ext = output_path.suffix.lower()

    if ext in (".tif", ".tiff"):
        tiff.imwrite(output_path, masks, compression="lzw")

    elif ext == ".zarr":
        if not _ZARR_AVAILABLE:
            raise ImportError(
                "zarr is required to save .zarr labels. "
                "Install it with: pip install zarr"
            )
        chunks = chunks or ((1, masks.shape[-2], masks.shape[-1]) if masks.ndim == 3 else None)

        if _ZARR_MAJOR >= 3:
            z = zarr.create_array(  # type: ignore[attr-defined]
                store=str(output_path),
                shape=masks.shape,
                dtype=masks.dtype,
                chunks=chunks,
                compressors=_BloscCodec(cname="zstd", clevel=3),  # type: ignore[name-defined]
                overwrite=True,
            )
        else:
            compressor = _Blosc(cname="zstd", clevel=3, shuffle=_Blosc.BITSHUFFLE)  # type: ignore[name-defined]
            z = zarr.open(
                str(output_path), mode="w",
                shape=masks.shape, dtype=masks.dtype,
                chunks=chunks, compressor=compressor,
            )
        z[:] = masks

    elif ext == ".h5":
        if not _H5PY_AVAILABLE:
            raise ImportError(
                "h5py is required to save .h5 labels. "
                "Install it with: pip install h5py"
            )
        import h5py
        chunks = chunks or ((1, masks.shape[-2], masks.shape[-1]) if masks.ndim == 3 else None)
        with h5py.File(output_path, "w") as f:
            f.create_dataset(
                "labels", data=masks,
                compression="gzip", compression_opts=4,
                chunks=chunks,
            )

    else:
        raise ValueError(f"Unsupported label file extension: {ext!r}")


def load_labels(input_path: Union[str, Path]) -> np.ndarray:
    """
    Load an instance-label array from a tif, zarr, or hdf5 file.
    """
    input_path = Path(input_path)
    ext = input_path.suffix.lower()

    if ext in (".tif", ".tiff"):
        return tiff.imread(str(input_path))

    elif ext == ".zarr":
        if not _ZARR_AVAILABLE:
            raise ImportError(
                "zarr is required to load .zarr labels. "
                "Install it with: pip install zarr"
            )
        z = zarr.open_array(str(input_path), mode="r") if _ZARR_MAJOR >= 3 else zarr.open(str(input_path), mode="r")  # type: ignore[attr-defined]
        return np.asarray(z)

    elif ext == ".h5":
        if not _H5PY_AVAILABLE:
            raise ImportError(
                "h5py is required to load .h5 labels. "
                "Install it with: pip install h5py"
            )
        import h5py
        with h5py.File(input_path, "r") as f:
            ds = f["labels"]
            assert isinstance(ds, h5py.Dataset)
            return np.asarray(ds)

    else:
        raise ValueError(f"Unsupported label file extension: {ext!r}")
