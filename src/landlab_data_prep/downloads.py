"""Downloads into a persistent cache, written atomically.

A cached file only ever appears after a complete download, so reusing it is
safe. Delete the cache directory to force a fresh download.

A remote GeoTIFF is read in place when a grid is known: only the part around
the grid is fetched, by HTTP range requests, and cached at the file's own
pixels. A nationwide soil or land-cover layer then costs megabytes, not
gigabytes, for a small AOI. Zips are always downloaded whole.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING
import zipfile

import rasterio
from rasterio.errors import RasterioError
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
import requests

if TYPE_CHECKING:
    from landlab_data_prep.analysis_grid import Grid

LOG = logging.getLogger(__name__)
CHUNK_BYTES = 16 * 1024 * 1024
_EXTRACTED_MARKER = ".extracted"
# The window around the grid reaches past it by more than any resampling kernel
# looks: a cubic kernel spans 2 source pixels when the grid is finer than the
# source, and 2 grid cells when it is coarser. A warp from the window is then
# the same as a warp from the whole file, to float32 rounding.
WINDOW_MARGIN_CELLS = 4
WINDOW_PAD_PIXELS = 4
# GDAL settings for reading a remote file in place.
REMOTE_READ_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",  # do not list the server's folder
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


def _cache_name(url: str) -> str:
    """A file name for ``url``: a short hash, so two URLs never share one, and the URL's own name."""
    name = Path(url.split("?", 1)[0]).name or "download"
    return f"{hashlib.sha1(url.encode()).hexdigest()[:12]}_{name}"


def is_zip_url(url: str) -> bool:
    return Path(url.split("?", 1)[0]).suffix.lower() == ".zip"


def fetch_raster(url: str, cache_dir: str | Path, grid: Grid | None = None) -> Path:
    """A local GeoTIFF for a remote one: the window around ``grid`` when given, else the whole file.

    A file that cannot be read in place, such as one on a server without range
    requests, is downloaded whole instead.
    """
    if grid is not None:
        try:
            return cached_window(url, cache_dir, grid)
        except (RasterioError, OSError) as exc:
            LOG.warning("Cannot read %s in place (%s); downloading the whole file instead.", url, exc)
    return cached_download(url, cache_dir)


def cached_window(url: str, cache_dir: str | Path, grid: Grid) -> Path:
    """The part of a remote GeoTIFF that covers ``grid``, cached at the file's own pixels and values.

    Band 1 only, with the source's CRS, nodata and scale. Putting it on the grid
    is left to ``analysis_grid.align_to_grid``, as for any local file.
    """
    target = Path(cache_dir) / "windows" / _cache_name(url) / f"{grid.cache_key}.tif"
    if target.is_file() and target.stat().st_size > 0:
        return target

    with rasterio.Env(**REMOTE_READ_OPTIONS), rasterio.open(url) as src:
        window = _window_around(src, grid)
        data = src.read(1, window=window)
        profile = {
            "driver": "GTiff",
            "width": int(window.width),
            "height": int(window.height),
            "count": 1,
            "dtype": src.dtypes[0],
            "crs": src.crs,
            "transform": src.window_transform(window),
            "nodata": src.nodata,
            "compress": "deflate",
        }
        scale, offset = src.scales[0], src.offsets[0]

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        with rasterio.open(partial, "w", **profile) as dst:
            dst.write(data, 1)
            dst.scales, dst.offsets = (scale,), (offset,)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def _window_around(src, grid: Grid) -> Window:
    """Source pixels under ``grid``, its margin and a few pixels of padding, clipped to the file."""
    left, bottom, right, top = grid.bounds
    margin = WINDOW_MARGIN_CELLS * grid.resolution
    bounds = transform_bounds(
        grid.crs, src.crs, left - margin, bottom - margin, right + margin, top + margin, densify_pts=21
    )
    window = from_bounds(*bounds, transform=src.transform)
    col0 = max(math.floor(window.col_off) - WINDOW_PAD_PIXELS, 0)
    row0 = max(math.floor(window.row_off) - WINDOW_PAD_PIXELS, 0)
    col1 = min(math.ceil(window.col_off + window.width) + WINDOW_PAD_PIXELS, src.width)
    row1 = min(math.ceil(window.row_off + window.height) + WINDOW_PAD_PIXELS, src.height)
    if col0 >= col1 or row0 >= row1:
        raise ValueError(f"{src.name} does not cover the AOI")
    return Window(col0, row0, col1 - col0, row1 - row0)


def cached_download(
    url: str,
    cache_dir: str | Path,
    *,
    timeout: float = 60,
    retries: int = 3,
    retry_delay: float = 2.0,
    filename: str | None = None,
) -> Path:
    """Return the cached copy of ``url``, downloading it first if needed.

    ``filename`` names the cached file; by default it is derived from the URL.
    Client errors such as 404 fail at once; server and network errors retry.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / (filename or _cache_name(url))
    if target.is_file() and target.stat().st_size > 0:
        return target

    partial = target.with_name(target.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _stream_to(url, partial, timeout)
            partial.replace(target)
            return target
        except requests.HTTPError as exc:
            partial.unlink(missing_ok=True)
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 400 <= status < 500 and status != 429:
                raise RuntimeError(f"Download failed: {url}: {exc}") from None
            last_error = exc
        except (requests.RequestException, OSError) as exc:
            partial.unlink(missing_ok=True)
            last_error = exc
        if attempt < retries:
            time.sleep(retry_delay * attempt)
    raise RuntimeError(f"Download failed after {retries} attempts: {url}: {last_error}")


def _stream_to(url: str, destination: Path, timeout: float) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        expected = response.headers.get("Content-Length")
        encoded = response.headers.get("Content-Encoding", "identity") not in ("identity", "")
        written = 0
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
    if written == 0:
        raise OSError("empty download")
    if expected is not None and not encoded and written != int(expected):
        raise OSError(f"incomplete download: {written} of {expected} bytes")


def extract_tifs(zip_path: str | Path, dest_dir: str | Path) -> list[Path]:
    """Extract a zip once and list its GeoTIFFs; a marker records a complete extraction."""
    zip_path, dest = Path(zip_path), Path(dest_dir)
    if not (dest / _EXTRACTED_MARKER).is_file():
        staging = dest.with_name(dest.name + ".part")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(staging)
        (staging / _EXTRACTED_MARKER).write_text(str(zip_path))
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(staging, dest)

    tifs = sorted(p for p in dest.rglob("*") if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
    if not tifs:
        raise FileNotFoundError(f"No .tif found inside {zip_path}")
    return tifs


def extract_first_tif(zip_path: str | Path, dest_dir: str | Path) -> Path:
    return extract_tifs(zip_path, dest_dir)[0]


def extract_tif_by_suffix(zip_path: str | Path, dest_dir: str | Path, suffix: str) -> Path:
    """BAER bundles hold several GeoTIFFs; pick the product by its filename suffix."""
    tifs = extract_tifs(zip_path, dest_dir)
    matches = [p for p in tifs if p.name.lower().endswith(suffix.lower())]
    if not matches:
        raise FileNotFoundError(
            f"No .tif ending in {suffix!r} inside {zip_path}. Found: {[p.name for p in tifs]}"
        )
    return matches[0]
