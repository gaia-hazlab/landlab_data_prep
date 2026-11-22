from __future__ import annotations

from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from landlab_data_prep import downloads
from landlab_data_prep.analysis_grid import align_to_grid, snap_grid
from landlab_data_prep.downloads import (
    cached_download,
    cached_window,
    extract_first_tif,
    extract_tif_by_suffix,
    extract_tifs,
    fetch_raster,
)


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200, length: int | None = None) -> None:
        self.body = body
        self.status_code = status
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self.body), 4):
            yield self.body[i : i + 4]


def test_download_is_cached_and_reused(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_get(url, stream, timeout):
        calls.append(url)
        return FakeResponse(b"GeoTIFF-bytes", length=13)

    monkeypatch.setattr(downloads.requests, "get", fake_get)
    first = cached_download("https://example.org/data/soil.tif", tmp_path)
    second = cached_download("https://example.org/data/soil.tif", tmp_path)
    assert first == second and first.read_bytes() == b"GeoTIFF-bytes"
    assert first.name.endswith("_soil.tif")
    assert len(calls) == 1
    assert not list(tmp_path.glob("*.part"))


def test_download_can_be_given_a_file_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(downloads.requests, "get", lambda url, stream, timeout: FakeResponse(b"zipbytes", length=8))
    path = cached_download("https://example.org/get/us/800m/ppt/20251207", tmp_path, filename="prism_ppt.zip")
    assert path == tmp_path / "prism_ppt.zip" and path.read_bytes() == b"zipbytes"


def test_truncated_download_leaves_nothing_behind(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(downloads.requests, "get", lambda url, stream, timeout: FakeResponse(b"short", length=100))
    monkeypatch.setattr(downloads.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="incomplete download"):
        cached_download("https://example.org/big.zip", tmp_path, retries=2)
    assert list(tmp_path.iterdir()) == []


def test_client_errors_fail_fast_without_retrying(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_get(url, stream, timeout):
        calls.append(url)
        return FakeResponse(b"", status=404)

    monkeypatch.setattr(downloads.requests, "get", fake_get)
    with pytest.raises(RuntimeError, match="404"):
        cached_download("https://example.org/missing_sbs.zip", tmp_path, retries=3)
    assert len(calls) == 1


def test_zip_is_extracted_once(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner/b_dnbr.tif", b"x")
        zf.writestr("inner/a_sbs.tif", b"y")
        zf.writestr("readme.txt", b"z")
    dest = tmp_path / "bundle_extracted"

    assert [p.name for p in extract_tifs(archive, dest)] == ["a_sbs.tif", "b_dnbr.tif"]
    monkeypatch.setattr(downloads.zipfile, "ZipFile", lambda *a, **k: pytest.fail("extracted twice"))
    assert extract_first_tif(archive, dest).name == "a_sbs.tif"
    assert extract_tif_by_suffix(archive, dest, "_dnbr.tif").name == "b_dnbr.tif"
    with pytest.raises(FileNotFoundError, match="_missing.tif"):
        extract_tif_by_suffix(archive, dest, "_missing.tif")


# ---------------------------------------------------------------- remote GeoTIFFs read in place

# A 4 km by 2 km area in the 2024 Pioneer Fire, Washington, on a UTM grid.
AOI_BOUNDS = (674247.0, 5352375.0, 678188.0, 5353994.0)
AOI_CRS = "EPSG:6339"


def _albers_source(path: Path, pixel: float = 100.0) -> Path:
    """A national-style layer: Albers, coarse pixels, a smooth field with nodata holes, around the AOI."""
    west, south, east, north = transform_bounds(AOI_CRS, "EPSG:5070", *AOI_BOUNDS)
    left = (west // pixel - 60) * pixel
    top = (north // pixel + 60) * pixel
    width = int((east - west) // pixel) + 120
    height = int((north - south) // pixel) + 120
    rows, cols = np.mgrid[0:height, 0:width]
    data = (50 + 20 * np.sin(rows / 7.0) * np.cos(cols / 5.0) + rows * 0.1).astype("float32")
    data[(rows * 7 + cols * 3) % 23 == 0] = -9999.0
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1, dtype="float32",
        crs="EPSG:5070", transform=from_origin(left, top, pixel, pixel), nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


def test_window_keeps_the_source_pixels_around_the_grid(tmp_path: Path) -> None:
    source = _albers_source(tmp_path / "national.tif")
    grid = snap_grid(AOI_BOUNDS, AOI_CRS, 10.0)
    window = cached_window(str(source), tmp_path / "cache", grid)

    with rasterio.open(source) as full, rasterio.open(window) as part:
        assert part.crs == full.crs and part.res == full.res and part.nodata == full.nodata
        col = (part.transform.c - full.transform.c) / full.transform.a
        row = (part.transform.f - full.transform.f) / full.transform.e
        assert (col, row) == (round(col), round(row))  # same pixel phase as the source
        expected = full.read(1, window=rasterio.windows.Window(round(col), round(row), part.width, part.height))
        assert np.array_equal(part.read(1), expected)
        assert part.width * part.height < full.width * full.height / 4  # a small part of the file
        grid_in_source = transform_bounds(grid.crs, part.crs, *grid.bounds)
        assert part.bounds.left < grid_in_source[0] and part.bounds.right > grid_in_source[2]
        assert part.bounds.bottom < grid_in_source[1] and part.bounds.top > grid_in_source[3]


@pytest.mark.parametrize("resampling", ["nearest", "bilinear", "cubic", "average", "mode"])
@pytest.mark.parametrize("target_res", [10.0, 250.0])  # finer and coarser than the 100 m source
def test_a_warp_from_the_window_equals_a_warp_from_the_whole_file(tmp_path: Path, resampling, target_res) -> None:
    source = _albers_source(tmp_path / "national.tif")
    grid = snap_grid(AOI_BOUNDS, AOI_CRS, target_res)
    window = cached_window(str(source), tmp_path / "cache", grid)

    from_full = align_to_grid(source, tmp_path / "full.tif", grid, resampling)
    from_window = align_to_grid(window, tmp_path / "window.tif", grid, resampling)
    with rasterio.open(from_full) as a, rasterio.open(from_window) as b:
        full, part = a.read(1), b.read(1)
    # A window too small would change edge cells or leave them empty. The window's
    # origin is a different float than the file's, which can move a value by one
    # float32 rounding step, nothing more.
    assert np.array_equal(full == -9999.0, part == -9999.0)
    np.testing.assert_allclose(part, full, rtol=1e-6, atol=0)


def test_window_is_cached_and_reused_without_the_source(tmp_path: Path) -> None:
    source = _albers_source(tmp_path / "national.tif")
    grid = snap_grid(AOI_BOUNDS, AOI_CRS, 10.0)
    first = cached_window(str(source), tmp_path / "cache", grid)
    source.unlink()
    assert cached_window(str(source), tmp_path / "cache", grid) == first
    assert not list((tmp_path / "cache").rglob("*.part"))
    other = snap_grid(AOI_BOUNDS, AOI_CRS, 30.0)
    with pytest.raises(RasterioIOError):
        cached_window(str(source), tmp_path / "cache", other)  # another grid is another window


def test_a_grid_outside_the_source_is_an_error(tmp_path: Path) -> None:
    source = _albers_source(tmp_path / "national.tif")
    far_away = snap_grid((500000.0, 4100000.0, 501000.0, 4101000.0), "EPSG:32611", 10.0)
    with pytest.raises(ValueError, match="does not cover the AOI"):
        cached_window(str(source), tmp_path / "cache", far_away)
    assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").rglob("*.tif"))


def test_a_file_that_cannot_be_read_in_place_is_downloaded_whole(tmp_path: Path, monkeypatch) -> None:
    grid = snap_grid(AOI_BOUNDS, AOI_CRS, 10.0)
    whole = tmp_path / "whole.tif"
    downloaded = []

    def no_ranges(url, cache_dir, grid):
        raise RasterioIOError("Range downloading not supported by this server")

    def download(url, cache_dir):
        downloaded.append(url)
        return whole

    monkeypatch.setattr(downloads, "cached_window", no_ranges)
    monkeypatch.setattr(downloads, "cached_download", download)
    assert fetch_raster("https://example.org/layer.tif", tmp_path, grid) == whole
    assert fetch_raster("https://example.org/layer.tif", tmp_path) == whole  # no grid: whole file
    assert downloaded == ["https://example.org/layer.tif"] * 2
