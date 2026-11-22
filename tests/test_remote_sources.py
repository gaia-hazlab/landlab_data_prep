"""The remote sources the config template names still answer, and read in place for a small AOI.

They change without notice: the ScienceBase K-factor link broke in 2026. These
tests reach the real servers, so they run on their own (python -m pytest -m
network) and weekly in CI, not with the unit tests.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from landlab_data_prep.analysis_grid import NODATA, align_to_grid, snap_grid
from landlab_data_prep.downloads import cached_window
from landlab_data_prep.init_config import template_text

pytestmark = pytest.mark.network

TEMPLATE = yaml.safe_load(template_text())
SOURCES = {
    f"{group}.{key}": info
    for group in ("rasters", "landcover")
    for key, info in TEMPLATE["feature_sources"][group].items()
}
# Around the Hazard Creek basin, 2024 Pioneer Fire, Washington: the example event.
GRID = snap_grid((674247.0, 5352375.0, 678188.0, 5353994.0), "EPSG:6339", 10.0)


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_template_source_reads_in_place_for_a_small_aoi(name: str, tmp_path: Path) -> None:
    info = SOURCES[name]
    window = cached_window(info["url"], tmp_path, GRID)
    assert window.stat().st_size < 5_000_000  # megabytes, not the whole national file
    aligned = align_to_grid(
        window, tmp_path / "aligned.tif", GRID, info["resampling"],
        missing_values=tuple(info.get("missing_values", ())),
    )
    with rasterio.open(aligned) as src:
        assert np.mean(src.read(1) != NODATA) > 0.5


def test_burn_severity_of_the_example_fire_is_on_baer() -> None:
    remote = TEMPLATE["burn_severity"]["remote"]
    urls = [
        pattern.format(base_url=remote["base_url"], fire_name_fmt="pioneer", fire_id_fmt="wa4818312053120240608")
        for pattern in remote["candidates"]
    ]
    status = {url: requests.head(url, timeout=60, allow_redirects=True).status_code for url in urls}
    assert 200 in status.values(), status
