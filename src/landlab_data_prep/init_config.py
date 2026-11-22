"""Write a starting config, so an installed copy needs no clone of this repo.

``--example`` writes a config that runs as it is: a 3.3 km2 basin burned by the
2024 Pioneer Fire, Washington, whose burn severity map USGS BAER publishes
under the fire's name and id.
"""

from __future__ import annotations

import argparse
from importlib import resources
import json
import os
from pathlib import Path
import sys

TEMPLATE_NAME = "base.example.yaml"
EXAMPLE_AOI_NAME = "example_hazard_creek.geojson"
API_KEY_URL = "https://portal.opentopography.org/requestService?service=api"
# Template lines the example fills in; {aoi}, {output} and {cache} are absolute paths.
EXAMPLE_EDITS = {
    "aoi: /abs/path/to/aoi.shp": "aoi: {aoi}",
    "output_dir: /abs/path/to/outputs": "output_dir: {output}",
    "cache_dir: /abs/path/to/cache": "cache_dir: {cache}",
    "name: Example Fire": "name: Pioneer",
    "id: ST0000000000000000000": "id: WA4818312053120240608",
}


def template_text() -> str:
    """The documented config template shipped with the package."""
    return _package_file(TEMPLATE_NAME).read_text(encoding="utf-8")


def example_text(aoi: Path, output_dir: Path, cache_dir: Path) -> str:
    """The template filled in for the example basin."""
    # JSON strings are valid YAML, and quoting keeps any path readable.
    values = {key: json.dumps(path.resolve().as_posix()) for key, path in
              (("aoi", aoi), ("output", output_dir), ("cache", cache_dir))}
    text = template_text()
    for old, new in EXAMPLE_EDITS.items():
        if text.count(old) != 1:
            raise RuntimeError(f"The config template has changed; cannot find {old!r}")
        text = text.replace(old, new.format(**values))
    return text


def _package_file(name: str):
    return resources.files("landlab_data_prep").joinpath(name)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a starting config to edit for your event.")
    parser.add_argument("--output", default="config/base.yaml", help="Where to write it. Default: config/base.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite the file if it exists.")
    parser.add_argument(
        "--example",
        action="store_true",
        help="Write a config that runs as it is, for a small basin of the 2024 Pioneer Fire, Washington.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = Path(args.output)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists. Pass --force to overwrite it.")
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.example:
        # Outputs and downloads go next to the config folder: ./output and ./cache for config/base.yaml.
        project = out.parent.parent if out.parent.name == "config" else out.parent
        aoi = out.parent / EXAMPLE_AOI_NAME
        aoi.write_bytes(_package_file(EXAMPLE_AOI_NAME).read_bytes())
        text = example_text(aoi, project / "output", project / "cache")
    else:
        text = template_text()
    out.write_text(text, encoding="utf-8")
    if os.name == "posix":
        # The config can hold an API key, so keep it readable by its owner only.
        os.chmod(out, 0o600)

    print(out)
    if args.example:
        print(f"Example AOI: {aoi}")
        print(f"Next: set OPENTOPOGRAPHY_API_KEY (a free key from {API_KEY_URL}), then run")
    else:
        print("Next: edit it for your event (AOI, folders, fire name and id), then run")
    print(f"  landlab-prep-pipeline --config {out} --export-final-tifs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
