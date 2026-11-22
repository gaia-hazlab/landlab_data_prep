from __future__ import annotations

import argparse
from pathlib import Path

from landlab_data_prep.analysis_grid import grid_from_config
from landlab_data_prep.config import ConfigError, load_config
from landlab_data_prep.soil_data.core import (
    fetch_soil_layers,
    parse_soil_keys,
    resolve_aoi_path,
    resolve_output_dir,
    resolve_soil_specs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch SoLUS soil layers and crop them to the config's AOI, without aligning them."
    )
    parser.add_argument(
        "--config",
        default="config/base.yaml",
        help="Config that defines the AOI and analysis grid.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for fetched GeoTIFFs.")
    parser.add_argument("--soil-keys", default=None, help="Comma-separated subset of soil keys.")
    parser.add_argument("--no-clip", action="store_true", help="Download whole files; do not crop to the AOI.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fetched GeoTIFFs.")
    parser.add_argument("--keep-intermediates", action="store_true", help="Keep temporary files.")
    return parser.parse_args()


def _load(config: str) -> dict:
    try:
        return load_config(config, "soil")
    except (ConfigError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None


def _cache_dir(cfg: dict) -> Path | None:
    value = cfg["paths"].get("cache_dir")
    return Path(value) if value else None


def main() -> None:
    args = _parse_args()
    cfg = _load(args.config)
    specs = resolve_soil_specs(parse_soil_keys(args.soil_keys), cfg)
    try:
        manifest_path = fetch_soil_layers(
            aoi_path=resolve_aoi_path(None, cfg),
            output_dir=Path(resolve_output_dir(args.output_dir, cfg)),
            specs=specs,
            cache_dir=_cache_dir(cfg),
            clip_to_aoi=not args.no_clip,
            overwrite=args.overwrite,
            keep_intermediates=args.keep_intermediates,
            grid=grid_from_config(cfg),
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
    print(f"Saved fetch manifest: {manifest_path}")
    print(f"Soil layers fetched: {len(specs)}")


if __name__ == "__main__":
    main()
