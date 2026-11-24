from __future__ import annotations

import argparse
import logging
from pathlib import Path

from landlab_data_prep.analysis_grid import grid_from_config
from landlab_data_prep.export_ascii_to_tif import export_ascii_dir_to_tifs
from landlab_data_prep.config import load_config
from landlab_data_prep.pipeline import run_landlab_pipeline, run_raster_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run debris-flow raster/landlab pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. config/base.yaml).",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep downloads and aligned intermediates (_downloads/, _aligned/).",
    )
    parser.add_argument(
        "--raster-only",
        action="store_true",
        help="Only run raster processing; skip Landlab feature generation.",
    )
    parser.add_argument(
        "--export-final-tifs",
        action="store_true",
        help="Also write a GeoTIFF next to every ASCII output.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING).",
    )
    return parser.parse_args()


def _resolve_output_crs(cfg: dict) -> str:
    # Exported GeoTIFFs sit on the analysis grid, so they carry its CRS.
    return grid_from_config(cfg).crs


def main() -> None:
    args = _parse_args()
    debug = args.log_level.upper() == "DEBUG"
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    if not debug:
        logging.getLogger("rasterio").setLevel(logging.WARNING)  # e.g. "boto3 not available": harmless, but alarming

    try:
        cfg = load_config(args.config, "pipeline")
        skipped = _run(cfg, args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        if debug:
            raise
        # Expected problems (a bad path, a missing key, a failed download) read best as one message.
        raise SystemExit(f"{exc}\n(Rerun with --log-level DEBUG for the full traceback.)") from None
    print(f"Done. Layers are in {cfg['paths']['output_dir']}")
    if skipped:
        print(f"Skipped {len(skipped)} layer(s) that could not be prepared, and the fields computed from them:")
        for key, reason in skipped.items():
            print(f"  - {key}: {reason}")


def _run(cfg: dict, args: argparse.Namespace) -> dict[str, str]:
    """Build the layers; returns the sources that could not be prepared (key -> reason)."""
    skipped: dict[str, str] = {}
    # A layer that cannot be prepared (e.g. a fire with no BAER map) is left out; the rest is built.
    outputs = run_raster_pipeline(
        cfg,
        cleanup_intermediates=not args.keep_intermediates,
        skipped=skipped,
    )
    if not args.raster_only:
        run_landlab_pipeline(cfg, outputs, strict=not skipped)
    if args.export_final_tifs:
        export_ascii_dir_to_tifs(
            Path(cfg["paths"]["output_dir"]),
            overwrite=True,
            crs=_resolve_output_crs(cfg),
        )
    return skipped


if __name__ == "__main__":
    main()
