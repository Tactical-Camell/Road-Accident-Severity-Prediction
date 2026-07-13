#!/usr/bin/env python3
"""
Step 3: Spatially join crash points to NSW postcode polygons.

- Loads crash points and NSW postcode polygons (both GeoJSON, ideally EPSG:4326)
- Prints diagnostics for both layers
- Ensures both are in the same CRS (EPSG:4326)
- Performs a spatial join with predicate="within"
- Preserves all crash columns and adds POA_CODE21 and POA_NAME21
- Saves the joined layer to data/output/crashes_with_postcodes.geojson

Run from project root:
    python step3_spatial_join_postcodes.py
"""
from pathlib import Path
import logging
import sys
from typing import Tuple

try:
    import geopandas as gpd
    import pandas as pd
except Exception as e:  # pragma: no cover
    print("geopandas and pandas are required. Install with: pip install geopandas pandas", file=sys.stderr)
    raise


def project_root() -> Path:
    """Return the project root (directory containing this script)."""
    return Path(__file__).resolve().parent


def get_paths() -> Tuple[Path, Path, Path]:
    """Compute input/output paths relative to the project root."""
    root = project_root()
    shapes_path = root / "data" / "output" / "nsw_postcodes.geojson"
    crashes_path = root / "data" / "output" / "crashes_with_points.geojson"
    output_path = root / "data" / "output" / "crashes_with_postcodes.geojson"
    return shapes_path, crashes_path, output_path


def load_geojson(path: Path, what: str) -> gpd.GeoDataFrame:
    """Load a GeoJSON file into a GeoDataFrame with a log message."""
    logging.info("Loading %s...", what)
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return gpd.read_file(path)


def print_diagnostics(crashes: gpd.GeoDataFrame, poa: gpd.GeoDataFrame) -> None:
    """Print counts, CRS, and head for both layers."""
    logging.info("Number of crashes: %d", len(crashes))
    logging.info("Number of postcode polygons: %d", len(poa))
    logging.info("CRS (crashes): %s", crashes.crs)
    logging.info("CRS (postcodes): %s", poa.crs)
    logging.info("Crashes head:\n%s", crashes.head().to_string())
    logging.info("Postcodes head:\n%s", poa.head().to_string())


def ensure_wgs84(gdf: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
    """Ensure GeoDataFrame is EPSG:4326, converting if necessary.

    If CRS is missing, assume EPSG:4326 to proceed (both upstream steps should already output WGS84).
    """
    if gdf.crs is None:
        logging.warning("%s CRS missing; assuming EPSG:4326", layer_name)
        return gdf.set_crs(epsg=4326, allow_override=True)
    try:
        epsg = gdf.crs.to_epsg() if hasattr(gdf.crs, "to_epsg") else None
    except Exception:
        epsg = None
    if epsg != 4326:
        logging.info("Reprojecting %s to EPSG:4326 from %s", layer_name, epsg)
        return gdf.to_crs(epsg=4326)
    return gdf


def validate_poa_columns(poa: gpd.GeoDataFrame) -> None:
    """Validate required POA columns exist."""
    required = {"POA_CODE21", "POA_NAME21"}
    missing = [c for c in required if c not in poa.columns]
    if missing:
        raise KeyError(f"Missing expected POA columns: {missing}. Available: {list(poa.columns)}")


def normalize_join_columns(joined: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalize POA column names after sjoin in case of suffixing."""
    rename_map = {}
    for col in ["POA_CODE21", "POA_NAME21"]:
        if col in joined.columns:
            continue
        # look for possible suffixed variants
        candidates = [f"{col}_right", f"{col}_poa", f"right_{col}"]
        for cand in candidates:
            if cand in joined.columns:
                rename_map[cand] = col
                break
    if rename_map:
        joined = joined.rename(columns=rename_map)
    return joined


def spatial_join_crashes_to_poa(crashes: gpd.GeoDataFrame, poa: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Join crash points to POA polygons using predicate='within' (left join).

    Preserves all crash columns and adds POA_CODE21 and POA_NAME21. Drops other POA attributes.
    """
    logging.info("Performing spatial join...")
    joined = gpd.sjoin(crashes, poa, how="left", predicate="within")
    joined = normalize_join_columns(joined)

    # Keep all crash columns + POA columns
    keep_cols = list(crashes.columns)
    # ensure geometry is last to avoid writing issues in some contexts
    if "geometry" in keep_cols:
        keep_cols.remove("geometry")
    keep_cols += ["POA_CODE21", "POA_NAME21", "geometry"]
    keep_cols = [c for c in keep_cols if c in joined.columns]
    joined = joined[keep_cols]
    return joined


def save_geojson(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """Save GeoDataFrame to GeoJSON."""
    logging.info("Saving output...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")


def main() -> None:
    """Entrypoint for spatial join step."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        shapes_path, crashes_path, output_path = get_paths()

        # Load inputs
        crashes = load_geojson(crashes_path, what="crash points")
        poa = load_geojson(shapes_path, what="NSW postcode polygons")

        # Diagnostics
        print_diagnostics(crashes, poa)

        # Ensure matching CRS
        logging.info("Ensuring matching CRS...")
        crashes = ensure_wgs84(crashes, layer_name="crashes")
        poa = ensure_wgs84(poa, layer_name="postcodes")

        # Confirm required POA columns
        validate_poa_columns(poa)

        # Spatial join
        joined = spatial_join_crashes_to_poa(crashes, poa)

        # Post-join diagnostics
        total = len(crashes)
        matched = int(joined["POA_CODE21"].notna().sum()) if "POA_CODE21" in joined.columns else 0
        unmatched = total - matched
        logging.info("Number of matched crashes: %d", matched)
        if unmatched:
            logging.info("Number of unmatched crashes: %d", unmatched)
        # Unique code counts summary (top 10)
        if "POA_CODE21" in joined.columns:
            vc = joined["POA_CODE21"].value_counts(dropna=True)
            logging.info("Unique POA_CODE21 count: %d", int(vc.index.nunique()))
            logging.info("Top POA_CODE21 frequency:\n%s", vc.head(10).to_string())

        # Save
        save_geojson(joined, output_path)
        logging.info("Completed successfully. Output written to: %s", output_path)
    except Exception as exc:
        logging.exception("Error during spatial join: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
