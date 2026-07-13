#!/usr/bin/env python3
"""
Prepare NSW postal areas from the ABS 2021 POA shapefile.

- Loads shapefile from data/shapes/ using pathlib-relative paths
- Prints diagnostics (count, columns, head, CRS)
- Filters NSW POAs where POA_CODE21 starts with "2"
- Converts CRS from EPSG:7844 (GDA2020) to EPSG:4326 (WGS84)
- Saves to data/output/nsw_postcodes.geojson

Run from project root:
    python step1_prepare_postcodes.py
"""
from pathlib import Path
import logging
from typing import Tuple
import sys

try:
    import geopandas as gpd
except Exception as e:  # pragma: no cover
    print("geopandas is required. Install with: pip install geopandas pyproj shapely fiona", file=sys.stderr)
    raise


def project_root() -> Path:
    """Return the project root (directory containing this script)."""
    return Path(__file__).resolve().parent


def get_paths() -> Tuple[Path, Path]:
    """Compute input/output paths relative to the project root."""
    root = project_root()
    poa_shp = root / "data" / "shapes" / "POA_2021_AUST_GDA2020.shp"
    out_geojson = root / "data" / "output" / "nsw_postcodes.geojson"
    return poa_shp, out_geojson


def load_shapefile(poa_path: Path) -> gpd.GeoDataFrame:
    """Load the ABS POA shapefile.

    Raises FileNotFoundError if not present.
    """
    logging.info("Loading shapefile...")
    if not poa_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {poa_path}")
    gdf = gpd.read_file(poa_path)
    return gdf


def print_diagnostics(gdf: gpd.GeoDataFrame) -> None:
    """Print basic diagnostics about the GeoDataFrame."""
    logging.info("Number of polygons: %d", len(gdf))
    logging.info("Columns: %s", list(gdf.columns))
    # Show first 5 rows for a quick preview
    logging.info("Head:\n%s", gdf.head().to_string())
    logging.info("CRS information: %s", gdf.crs)


def filter_nsw(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only rows where POA_CODE21 starts with '2' (NSW)."""
    logging.info("Filtering NSW postal areas...")
    if "POA_CODE21" not in gdf.columns:
        raise KeyError("Expected column 'POA_CODE21' not found in shapefile attributes")
    filtered = gdf[gdf["POA_CODE21"].astype(str).str.startswith("2", na=False)].copy()
    logging.info("Kept %d NSW postal areas", len(filtered))
    return filtered


def convert_to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert CRS to EPSG:4326 (WGS84). Assumes input is GDA2020 (EPSG:7844) if unspecified."""
    logging.info("Converting CRS...")
    # If CRS is missing, assume EPSG:7844 as indicated by the dataset name
    if gdf.crs is None:
        logging.warning("Input CRS missing; assuming EPSG:7844 (GDA2020)")
        gdf = gdf.set_crs(epsg=7844, allow_override=True)
    else:
        try:
            epsg = gdf.crs.to_epsg() if hasattr(gdf.crs, "to_epsg") else None
            if epsg != 7844:
                logging.warning("Input CRS EPSG is %s; expected EPSG:7844 (GDA2020). Proceeding with transform anyway.", epsg)
        except Exception:  # pragma: no cover
            logging.debug("Could not resolve EPSG code for input CRS")
    return gdf.to_crs(epsg=4326)


def save_geojson(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """Save the GeoDataFrame to GeoJSON."""
    logging.info("Saving NSW GeoJSON...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")


def main() -> None:
    """Entrypoint for preparing NSW postal areas GeoJSON."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        poa_path, out_geojson = get_paths()
        gdf = load_shapefile(poa_path)
        print_diagnostics(gdf)
        gdf_nsw = filter_nsw(gdf)
        gdf_nsw_wgs84 = convert_to_wgs84(gdf_nsw)
        save_geojson(gdf_nsw_wgs84, out_geojson)
        logging.info("Completed successfully. Output written to: %s", out_geojson)
    except Exception as exc:
        logging.exception("Error while preparing NSW postcodes: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
