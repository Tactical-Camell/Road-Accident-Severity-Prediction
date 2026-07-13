#!/usr/bin/env python3
"""
Template to prepare crash points GeoJSON.

- Loads crash CSV from Dataset/cleaned_dataset.csv
- Converts latitude/longitude to Point geometries
- Assigns CRS and saves GeoJSON to data/output/crashes_with_points.geojson

Run from project root:
    python step2_prepare_crash_points.py
"""
from pathlib import Path
from typing import Tuple
import logging
import sys

try:
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
except Exception as e:  # pragma: no cover
    print("pandas, geopandas, and shapely are required. Install with: pip install pandas geopandas shapely", file=sys.stderr)
    raise


def project_root() -> Path:
    """Return the project root (directory containing this script)."""
    return Path(__file__).resolve().parent


def get_paths() -> Tuple[Path, Path]:
    """Compute input/output paths relative to the project root."""
    root = project_root()
    csv_path = root / "Dataset" / "cleaned_dataset.csv"
    out_geojson = root / "data" / "output" / "crashes_with_points.geojson"
    return csv_path, out_geojson


def load_crash_csv(csv_path: Path) -> pd.DataFrame:
    """Load the crash CSV as a DataFrame from a fixed path."""
    logging.info("Loading crash CSV: %s", csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Crash CSV not found: {csv_path}")
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            logging.info("Loaded CSV with encoding: %s", enc)
            return df
        except UnicodeDecodeError as e:
            logging.warning("Failed reading with encoding %s: %s", enc, e)
        except pd.errors.ParserError as e:
            logging.warning(
                "Parser error with encoding %s using default engine: %s; retrying with python engine and sep auto-detect",
                enc,
                e,
            )
            # Try more permissive python engine with auto-detected separator
            try:
                df = pd.read_csv(csv_path, encoding=enc, engine="python", sep=None)
                logging.info("Loaded CSV with encoding: %s, engine: python (auto-detected separator)", enc)
                return df
            except Exception as e2:
                logging.warning("Python engine auto-detect failed for encoding %s: %s", enc, e2)
                # Try common separators explicitly
                for sep in [",", ";", "\t", "|"]:
                    try:
                        df = pd.read_csv(csv_path, encoding=enc, engine="python", sep=sep)
                        logging.info("Loaded CSV with encoding: %s, engine: python, sep: %r", enc, sep)
                        return df
                    except Exception:
                        continue
    logging.warning("Falling back to utf-8 with replacement and skipping bad lines")
    return pd.read_csv(
        csv_path,
        encoding="utf-8",
        encoding_errors="replace",
        engine="python",
        sep=None,
        on_bad_lines="skip",
    )


def dataframe_to_points(df: pd.DataFrame, lon_col: str = "longitude", lat_col: str = "latitude") -> gpd.GeoDataFrame:
    """Convert DataFrame rows to Point geometry (defaults to 'longitude'/'latitude').

    TODO: If your CSV uses different column names, update 'lon_col' and 'lat_col'.
    """
    logging.info("Converting latitude/longitude to Point geometry...")
    # Fallback to common alternatives if defaults not present
    if lon_col not in df.columns or lat_col not in df.columns:
        lon_alt = ["lon", "lng", "Longitude", "LONGITUDE"]
        lat_alt = ["lat", "Latitude", "LATITUDE"]
        if lon_col not in df.columns:
            lon_col = next((c for c in lon_alt if c in df.columns), None)
        if lat_col not in df.columns:
            lat_col = next((c for c in lat_alt if c in df.columns), None)
        if not lon_col or not lat_col:
            raise KeyError("Expected longitude/latitude columns not found. TODO: adjust 'lon_col' and 'lat_col'.")
        logging.info("Using detected columns lon=%s lat=%s", lon_col, lat_col)
    df = df.copy()
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[lon_col, lat_col])
    dropped = before - len(df)
    if dropped:
        logging.warning("Dropped %d rows with missing/invalid coordinates", dropped)
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure GeoDataFrame is in EPSG:4326. Assign if missing, convert if different."""
    if gdf.crs is None:
        logging.info("Assigning CRS EPSG:4326 to crash points")
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    else:
        try:
            epsg = gdf.crs.to_epsg() if hasattr(gdf.crs, "to_epsg") else None
            if epsg != 4326:
                logging.info("Reprojecting crash points to EPSG:4326 from %s", epsg)
                gdf = gdf.to_crs(epsg=4326)
        except Exception:  # pragma: no cover
            logging.info("Reprojecting crash points to EPSG:4326")
            gdf = gdf.to_crs(epsg=4326)
    return gdf


def save_geojson(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """Save crash points to GeoJSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Saving crash points GeoJSON: %s", out_path)
    gdf.to_file(out_path, driver="GeoJSON")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        csv_path, out_geojson = get_paths()
        df = load_crash_csv(csv_path)
        # TODO: If your CSV uses different names, set lon_col/lat_col explicitly
        gdf_points = dataframe_to_points(df, lon_col="Longitude", lat_col="Latitude")
        gdf_points = ensure_wgs84(gdf_points)
        save_geojson(gdf_points, out_geojson)
        logging.info("Completed successfully. Output written to: %s", out_geojson)
    except Exception as exc:
        logging.exception("Error while preparing crash points: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
