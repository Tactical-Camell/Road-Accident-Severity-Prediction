#!/usr/bin/env python3
"""
Step 4: Cluster crash hotspots with DBSCAN in a projected CRS.

- Loads crash points (GeoJSON, expected EPSG:4326)
- Reprojects to EPSG:3857 (meters)
- Runs DBSCAN (eps meters, Euclidean distance)
- Writes cluster labels back, reprojects to EPSG:4326 and saves

Run from project root:
    python step4_cluster_hotspots.py
"""
from pathlib import Path
import logging
import sys
from typing import Tuple

try:
    import numpy as np
    import geopandas as gpd
    from sklearn.cluster import DBSCAN
except Exception as e:  # pragma: no cover
    print(
        "This script requires numpy, geopandas, and scikit-learn.\n"
        "Install with: pip install numpy geopandas scikit-learn",
        file=sys.stderr,
    )
    raise


def project_root() -> Path:
    """Return the project root (directory containing this script)."""
    return Path(__file__).resolve().parent


def get_paths() -> Tuple[Path, Path]:
    """Compute input/output paths relative to the project root."""
    root = project_root()
    crashes_path = root / "data" / "output" / "crashes_with_postcodes.geojson"
    output_path = root / "data" / "output" / "crashes_with_clusters.geojson"
    return crashes_path, output_path


def load_crashes(path: Path) -> gpd.GeoDataFrame:
    """Load crash points GeoJSON."""
    logging.info("Loading crash points...")
    if not path.exists():
        raise FileNotFoundError(f"Crash points not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Input crash dataset is empty.")
    if "geometry" not in gdf.columns:
        raise ValueError("Input dataset has no geometry column.")
    return gdf


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure input CRS is EPSG:4326 (WGS84)."""
    if gdf.crs is None:
        logging.warning("Input CRS missing; assuming EPSG:4326")
        return gdf.set_crs(epsg=4326, allow_override=True)
    try:
        epsg = gdf.crs.to_epsg() if hasattr(gdf.crs, "to_epsg") else None
    except Exception:
        epsg = None
    if epsg != 4326:
        logging.info("Reprojecting to EPSG:4326 from %s", epsg)
        return gdf.to_crs(epsg=4326)
    return gdf


def to_meters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject to EPSG:3857 (meters) for Euclidean DBSCAN."""
    logging.info("Reprojecting to meters...")
    return gdf.to_crs(epsg=3857)


def extract_coords(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Extract X,Y coordinates as an (N, 2) NumPy array from geometry."""
    # Drop rows with missing/empty geometry
    mask = gdf.geometry.notna() & ~gdf.geometry.is_empty
    if not mask.all():
        dropped = int((~mask).sum())
        if dropped:
            logging.warning("Dropping %d rows with missing/empty geometry before clustering", dropped)
        gdf = gdf.loc[mask].copy()
    coords = np.vstack([gdf.geometry.x, gdf.geometry.y]).T
    return gdf, coords


def run_dbscan(coords: np.ndarray, eps_m: float = 500.0, min_samples: int = 10) -> np.ndarray:
    """Run DBSCAN clustering with Euclidean metric in meters."""
    logging.info("Running DBSCAN...")
    if coords.size == 0:
        return np.array([], dtype=int)
    db = DBSCAN(eps=eps_m, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(coords)
    return labels.astype(int, copy=False)


def attach_labels(gdf_meters: gpd.GeoDataFrame, labels: np.ndarray) -> gpd.GeoDataFrame:
    """Attach cluster labels to the GeoDataFrame as 'cluster_id'."""
    gdf_out = gdf_meters.copy()
    if len(labels) != len(gdf_out):
        # This should not happen if we handled geometry filtering consistently
        raise ValueError("Label count does not match number of geometries after preprocessing.")
    gdf_out["cluster_id"] = labels
    return gdf_out


def diagnostics(gdf_labeled: gpd.GeoDataFrame) -> None:
    """Print clustering diagnostics: cluster count, noise points, cluster sizes."""
    if "cluster_id" not in gdf_labeled.columns:
        logging.info("No cluster labels found for diagnostics.")
        return
    labels = gdf_labeled["cluster_id"]
    noise = int((labels == -1).sum())
    clusters = labels[labels != -1]
    n_clusters = int(clusters.nunique())
    logging.info("Clusters detected (excluding noise): %d", n_clusters)
    logging.info("Noise points: %d", noise)
    if n_clusters > 0:
        size_series = clusters.value_counts().sort_values(ascending=False)
        logging.info("Cluster sizes (cluster_id -> count):\n%s", size_series.to_string())


def save_geojson(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """Save clustered output to GeoJSON."""
    logging.info("Saving clustered output...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")


def main() -> None:
    """Entrypoint for DBSCAN hotspot clustering."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        crashes_path, output_path = get_paths()
        gdf = load_crashes(crashes_path)
        gdf = ensure_wgs84(gdf)

        # Project to meters for DBSCAN
        gdf_m = to_meters(gdf)
        gdf_m_filtered, coords = extract_coords(gdf_m)

        # Cluster
        labels = run_dbscan(coords, eps_m=500.0, min_samples=10)
        gdf_labeled_m = attach_labels(gdf_m_filtered, labels)

        # Return to WGS84 for output
        gdf_labeled = gdf_labeled_m.to_crs(epsg=4326)

        # Diagnostics
        diagnostics(gdf_labeled)

        # Save
        save_geojson(gdf_labeled, output_path)
        logging.info("Completed successfully. Output written to: %s", output_path)
    except Exception as exc:
        logging.exception("Error during clustering: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
