#!/usr/bin/env python3
"""
Step 5: Compute ZIP/POA-level and cluster-level risk scores from crash data.

Outputs:
- data/output/zip_risk_table.csv
- data/output/cluster_risk_table.csv

Run from project root:
    python step5_compute_risk_scores.py
"""
from pathlib import Path
import logging
import sys
from typing import Tuple

try:
    import geopandas as gpd
    import pandas as pd
except Exception as e:  # pragma: no cover
    print("This script requires geopandas and pandas. Install with: pip install geopandas pandas", file=sys.stderr)
    raise


# Crash severity column names expected in the input datasets
KILLED_COL = "No. killed"
SERIOUS_COL = "No. seriously injured"
MODERATE_COL = "No. moderately injured"
MINOR_COL = "No. minor-other injured"


def project_root() -> Path:
    """Return the project root (directory containing this script)."""
    return Path(__file__).resolve().parent


def get_paths() -> Tuple[Path, Path, Path, Path]:
    """Compute input and output paths relative to the project root."""
    root = project_root()
    crashes_with_poa = root / "data" / "output" / "crashes_with_postcodes.geojson"
    crashes_with_clusters = root / "data" / "output" / "crashes_with_clusters.geojson"
    out_zip_csv = root / "data" / "output" / "zip_risk_table.csv"
    out_cluster_csv = root / "data" / "output" / "cluster_risk_table.csv"
    return crashes_with_poa, crashes_with_clusters, out_zip_csv, out_cluster_csv


def load_gdf(path: Path, label: str) -> gpd.GeoDataFrame:
    """Load a GeoJSON file with logging and basic checks."""
    logging.info("Loading crash data... (%s)", label)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Input dataset is empty: {path}")
    return gdf


def ensure_columns_exist(df: pd.DataFrame, cols: list[str]) -> None:
    """Ensure required columns exist in the DataFrame."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")


def compute_weighted_score(df: pd.DataFrame) -> pd.Series:
    """Compute weighted severity score per row given component columns already aggregated."""
    numerator = 3 * df["killed"] + 2 * df["serious"] + 1 * df["moderate"] + 0.5 * df["minor"]
    denom = df["total_crashes"].replace(0, pd.NA)
    score = numerator / denom
    return score.fillna(0.0)


def compute_zip_risk(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute risk metrics grouped by POA_CODE21."""
    logging.info("Computing ZIP risk...")
    ensure_columns_exist(
        gdf,
        [
            "POA_CODE21",
            KILLED_COL,
            SERIOUS_COL,
            MODERATE_COL,
            MINOR_COL,
        ],
    )
    grp = gdf.groupby("POA_CODE21", dropna=True)
    size = grp.size().rename("total_crashes")
    sums = grp[[KILLED_COL, SERIOUS_COL, MODERATE_COL, MINOR_COL]].sum(min_count=1)
    sums = sums.rename(
        columns={
            KILLED_COL: "killed",
            SERIOUS_COL: "serious",
            MODERATE_COL: "moderate",
            MINOR_COL: "minor",
        }
    )
    out = pd.concat([size, sums], axis=1).reset_index()
    out["weighted_score"] = compute_weighted_score(out)
    out = out[["POA_CODE21", "total_crashes", "killed", "serious", "moderate", "minor", "weighted_score"]]
    return out


def compute_cluster_risk(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute risk metrics grouped by cluster_id (excluding noise = -1) and append centroids."""
    logging.info("Computing cluster risk...")
    ensure_columns_exist(
        gdf,
        [
            "cluster_id",
            KILLED_COL,
            SERIOUS_COL,
            MODERATE_COL,
            MINOR_COL,
            "geometry",
        ],
    )
    # Exclude noise clusters (-1)
    gdf_valid = gdf[(gdf["cluster_id"].notna()) & (gdf["cluster_id"] != -1)].copy()
    if gdf_valid.empty:
        # Return empty table with expected columns
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "total_crashes",
                "killed",
                "serious",
                "moderate",
                "minor",
                "weighted_score",
                "centroid_lon",
                "centroid_lat",
            ]
        )

    grp = gdf_valid.groupby("cluster_id")
    size = grp.size().rename("total_crashes")
    sums = grp[[KILLED_COL, SERIOUS_COL, MODERATE_COL, MINOR_COL]].sum(min_count=1)
    sums = sums.rename(
        columns={
            KILLED_COL: "killed",
            SERIOUS_COL: "serious",
            MODERATE_COL: "moderate",
            MINOR_COL: "minor",
        }
    )
    out = pd.concat([size, sums], axis=1)

    # Compute centroid per cluster using the union of point geometries
    # (cluster centroid in EPSG:4326, since inputs come from GeoJSON in WGS84)
    centroids = grp["geometry"].apply(lambda s: s.unary_union.centroid if not s.empty else None)
    out["centroid_lon"] = centroids.apply(lambda g: g.x if g is not None and not g.is_empty else pd.NA)
    out["centroid_lat"] = centroids.apply(lambda g: g.y if g is not None and not g.is_empty else pd.NA)

    out = out.reset_index()
    out["weighted_score"] = compute_weighted_score(out)
    out = out[
        [
            "cluster_id",
            "total_crashes",
            "killed",
            "serious",
            "moderate",
            "minor",
            "weighted_score",
            "centroid_lon",
            "centroid_lat",
        ]
    ]
    return out


def save_tables(zip_df: pd.DataFrame, cluster_df: pd.DataFrame, out_zip: Path, out_cluster: Path) -> None:
    """Save risk tables to CSV files."""
    logging.info("Saving tables...")
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    zip_df.to_csv(out_zip, index=False)
    cluster_df.to_csv(out_cluster, index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        crashes_with_poa, crashes_with_clusters, out_zip_csv, out_cluster_csv = get_paths()

        # Load inputs
        gdf_poa = load_gdf(crashes_with_poa, label="crashes_with_postcodes.geojson")
        gdf_clusters = load_gdf(crashes_with_clusters, label="crashes_with_clusters.geojson")

        # Compute risk tables
        zip_df = compute_zip_risk(gdf_poa)
        cluster_df = compute_cluster_risk(gdf_clusters)

        # Save
        save_tables(zip_df, cluster_df, out_zip_csv, out_cluster_csv)

        logging.info("Risk score tables created successfully.")
        print(f"Completed. ZIP risk -> {out_zip_csv}\nCompleted. Cluster risk -> {out_cluster_csv}")
    except Exception as exc:
        logging.exception("Error while computing risk scores: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
