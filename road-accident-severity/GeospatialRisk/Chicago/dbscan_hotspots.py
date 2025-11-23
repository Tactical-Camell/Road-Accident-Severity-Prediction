from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUSTERS_SOURCE_PATH = PROJECT_ROOT / "data" / "output" / "chicago_crashes_with_areas.geojson"
CLUSTERS_OUTPUT_PATH = PROJECT_ROOT / "data" / "output" / "chicago_crashes_with_clusters.geojson"
CENTROIDS_OUTPUT_PATH = PROJECT_ROOT / "data" / "output" / "chicago_cluster_centroids.geojson"
RISK_TABLE_PATH = PROJECT_ROOT / "data" / "output" / "cluster_risk_table.csv"


def prepare_coords_for_haversine(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Return coordinates in radians ready for haversine DBSCAN."""

    if gdf.empty:
        raise ValueError("GeoDataFrame must contain at least one row")

    if "geometry" not in gdf.columns:
        raise ValueError("GeoDataFrame must include a 'geometry' column")

    coords_deg = np.vstack([gdf.geometry.x, gdf.geometry.y]).T
    coords_rad = np.radians(coords_deg).astype(np.float32, copy=False)
    print(f"Prepared coords with shape {coords_rad.shape}, example (deg): {coords_deg[0]}")
    return coords_rad


def run_dbscan_haversine(coords_rad: np.ndarray, eps_meters: float = 250.0, min_samples: int = 12) -> np.ndarray:
    """Run DBSCAN with the haversine metric and return cluster labels."""

    if coords_rad.ndim != 2 or coords_rad.shape[1] != 2:
        raise ValueError("Coordinate array must be shape (n_samples, 2)")

    eps_radians = eps_meters / 6371000.0
    print(
        "Using eps=250m (~0.25km) for dense urban Chicago hotspots converted to radians",
        f"({eps_radians:.9f}) and min_samples={min_samples} to reduce noise.",
    )
    clusterer = DBSCAN(
        eps=eps_radians,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    labels = clusterer.fit_predict(coords_rad)
    return labels


def compute_cluster_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return GeoDataFrame of centroid points for each labeled cluster."""

    clusters = gdf[gdf["cluster_id"] >= 0].copy()
    if clusters.empty:
        return gpd.GeoDataFrame(
            columns=["cluster_id", "centroid_lon", "centroid_lat", "num_points", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    agg = clusters.groupby("cluster_id").agg(
        centroid_lon=("geometry", lambda pts: pts.x.mean()),
        centroid_lat=("geometry", lambda pts: pts.y.mean()),
        num_points=("geometry", "size"),
    )
    agg = agg.reset_index()
    agg["geometry"] = agg.apply(
        lambda row: Point(row["centroid_lon"], row["centroid_lat"]), axis=1
    )
    centroids = gpd.GeoDataFrame(agg, geometry="geometry", crs="EPSG:4326")
    return centroids


def compute_cluster_severity(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute severity metrics and weighted score for each cluster."""

    cluster_rows = gdf[gdf["cluster_id"] >= 0].copy()
    if cluster_rows.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "total_crashes",
                "sum_fatal",
                "sum_incapacitating",
                "sum_non_incapacitating",
                "sum_reported_not_evident",
                "sum_no_indication",
                "weighted_score",
            ]
        )

    agg = cluster_rows.groupby("cluster_id").agg(
        total_crashes=("cluster_id", "size"),
        sum_fatal=("INJURIES_FATAL", "sum"),
        sum_incapacitating=("INJURIES_INCAPACITATING", "sum"),
        sum_non_incapacitating=("INJURIES_NON_INCAPACITATING", "sum"),
        sum_reported_not_evident=("INJURIES_REPORTED_NOT_EVIDENT", "sum"),
        sum_no_indication=("INJURIES_NO_INDICATION", "sum"),
    )
    agg = agg.reset_index()

    def weighted_score(row: pd.Series) -> float:
        numerator = (
            3 * row["sum_fatal"]
            + 2 * row["sum_incapacitating"]
            + 1 * row["sum_non_incapacitating"]
            + 0.5 * row["sum_reported_not_evident"]
        )
        denom = (
            row["sum_fatal"]
            + row["sum_incapacitating"]
            + row["sum_non_incapacitating"]
            + row["sum_reported_not_evident"]
            + row["sum_no_indication"]
        )
        return numerator / denom if denom > 0 else 0.0

    agg["weighted_score"] = agg.apply(weighted_score, axis=1)
    return agg


def _cluster_subset(subset: gpd.GeoDataFrame, start_cluster_id: int) -> tuple[gpd.GeoDataFrame, int]:
    coords_rad = prepare_coords_for_haversine(subset)
    labels = run_dbscan_haversine(coords_rad)

    subset = subset.copy()
    adjusted_labels = labels.copy()
    mask = adjusted_labels >= 0
    if mask.any():
        adjusted_labels[mask] = adjusted_labels[mask] + start_cluster_id
        next_cluster_id = int(adjusted_labels[mask].max() + 1)
    else:
        next_cluster_id = start_cluster_id
    subset["cluster_id"] = adjusted_labels
    return subset, next_cluster_id


def compute_dbscan_hotspots(
    gdf: gpd.GeoDataFrame, group_col: str = "community_area_number"
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """Run DBSCAN per community area (to limit memory) and return derived tables."""

    if group_col not in gdf.columns:
        grouped_iter = [(None, gdf)]
    else:
        grouped_iter = gdf.groupby(group_col, dropna=False)

    clustered_parts: list[gpd.GeoDataFrame] = []
    next_cluster_id = 0
    for group_value, subset in grouped_iter:
        if isinstance(group_value, tuple):  # pandas may return tuples for multi-index
            group_value = group_value[0]
        print(f"Running DBSCAN for {group_col}={group_value} (rows={len(subset)})")
        clustered_subset, next_cluster_id = _cluster_subset(subset, next_cluster_id)
        clustered_parts.append(clustered_subset)

    clustered = pd.concat(clustered_parts, ignore_index=True)

    num_clusters = clustered.loc[clustered["cluster_id"] >= 0, "cluster_id"].nunique()
    noise = int((clustered["cluster_id"] == -1).sum())
    print(f"Clusters found (excluding noise): {num_clusters}")
    print(f"Noise points (label == -1): {noise}")

    cluster_sizes = clustered["cluster_id"].value_counts().sort_index()
    print("Cluster sizes (label:count):")
    print(cluster_sizes.to_string())

    centroids = compute_cluster_centroids(clustered)
    severity = compute_cluster_severity(clustered)
    return clustered, centroids, severity


def main() -> None:
    gdf = gpd.read_file(CLUSTERS_SOURCE_PATH)

    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    if gdf.empty:
        raise ValueError("No crashes available for clustering")

    clustered_gdf, centroids, severity = compute_dbscan_hotspots(gdf)

    CLUSTERS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clustered_gdf.to_file(CLUSTERS_OUTPUT_PATH, driver="GeoJSON")
    print(f"Saved clustered crashes to {CLUSTERS_OUTPUT_PATH}")

    centroids.to_file(CENTROIDS_OUTPUT_PATH, driver="GeoJSON")
    print(f"Saved cluster centroids to {CENTROIDS_OUTPUT_PATH}")

    severity.to_csv(RISK_TABLE_PATH, index=False)
    print(f"Saved cluster risk table to {RISK_TABLE_PATH}")

    print("Worst 10 clusters by severity score:")
    print(severity.sort_values("weighted_score", ascending=False).head(10))
    print("Largest 10 clusters by total crashes:")
    print(severity.sort_values("total_crashes", ascending=False).head(10))
    print("Top 10 centroid coordinates:")
    print(centroids[["cluster_id", "centroid_lon", "centroid_lat"]].head(10))


if __name__ == "__main__":
    main()
