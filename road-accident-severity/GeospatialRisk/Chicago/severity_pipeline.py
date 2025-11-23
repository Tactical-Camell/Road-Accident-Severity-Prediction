from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CRASHES_WITH_CLUSTERS_PATH = PROJECT_ROOT / "data" / "output" / "chicago_crashes_with_clusters.geojson"
SEVERITY_OUTPUT_PATH = PROJECT_ROOT / "data" / "output" / "chicago_crashes_with_severity.geojson"
COMMUNITY_RISK_PATH = PROJECT_ROOT / "data" / "output" / "community_area_risk.csv"
CLUSTER_RISK_PATH = PROJECT_ROOT / "data" / "output" / "cluster_risk_table.csv"

INJURY_COLUMNS = [
    "INJURIES_FATAL",
    "INJURIES_INCAPACITATING",
    "INJURIES_NON_INCAPACITATING",
    "INJURIES_REPORTED_NOT_EVIDENT",
    "INJURIES_NO_INDICATION",
]


def compute_crash_severity(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add crash-level weighted severity scores."""

    missing = [col for col in INJURY_COLUMNS if col not in gdf.columns]
    if missing:
        raise KeyError(f"Missing injury columns: {missing}")

    gdf = gdf.copy()
    numer = (
        3 * gdf["INJURIES_FATAL"]
        + 2 * gdf["INJURIES_INCAPACITATING"]
        + 1 * gdf["INJURIES_NON_INCAPACITATING"]
        + 0.5 * gdf["INJURIES_REPORTED_NOT_EVIDENT"]
    )
    denom = (
        gdf["INJURIES_FATAL"]
        + gdf["INJURIES_INCAPACITATING"]
        + gdf["INJURIES_NON_INCAPACITATING"]
        + gdf["INJURIES_REPORTED_NOT_EVIDENT"]
        + gdf["INJURIES_NO_INDICATION"]
    )
    gdf["weighted_severity"] = np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom > 0)
    return gdf


def _aggregate_common(df: pd.DataFrame) -> pd.DataFrame:
    total_injuries = (
        df["fatal"].astype(float)
        + df["serious"].astype(float)
        + df["moderate"].astype(float)
        + df["minor"].astype(float)
        + df["none"].astype(float)
    )
    df["risk_index"] = np.divide(
        3 * df["fatal"]
            + 2 * df["serious"]
            + 1 * df["moderate"]
            + 0.5 * df["minor"],
        total_injuries,
        out=np.zeros_like(total_injuries, dtype=float),
        where=total_injuries > 0,
    )
    return df


def compute_community_risk(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate severity metrics per community area."""

    if "community_area_number" not in gdf.columns:
        raise KeyError("community_area_number column missing")

    grouped = gdf.groupby("community_area_number", dropna=False)
    agg = grouped.agg(
        total_crashes=("geometry", "size"),
        fatal=("INJURIES_FATAL", "sum"),
        serious=("INJURIES_INCAPACITATING", "sum"),
        moderate=("INJURIES_NON_INCAPACITATING", "sum"),
        minor=("INJURIES_REPORTED_NOT_EVIDENT", "sum"),
        none=("INJURIES_NO_INDICATION", "sum"),
        mean_weighted_severity=("weighted_severity", "mean"),
    ).reset_index()

    # Ensure all 77 community areas are present, even if zero crashes
    area_numbers = pd.Series(range(1, 78), dtype="Int64", name="community_area_number")
    agg["community_area_number"] = agg["community_area_number"].astype("Int64")
    agg = area_numbers.to_frame().merge(agg, on="community_area_number", how="left")
    agg[["total_crashes", "fatal", "serious", "moderate", "minor", "none"]] = agg[
        ["total_crashes", "fatal", "serious", "moderate", "minor", "none"]
    ].fillna(0)
    agg["mean_weighted_severity"] = agg["mean_weighted_severity"].fillna(0.0)

    agg = _aggregate_common(agg)
    return agg


def compute_cluster_risk(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate severity metrics per DBSCAN cluster."""

    if "cluster_id" not in gdf.columns:
        raise KeyError("cluster_id column missing")

    clusters = gdf[gdf["cluster_id"] >= 0]
    grouped = clusters.groupby("cluster_id")
    agg = grouped.agg(
        total_crashes=("geometry", "size"),
        fatal=("INJURIES_FATAL", "sum"),
        serious=("INJURIES_INCAPACITATING", "sum"),
        moderate=("INJURIES_NON_INCAPACITATING", "sum"),
        minor=("INJURIES_REPORTED_NOT_EVIDENT", "sum"),
        none=("INJURIES_NO_INDICATION", "sum"),
    ).reset_index()
    agg = _aggregate_common(agg)
    agg = agg.rename(columns={"risk_index": "weighted_score"})
    return agg


def compute_severity_pipeline(gdf: gpd.GeoDataFrame):
    """Full crash + community + cluster severity scoring pipeline."""

    gdf = compute_crash_severity(gdf)
    community_risk = compute_community_risk(gdf)
    cluster_risk = compute_cluster_risk(gdf)
    return gdf, community_risk, cluster_risk


def main() -> None:
    gdf = gpd.read_file(CRASHES_WITH_CLUSTERS_PATH)

    required_cols = {
        "community_area_number",
        "cluster_id",
        *INJURY_COLUMNS,
    }
    missing = required_cols.difference(gdf.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf, community_risk, cluster_risk = compute_severity_pipeline(gdf)

    SEVERITY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(SEVERITY_OUTPUT_PATH, driver="GeoJSON")
    community_risk.to_csv(COMMUNITY_RISK_PATH, index=False)
    cluster_risk.to_csv(CLUSTER_RISK_PATH, index=False)

    print("Top 10 community areas by risk index:")
    print(community_risk.sort_values("risk_index", ascending=False).head(10))

    print("Bottom 10 community areas by risk index:")
    print(community_risk.sort_values("risk_index", ascending=True).head(10))

    print("Top 10 clusters by weighted score:")
    print(cluster_risk.sort_values("weighted_score", ascending=False).head(10))

    zero_crash = community_risk[community_risk["total_crashes"] == 0]
    print(f"Community areas with zero crashes: {len(zero_crash)}")

    print("weighted_severity summary stats:")
    print(gdf["weighted_severity"].describe())


if __name__ == "__main__":
    main()
