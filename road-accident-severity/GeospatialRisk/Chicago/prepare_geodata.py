from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .cleaning import clean_chicago_crash_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "Chicago" / "dataset" / "Traffic_Crashes_-_Crashes.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "output" / "chicago_crash_points.geojson"
COMMUNITY_SHP_PATH = PROJECT_ROOT / "Chicago" / "data" / "Boundaries" / "geo_export_273d6492-11b0-415d-8d9d-bf83ed5c6833.shp"
COMMUNITY_OUTPUT_PATH = PROJECT_ROOT / "data" / "output" / "chicago_crashes_with_areas.geojson"


def convert_chicago_to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Cleans coordinates, converts to Point geometry, and returns a GeoDataFrame."""

    df = df.copy()
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")

    lat_mask = df["LATITUDE"].between(-90, 90)
    lon_mask = df["LONGITUDE"].between(-180, 180)
    nonzero_mask = ~((df["LATITUDE"] == 0) & (df["LONGITUDE"] == 0))

    coord_mask = (
        df["LATITUDE"].notna()
        & df["LONGITUDE"].notna()
        & lat_mask
        & lon_mask
        & nonzero_mask
    )

    df = df.loc[coord_mask]
    geometry = [Point(lon, lat) for lon, lat in zip(df["LONGITUDE"], df["LATITUDE"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def load_community_areas(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    rename_map = {}
    if "area_num_1" in gdf.columns:
        rename_map["area_num_1"] = "community_area_number"
    elif "area_numbe" in gdf.columns:
        rename_map["area_numbe"] = "community_area_number"
    if rename_map:
        gdf = gdf.rename(columns=rename_map)

    if "community" in gdf.columns and "community_area_name" not in gdf.columns:
        gdf["community_area_name"] = gdf["community"]

    if "community_area_number" in gdf.columns:
        gdf["community_area_number"] = pd.to_numeric(
            gdf["community_area_number"], errors="coerce"
        )

    if "geometry" not in gdf.columns:
        raise ValueError("Community area GeoDataFrame requires a 'geometry' column.")

    return gdf


def spatial_join_chicago(crash_gdf: gpd.GeoDataFrame, community_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "geometry" not in crash_gdf.columns or "geometry" not in community_gdf.columns:
        raise ValueError("Both GeoDataFrames must include a 'geometry' column for spatial join.")

    merged = gpd.sjoin(
        crash_gdf,
        community_gdf,
        how="left",
        predicate="within",
        lsuffix="_crash",
        rsuffix="_community",
    )

    drop_cols = [col for col in merged.columns if col.startswith("index_")]
    merged = merged.drop(columns=drop_cols)

    return merged


def main() -> None:
    df_raw = pd.read_excel(RAW_DATA_PATH, engine="openpyxl")
    df_clean = clean_chicago_crash_dataset(df_raw)

    pre_count = len(df_clean)
    gdf = convert_chicago_to_geodataframe(df_clean)
    invalid_removed = pre_count - len(gdf)

    lat_min = gdf["LATITUDE"].min()
    lat_max = gdf["LATITUDE"].max()
    lon_min = gdf["LONGITUDE"].min()
    lon_max = gdf["LONGITUDE"].max()

    print(f"Total crashes after geometry creation: {len(gdf)}")
    print(f"Invalid coordinates removed: {invalid_removed}")
    print(
        "Bounds (lat_min, lat_max, lon_min, lon_max): "
        f"{lat_min:.6f}, {lat_max:.6f}, {lon_min:.6f}, {lon_max:.6f}"
    )
    print(f"All geometries valid: {gdf.geometry.is_valid.all()}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"Saved GeoJSON to {OUTPUT_PATH}")

    community_gdf = load_community_areas(COMMUNITY_SHP_PATH)
    joined = spatial_join_chicago(gdf, community_gdf)

    assigned = joined["community_area_number"].notna().sum()
    unassigned = joined["community_area_number"].isna().sum()
    unique_areas = joined["community_area_number"].dropna().unique()

    duplicates_in_comm = community_gdf["community_area_number"].duplicated().sum()
    missing_area_codes = community_gdf["community_area_number"].isna().sum()

    print(f"Crashes assigned to community area: {assigned}")
    print(f"Crashes not assigned to any area: {unassigned}")
    print(f"Unique community_area_number values in joined data: {len(unique_areas)}")
    print(f"Duplicate area codes in shapefile: {duplicates_in_comm}")
    print(f"Missing area codes in shapefile: {missing_area_codes}")

    COMMUNITY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joined.to_file(COMMUNITY_OUTPUT_PATH, driver="GeoJSON")
    print(f"Saved joined GeoJSON to {COMMUNITY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
