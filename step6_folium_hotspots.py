# step6_folium_hotspots.py
#!/usr/bin/env python3
"""
Step 6b: Folium hotspot map for DBSCAN clusters and crash points.

- Loads:
    1) data/output/crashes_with_clusters.geojson  (POINTS with 'cluster_id')
    2) data/output/cluster_risk_table.csv         (cluster metrics incl. weighted_score, centroid_lon/lat)
    3) data/output/nsw_postcodes.geojson          (optional overlay)
- Creates layers:
    - Raw Crashes (points)
    - Clusters (DBSCAN) (centroid markers scaled + colored by risk)
    - NSW Postcode Areas (outline only)
    - Heatmap (crash density)
- Saves:
    data/output/folium_hotspot_map.html

Run from project root:
    python step6_folium_hotspots.py
"""
from pathlib import Path
import logging
import sys
from typing import Tuple, List

try:
    import geopandas as gpd
    import pandas as pd
    import folium
    from folium.plugins import HeatMap
except Exception as e:  # pragma: no cover
    print(
        "This script requires geopandas, pandas, and folium.\n"
        "Install with: pip install geopandas pandas folium",
        file=sys.stderr,
    )
    raise


def project_root() -> Path:
    """Return the project root (directory containing this script)."""
    return Path(__file__).resolve().parent


def get_paths() -> Tuple[Path, Path, Path, Path]:
    """Compute input/output paths relative to the project root."""
    root = project_root()
    crashes_path = root / "data" / "output" / "crashes_with_clusters.geojson"
    cluster_table_path = root / "data" / "output" / "cluster_risk_table.csv"
    poa_geojson_path = root / "data" / "output" / "nsw_postcodes.geojson"  # optional
    out_html_path = root / "data" / "output" / "folium_hotspot_map.html"
    return crashes_path, cluster_table_path, poa_geojson_path, out_html_path


def load_layers(crashes_path: Path, cluster_table_path: Path, poa_geojson_path: Path) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, gpd.GeoDataFrame | None]:
    """Load crashes, cluster risk table, and optional NSW postcode polygons."""
    logging.info("Loading layers...")
    if not crashes_path.exists():
        raise FileNotFoundError(f"Crashes with clusters not found: {crashes_path}")
    if not cluster_table_path.exists():
        raise FileNotFoundError(f"Cluster risk table not found: {cluster_table_path}")

    # Load crashes with clusters
    gdf_crashes = gpd.read_file(crashes_path)
    if gdf_crashes.empty:
        raise ValueError("Crashes dataset is empty.")
    # Ensure WGS84 for folium lat/lon
    if gdf_crashes.crs is None:
        logging.warning("Crash points CRS missing; assuming EPSG:4326")
        gdf_crashes = gdf_crashes.set_crs(epsg=4326, allow_override=True)
    elif (gdf_crashes.crs.to_epsg() if hasattr(gdf_crashes.crs, "to_epsg") else None) != 4326:
        gdf_crashes = gdf_crashes.to_crs(epsg=4326)

    # Load cluster table (centroids + metrics)
    df_clusters = pd.read_csv(cluster_table_path)
    # Ensure expected columns exist
    required = {
        "cluster_id",
        "total_crashes",
        "killed",
        "serious",
        "moderate",
        "minor",
        "weighted_score",
        "centroid_lon",
        "centroid_lat",
    }
    missing = [c for c in required if c not in df_clusters.columns]
    if missing:
        raise KeyError(f"Missing columns in cluster_risk_table.csv: {missing}")

    # Optional NSW postcodes overlay
    gdf_poa = None
    if poa_geojson_path.exists():
        try:
            gdf_poa = gpd.read_file(poa_geojson_path)
            if gdf_poa.crs is None:
                logging.warning("POA CRS missing; assuming EPSG:4326")
                gdf_poa = gdf_poa.set_crs(epsg=4326, allow_override=True)
            elif (gdf_poa.crs.to_epsg() if hasattr(gdf_poa.crs, "to_epsg") else None) != 4326:
                gdf_poa = gdf_poa.to_crs(epsg=4326)
        except Exception as e:
            logging.warning("Failed to load NSW postcodes overlay: %s", e)
            gdf_poa = None
    else:
        logging.info("NSW postcode overlay not found, skipping: %s", poa_geojson_path)

    return gdf_crashes, df_clusters, gdf_poa


def merge_cluster_metrics(gdf_crashes: gpd.GeoDataFrame, df_clusters: pd.DataFrame) -> pd.DataFrame:
    """Merge cluster metrics with cluster ids for easy lookup while rendering centroids."""
    logging.info("Merging cluster metrics...")
    # Exclude noise in aggregate table; we'll still draw raw crashes (including noise) on their own layer
    df_valid = df_clusters[df_clusters["cluster_id"].notna()].copy()
    return df_valid


def compute_marker_color(weighted_score: float) -> str:
    """Color rule based on weighted_score."""
    if pd.isna(weighted_score):
        return "gray"
    if weighted_score < 0.5:
        return "green"
    if weighted_score < 1.0:
        return "orange"
    if weighted_score < 1.5:
        return "red"
    return "darkred"


def compute_marker_radius(total_crashes: float) -> float:
    """Scale marker radius per spec: radius = max(5, min(25, total_crashes / 3))."""
    try:
        r = float(total_crashes) / 3.0
    except Exception:
        r = 5.0
    return max(5.0, min(25.0, r))


def make_base_map(gdf_crashes: gpd.GeoDataFrame) -> folium.Map:
    """Initialize a Folium map centered on mean lat/lon of crashes."""
    logging.info("Initializing Folium map...")
    # Compute center from crashes mean lat/lon
    lats = gdf_crashes.geometry.y
    lons = gdf_crashes.geometry.x
    center_lat = float(lats.mean())
    center_lon = float(lons.mean())
    m = folium.Map(location=[center_lat, center_lon], tiles="cartodbpositron", zoom_start=6, control_scale=True)
    return m


def add_raw_crashes_layer(m: folium.Map, gdf_crashes: gpd.GeoDataFrame) -> None:
    """Add raw crash points as a layer."""
    logging.info("Adding Raw Crashes layer...")
    fg = folium.FeatureGroup(name="Raw Crashes", show=False)
    # Draw as small circle markers for performance
    for geom in gdf_crashes.geometry:
        if geom is None or geom.is_empty:
            continue
        folium.CircleMarker(
            location=[geom.y, geom.x],
            radius=2,
            color="blue",
            fill=True,
            fill_color="blue",
            opacity=0.3,
            fill_opacity=0.3,
            weight=0,
        ).add_to(fg)
    fg.add_to(m)


def add_clusters_layer(m: folium.Map, df_clusters: pd.DataFrame) -> None:
    """Add cluster centroid markers (excluding noise cluster_id == -1)."""
    logging.info("Adding Clusters (DBSCAN) layer...")
    fg = folium.FeatureGroup(name="Clusters (DBSCAN)", show=True)

    # Filter valid clusters with valid centroid
    df = df_clusters.copy()
    df = df[(df["cluster_id"] != -1) & df["centroid_lat"].notna() & df["centroid_lon"].notna()]
    for _, row in df.iterrows():
        if pd.notna(row["weighted_score"]):
            ws = f"{float(row['weighted_score']):.3f}"
        else:
            ws = "NA"
        lat = float(row["centroid_lat"])
        lon = float(row["centroid_lon"])
        color = compute_marker_color(row["weighted_score"])
        radius = compute_marker_radius(row["total_crashes"])
        popup_html = (
            f"<b>Cluster ID:</b> {row['cluster_id']}<br>"
            f"Total crashes: {int(row['total_crashes']) if pd.notna(row['total_crashes']) else 'NA'}<br>"
            f"Killed: {int(row['killed']) if pd.notna(row['killed']) else 'NA'}<br>"
            f"Serious: {int(row['serious']) if pd.notna(row['serious']) else 'NA'}<br>"
            f"Moderate: {int(row['moderate']) if pd.notna(row['moderate']) else 'NA'}<br>"
            f"Minor: {int(row['minor']) if pd.notna(row['minor']) else 'NA'}<br>"
            f"Weighted score: {ws}"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            opacity=0.9,
            fill_opacity=0.6,
            weight=1,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(fg)
    fg.add_to(m)


def add_poa_overlay(m: folium.Map, gdf_poa: gpd.GeoDataFrame | None) -> None:
    """Add NSW postcode polygons overlay if available."""
    if gdf_poa is None or gdf_poa.empty:
        return
    logging.info("Adding NSW Postcode Areas overlay...")
    style_function = lambda x: {"color": "black", "weight": 0.5, "fillOpacity": 0}
    folium.GeoJson(
        data=gdf_poa.to_json(),
        name="NSW Postcode Areas",
        style_function=style_function,
        control=True,
        show=False,
    ).add_to(m)


def add_heatmap_layer(m: folium.Map, gdf_crashes: gpd.GeoDataFrame) -> None:
    """Add a heatmap of crash points."""
    logging.info("Adding Heatmap layer...")
    # Prepare heatmap points as [lat, lon]
    pts: List[List[float]] = []
    for g in gdf_crashes.geometry:
        if g is None or g.is_empty:
            continue
        pts.append([g.y, g.x])
    if not pts:
        logging.info("No points available for heatmap.")
        return
    fg = folium.FeatureGroup(name="Heatmap", show=False)
    HeatMap(pts, radius=10, blur=15, min_opacity=0.4).add_to(fg)
    fg.add_to(m)


def save_map(m: folium.Map, out_path: Path) -> None:
    """Save folium map to HTML."""
    logging.info("Saving map HTML...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print("✓ Folium hotspot map saved.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        crashes_path, cluster_table_path, poa_geojson_path, out_html = get_paths()

        # Load layers
        gdf_crashes, df_clusters, gdf_poa = load_layers(crashes_path, cluster_table_path, poa_geojson_path)
        df_clusters_valid = merge_cluster_metrics(gdf_crashes, df_clusters)

        # Initialize map
        fmap = make_base_map(gdf_crashes)

        # Add layers
        add_raw_crashes_layer(fmap, gdf_crashes)
        add_clusters_layer(fmap, df_clusters_valid)
        add_poa_overlay(fmap, gdf_poa)
        add_heatmap_layer(fmap, gdf_crashes)

        # Layer control
        folium.LayerControl(collapsed=False).add_to(fmap)

        # Save
        save_map(fmap, out_html)
        logging.info("Completed successfully. Output written to: %s", out_html)
    except Exception as exc:
        logging.exception("Error while creating Folium hotspot map: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()