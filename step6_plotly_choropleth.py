#!/usr/bin/env python3
"""
Step 6: Plotly choropleth of NSW POA risk scores.

- Loads NSW POA polygons (GeoJSON) and ZIP/POA risk table (CSV)
- Merges on POA_CODE21
- Builds a Plotly choropleth using weighted_score with Viridis scale
- Centers and fits the map to NSW automatically
- Saves to data/output/plotly_nsw_risk_map.html

Run from project root:
    python step6_plotly_choropleth.py
"""
from pathlib import Path
import logging
import sys
import json
from typing import Tuple

try:
    import geopandas as gpd
    import pandas as pd
    import plotly.express as px
except Exception as e:  # pragma: no cover
    print(
        "This script requires geopandas, pandas, and plotly.\n"
        "Install with: pip install geopandas pandas plotly",
        file=sys.stderr,
    )
    raise


def project_root() -> Path:
    """Return the project root (directory containing this script)."""
    return Path(__file__).resolve().parent


def get_paths() -> Tuple[Path, Path, Path]:
    """Compute input/output paths relative to the project root."""
    root = project_root()
    poa_geojson_path = root / "data" / "output" / "nsw_postcodes.geojson"
    zip_risk_csv_path = root / "data" / "output" / "zip_risk_table.csv"
    out_html_path = root / "data" / "output" / "plotly_nsw_risk_map.html"
    return poa_geojson_path, zip_risk_csv_path, out_html_path


def load_data(poa_path: Path, risk_path: Path) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Load NSW POA polygons and risk table."""
    logging.info("Loading data...")
    if not poa_path.exists():
        raise FileNotFoundError(f"POA GeoJSON not found: {poa_path}")
    if not risk_path.exists():
        raise FileNotFoundError(f"Risk table CSV not found: {risk_path}")

    poa_gdf = gpd.read_file(poa_path)
    if poa_gdf.crs is None:
        logging.warning("POA layer CRS missing; assuming EPSG:4326")
        poa_gdf = poa_gdf.set_crs(epsg=4326, allow_override=True)
    elif (poa_gdf.crs.to_epsg() if hasattr(poa_gdf.crs, 'to_epsg') else None) != 4326:
        poa_gdf = poa_gdf.to_crs(epsg=4326)

    # Simplify polygons to reduce client-side rendering cost in Plotly
    logging.info("Simplifying POA polygons (~75 m tolerance)...")
    try:
        poa_gdf_m = poa_gdf.to_crs(epsg=3857)
        poa_gdf_m["geometry"] = poa_gdf_m.geometry.simplify(75, preserve_topology=True)
        poa_gdf = poa_gdf_m.to_crs(epsg=4326)
        # Save a simplified copy for reuse
        simplified_path = poa_path.parent / "nsw_postcodes_simplified.geojson"
        try:
            poa_gdf.to_file(simplified_path, driver="GeoJSON")
            logging.info("Saved simplified POA polygons to: %s", simplified_path)
        except Exception as e:
            logging.warning("Could not save simplified POA polygons: %s", e)
    except Exception as e:
        logging.warning("Simplification step failed; proceeding with original geometry: %s", e)

    # Ensure POA_CODE21 is string in both for a clean merge
    risk_df = pd.read_csv(risk_path, dtype={"POA_CODE21": "string"})
    poa_gdf["POA_CODE21"] = poa_gdf["POA_CODE21"].astype("string")

    # Ensure weighted_score numeric for coloring
    if "weighted_score" in risk_df.columns:
        risk_df["weighted_score"] = pd.to_numeric(risk_df["weighted_score"], errors="coerce")

    # Ensure expected columns exist
    required_cols = {
        "POA_CODE21",
        "total_crashes",
        "killed",
        "serious",
        "moderate",
        "minor",
        "weighted_score",
    }
    missing = [c for c in required_cols if c not in risk_df.columns]
    if missing:
        raise KeyError(f"Missing columns in risk table: {missing}")

    return poa_gdf, risk_df


def merge_layers(poa_gdf: gpd.GeoDataFrame, risk_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Merge POA polygons with risk table on POA_CODE21."""
    logging.info("Merging...")
    merged = poa_gdf.merge(risk_df, on="POA_CODE21", how="left")
    return merged


def gdf_to_geojson_dict(gdf_wgs84: gpd.GeoDataFrame) -> dict:
    """Convert a WGS84 GeoDataFrame to a GeoJSON dict for Plotly."""
    # Ensure WGS84 for geo plotting
    if gdf_wgs84.crs is None or (gdf_wgs84.crs.to_epsg() if hasattr(gdf_wgs84.crs, 'to_epsg') else None) != 4326:
        gdf_wgs84 = gdf_wgs84.to_crs(epsg=4326)
    geojson_str = gdf_wgs84.to_json()
    return json.loads(geojson_str)


def build_choropleth(merged: gpd.GeoDataFrame, geojson_obj: dict):
    """Build a Plotly choropleth figure using weighted_score."""
    logging.info("Building choropleth...")

    # Prepare custom data for formatted hovertemplate
    custom_cols = [
        "POA_CODE21",
        "total_crashes",
        "killed",
        "serious",
        "moderate",
        "minor",
        "weighted_score",
    ]
    # Ensure columns exist; fill missing with NaN if any
    for c in custom_cols:
        if c not in merged.columns:
            merged[c] = pd.NA

    fig = px.choropleth(
        merged,
        geojson=geojson_obj,
        locations="POA_CODE21",
        featureidkey="properties.POA_CODE21",
        color="weighted_score",
        color_continuous_scale="Viridis",
        labels={"weighted_score": "Weighted score"},
        custom_data=merged[custom_cols],
    )

    # Hover formatting: weighted_score to 3 decimals
    fig.update_traces(
        hovertemplate=(
            "<b>POA:</b> %{customdata[0]}<br>"
            "Total crashes: %{customdata[1]}<br>"
            "Killed: %{customdata[2]}<br>"
            "Serious: %{customdata[3]}<br>"
            "Moderate: %{customdata[4]}<br>"
            "Minor: %{customdata[5]}<br>"
            "Weighted score: %{customdata[6]:.3f}<extra></extra>"
        )
    )

    # Fit to NSW polygons and style the geo
    fig.update_geos(
        fitbounds="locations",
        visible=True,
        showcountries=False,
        showcoastlines=True,
        coastlinecolor="lightgray",
        showland=True,
        landcolor="white",
        projection_type="mercator",
    )

    # Clean layout aesthetic
    fig.update_layout(
        title=dict(text="NSW Crash Risk by Postcode (Weighted Severity)", x=0.5, xanchor="center"),
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        coloraxis_colorbar=dict(
            title="Weighted score",
            ticks="outside",
        ),
    )
    return fig


def save_html(fig, out_path: Path) -> None:
    """Save the Plotly figure to HTML."""
    logging.info("Saving HTML map...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        poa_path, risk_path, out_html = get_paths()
        poa_gdf, risk_df = load_data(poa_path, risk_path)
        merged = merge_layers(poa_gdf, risk_df)
        geojson_obj = gdf_to_geojson_dict(merged)
        fig = build_choropleth(merged, geojson_obj)
        save_html(fig, out_html)
        logging.info("Completed successfully. Output written to: %s", out_html)
        print(f"Map exported to {out_html}")
    except Exception as exc:
        logging.exception("Error building choropleth: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
