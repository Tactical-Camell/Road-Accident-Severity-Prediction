#!/usr/bin/env python3
"""
Step 7: Dash web app to visualize NSW POA crash risk with interactivity.

- Loads NSW POA polygons (prefers simplified GeoJSON if present) and ZIP/POA risk table
- Choropleth Mapbox with controls for metric, minimum total crashes, and color scale
- Designed for smoother interaction using simplified geometries and Mapbox rendering

Run from project root:
    python step7_dash_app.py
"""
from pathlib import Path
import logging
import sys
import json
from typing import Tuple

try:
    from dash import Dash, dcc, html, Input, Output
    import plotly.express as px
    import geopandas as gpd
    import pandas as pd
except Exception as e:  # pragma: no cover
    print(
        "This app requires dash, plotly, geopandas, and pandas.\n"
        "Install with: pip install dash plotly geopandas pandas",
        file=sys.stderr,
    )
    raise


# ----------------------------- Paths & Loading ----------------------------- #

def project_root() -> Path:
    return Path(__file__).resolve().parent


def get_paths() -> Tuple[Path, Path]:
    root = project_root()
    simplified = root / "data" / "output" / "nsw_postcodes_simplified.geojson"
    original = root / "data" / "output" / "nsw_postcodes.geojson"
    risk_csv = root / "data" / "output" / "zip_risk_table.csv"
    poa_geo = simplified if simplified.exists() else original
    return poa_geo, risk_csv


def load_layers(poa_path: Path, risk_path: Path) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    logging.info("Loading layers...")
    if not poa_path.exists():
        raise FileNotFoundError(f"POA GeoJSON not found: {poa_path}")
    if not risk_path.exists():
        raise FileNotFoundError(f"Risk table not found: {risk_path}")

    gdf = gpd.read_file(poa_path)
    if gdf.crs is None:
        logging.warning("POA CRS missing; assuming EPSG:4326")
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    elif (gdf.crs.to_epsg() if hasattr(gdf.crs, 'to_epsg') else None) != 4326:
        gdf = gdf.to_crs(epsg=4326)

    risk = pd.read_csv(risk_path, dtype={"POA_CODE21": "string"})
    gdf["POA_CODE21"] = gdf["POA_CODE21"].astype("string")
    # Ensure numeric for metrics
    for c in [
        "total_crashes",
        "killed",
        "serious",
        "moderate",
        "minor",
        "weighted_score",
    ]:
        if c in risk.columns:
            risk[c] = pd.to_numeric(risk[c], errors="coerce")

    return gdf, risk


def merge_layers(poa_gdf: gpd.GeoDataFrame, risk_df: pd.DataFrame) -> gpd.GeoDataFrame:
    logging.info("Merging layers...")
    merged = poa_gdf.merge(risk_df, on="POA_CODE21", how="left")
    return merged


def gdf_to_geojson_dict(gdf: gpd.GeoDataFrame) -> dict:
    # Ensure WGS84
    if gdf.crs is None or (gdf.crs.to_epsg() if hasattr(gdf.crs, 'to_epsg') else None) != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return json.loads(gdf.to_json())


# ----------------------------- Figure Builder ----------------------------- #

def compute_center(gdf: gpd.GeoDataFrame) -> dict:
    minx, miny, maxx, maxy = gdf.total_bounds
    return {"lon": float((minx + maxx) / 2.0), "lat": float((miny + maxy) / 2.0)}


def build_figure(
    merged: gpd.GeoDataFrame,
    geojson_obj: dict,
    metric: str,
    min_total_crashes: int,
    color_scale: str,
):
    df = merged.copy()
    # Filter by min total crashes (keep polygons with NA as off by default)
    if "total_crashes" in df.columns:
        df = df[df["total_crashes"].fillna(0) >= min_total_crashes]

    # Prepare custom data for hover
    custom_cols = [
        "POA_CODE21",
        "total_crashes",
        "killed",
        "serious",
        "moderate",
        "minor",
        "weighted_score",
    ]
    for c in custom_cols:
        if c not in df.columns:
            df[c] = pd.NA

    center = compute_center(merged)  # center on full NSW extent

    fig = px.choropleth_mapbox(
        df,
        geojson=geojson_obj,
        locations="POA_CODE21",
        featureidkey="properties.POA_CODE21",
        color=metric,
        color_continuous_scale=color_scale,
        custom_data=df[custom_cols],
        center=center,
        zoom=5.3,
        opacity=0.9,
    )

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

    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text="NSW Crash Risk by Postcode", x=0.5, xanchor="center"),
        coloraxis_colorbar=dict(title=metric, ticks="outside"),
        height=750,
    )

    # Outline boundaries for visual clarity
    fig.update_traces(marker_line_width=0.2, marker_line_color="#222")

    return fig


# ----------------------------- Dash App ----------------------------- #

def create_app() -> Dash:
    poa_path, risk_path = get_paths()
    poa_gdf, risk_df = load_layers(poa_path, risk_path)
    merged = merge_layers(poa_gdf, risk_df)
    geojson_obj = gdf_to_geojson_dict(poa_gdf)

    metrics = [
        {"label": "Weighted score", "value": "weighted_score"},
        {"label": "Total crashes", "value": "total_crashes"},
        {"label": "Killed", "value": "killed"},
        {"label": "Serious", "value": "serious"},
        {"label": "Moderate", "value": "moderate"},
        {"label": "Minor", "value": "minor"},
    ]
    color_scales = [
        {"label": "Viridis", "value": "Viridis"},
        {"label": "Plasma", "value": "Plasma"},
        {"label": "Inferno", "value": "Inferno"},
        {"label": "Cividis", "value": "Cividis"},
        {"label": "YlGnBu", "value": "YlGnBu"},
        {"label": "Reds", "value": "Reds"},
    ]

    min_total = int(pd.to_numeric(merged.get("total_crashes"), errors="coerce").min() or 0)
    max_total = int(pd.to_numeric(merged.get("total_crashes"), errors="coerce").max() or 100)
    if max_total < 10:
        max_total = 10

    app = Dash(__name__)
    app.title = "NSW Crash Risk — Dash"

    app.layout = html.Div(
        [
            html.H2("NSW Crash Risk by Postcode (Dash)"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Metric"),
                            dcc.Dropdown(
                                id="metric",
                                options=metrics,
                                value="weighted_score",
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1", "minWidth": "220px", "marginRight": "12px"},
                    ),
                    html.Div(
                        [
                            html.Label("Min total crashes"),
                            dcc.Slider(
                                id="min_total",
                                min=min_total,
                                max=max_total,
                                step=1,
                                value=min(20, max_total),
                                marks={
                                    min_total: str(min_total),
                                    max_total: str(max_total),
                                },
                                tooltip={"always_visible": False},
                            ),
                        ],
                        style={"flex": "2", "minWidth": "320px", "marginRight": "12px"},
                    ),
                    html.Div(
                        [
                            html.Label("Color scale"),
                            dcc.Dropdown(
                                id="color_scale",
                                options=color_scales,
                                value="Viridis",
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1", "minWidth": "220px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "alignItems": "center",
                    "gap": "8px",
                    "marginBottom": "10px",
                },
            ),
            dcc.Graph(id="map", config={"displayModeBar": True}),
            # Hidden stores for data (so we don't reload in callbacks)
            dcc.Store(id="store-geojson", data=geojson_obj),
            dcc.Store(id="store-merged", data=merged.to_json()),
        ],
        style={"padding": "12px", "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial"},
    )

    @app.callback(
        Output("map", "figure"),
        [
            Input("metric", "value"),
            Input("min_total", "value"),
            Input("color_scale", "value"),
            Input("store-geojson", "data"),
            Input("store-merged", "data"),
        ],
    )
    def update_map(metric, min_total_value, color_scale_value, geojson_data, merged_json):  # type: ignore
        # Rebuild GeoDataFrame from stored GeoJSON string
        merged_dict = json.loads(merged_json) if isinstance(merged_json, str) else merged_json
        features = merged_dict["features"] if isinstance(merged_dict, dict) and "features" in merged_dict else merged_dict
        merged_df = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        return build_figure(
            merged_df,
            geojson_data,
            metric=metric,
            min_total_crashes=int(min_total_value or 0),
            color_scale=color_scale_value,
        )

    # Initial figure will be rendered by callback on first load

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        app = create_app()
        logging.info("Starting Dash server at http://127.0.0.1:8050 ...")
        app.run(debug=False, host="127.0.0.1", port=8050)
    except Exception as exc:
        logging.exception("Error running Dash app: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
