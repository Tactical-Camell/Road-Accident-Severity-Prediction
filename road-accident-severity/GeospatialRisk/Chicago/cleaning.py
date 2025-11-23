from __future__ import annotations

"""Utilities to clean the Chicago crash dataset."""

import pandas as pd


def clean_chicago_crash_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows and columns valid for geospatial risk mapping."""

    required_columns = [
        "POSTED_SPEED_LIMIT",
        "TRAFFIC_CONTROL_DEVICE",
        "DEVICE_CONDITION",
        "WEATHER_CONDITION",
        "LIGHTING_CONDITION",
        "FIRST_CRASH_TYPE",
        "TRAFFICWAY_TYPE",
        "ALIGNMENT",
        "ROADWAY_SURFACE_COND",
        "ROAD_DEFECT",
        "REPORT_TYPE",
        "CRASH_TYPE",
        "DAMAGE",
        "PRIM_CONTRIBUTORY_CAUSE",
        "SEC_CONTRIBUTORY_CAUSE",
        "NUM_UNITS",
        "MOST_SEVERE_INJURY",
        "INJURIES_TOTAL",
        "INJURIES_FATAL",
        "INJURIES_INCAPACITATING",
        "INJURIES_NON_INCAPACITATING",
        "INJURIES_REPORTED_NOT_EVIDENT",
        "INJURIES_NO_INDICATION",
        "INJURIES_UNKNOWN",
        "CRASH_HOUR",
        "CRASH_DAY_OF_WEEK",
        "CRASH_MONTH",
        "LATITUDE",
        "LONGITUDE",
    ]

    env_columns = [
        "TRAFFIC_CONTROL_DEVICE",
        "DEVICE_CONDITION",
        "WEATHER_CONDITION",
        "LIGHTING_CONDITION",
        "TRAFFICWAY_TYPE",
        "ALIGNMENT",
        "ROADWAY_SURFACE_COND",
        "ROAD_DEFECT",
    ]

    injury_columns = [
        "INJURIES_TOTAL",
        "INJURIES_FATAL",
        "INJURIES_INCAPACITATING",
        "INJURIES_NON_INCAPACITATING",
        "INJURIES_REPORTED_NOT_EVIDENT",
        "INJURIES_NO_INDICATION",
        "INJURIES_UNKNOWN",
    ]

    df = df.copy()

    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")

    valid_coords = (
        df["LATITUDE"].between(-90, 90)
        & df["LONGITUDE"].between(-180, 180)
        & df["LATITUDE"].notna()
        & df["LONGITUDE"].notna()
        & (df["LATITUDE"] != 0)
        & (df["LONGITUDE"] != 0)
    )

    env_empty = df[env_columns].isna().all(axis=1)

    injury_sum = df[injury_columns].fillna(0).sum(axis=1)
    zero_injury_driveaway = (
        (injury_sum == 0) & (df["CRASH_TYPE"] == "NO INJURY / DRIVE AWAY")
    )

    keep_mask = valid_coords & ~env_empty & ~zero_injury_driveaway
    return df.loc[keep_mask, required_columns].reset_index(drop=True)
