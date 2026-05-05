import pandas as pd
import numpy as np
import json
import os
from src.data_loader import load_data

EXTRA_PLACEHOLDERS = [
    "not available in demo dataset",
    "(not set)",
    "unknown.unknown",
    "",
    "NA"
]

# ----------------------------
# ALL JSON COLUMNS (UPDATED)
# ----------------------------
JSON_COLUMNS = [
    'device',
    'geoNetwork',
    'totals',
    'trafficSource'
]

# ----------------------------
# SAFE JSON PARSER
# ----------------------------
def _safe_json_load(x):
    try:
        return json.loads(x) if pd.notnull(x) else {}
    except Exception:
        return {}


# ----------------------------
# FLATTEN JSON COLUMNS
# ----------------------------
def _flatten_json(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten nested JSON columns safely"""

    for col in JSON_COLUMNS:
        if col not in df.columns:
            continue

        print(f"   ↳ Flattening {col}...")

        df[col] = df[col].apply(_safe_json_load)

        col_df = pd.json_normalize(df[col])

        # IMPORTANT FIX: safe column naming
        col_df.columns = [f"{col}_{subcol}" for subcol in col_df.columns]

        df = df.drop(columns=[col]).join(col_df)

    return df


# ----------------------------
# SAFE NUMERIC HELPER
# ----------------------------
def _safe_numeric(series, index):
    return (
        pd.to_numeric(series, errors='coerce')
        .fillna(0)
        if isinstance(series, pd.Series)
        else pd.Series(0, index=index)
    )


# ----------------------------
# CLEAN 
# ----------------------------
def _clean_all(df: pd.DataFrame) -> pd.DataFrame:
    """Clean entire dataframe without dropping any columns"""

    # ----------------------------
    # SAFE NUMERIC CONVERSIONS
    # ----------------------------
    if 'totals_transactions' in df.columns:
        df['totals_transactions'] = pd.to_numeric(df['totals_transactions'], errors='coerce').fillna(0)

    if 'totals_transactionRevenue' in df.columns:
        df['totals_transactionRevenue'] = pd.to_numeric(df['totals_transactionRevenue'], errors='coerce').fillna(0)

    # ----------------------------
    # REVENUE FEATURE
    # ----------------------------
    if 'totals_transactionRevenue' in df.columns:
        df['revenue'] = df['totals_transactionRevenue'] / 1e6
    else:
        df['revenue'] = 0

    # ----------------------------
    # TIMESTAMP CONVERSION
    # ----------------------------
    if 'visitStartTime' in df.columns:
        df['visitStartTime'] = pd.to_datetime(
            df['visitStartTime'],
            unit='s',
            errors='coerce'
        )

    return df

# ----------------------------
# MAIN PIPELINE
# ----------------------------
def preprocess_data(
    input_path: str,
    output_path: str = "data/processed/clean.parquet",
    nrows: int = None
):

    print("📥 Loading raw data...")

    df = load_data(input_path, nrows=nrows) if nrows else pd.read_csv(
        input_path,
        dtype={'fullVisitorId': 'str'}
    )

    print(f"   Shape after load: {df.shape}")

    print("🔄 Flattening JSON columns...")
    df = _flatten_json(df)

    print(f"   Shape after flatten: {df.shape}")

    print("🧹 Cleaning & selecting columns...")
    df = _clean_all(df)

    print(f"   Shape after clean: {df.shape}")

    print(f"💾 Saving processed data to {output_path}...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

    print("✅ Preprocessing complete!")

    return df


# ----------------------------
# SAVING OUTPUTS
# ----------------------------

def save_to_excel(df: pd.DataFrame, output_path: str = None):

    if output_path is None:
        output_path = input("Enter output file path (e.g. outputs/data.xlsx): ")

    if not output_path.endswith(".xlsx"):
        output_path += ".xlsx"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_excel(output_path, index=False)

    print(f"💾 Saved cleaned data to {output_path}")


# ------------------------------------------
# DROPPING MISSING VALUES/ UNAVAILABLE DATA
# ------------------------------------------


def clean_placeholders(df):

    cols_to_drop = [
        col for col in df.columns
        if df[col].isin(EXTRA_PLACEHOLDERS).all()
    ]

    df = df.drop(columns=cols_to_drop)

    # ----------------------------
    # STEP 2: REPLACE PARTIAL VALUES WITH NaN
    # ----------------------------
    df = df.replace(EXTRA_PLACEHOLDERS, np.nan)

    print(f"🧹 Dropped {len(cols_to_drop)} fully-placeholder columns")
    print(f"🧹 Replaced placeholder values with NaN")

    return df, cols_to_drop