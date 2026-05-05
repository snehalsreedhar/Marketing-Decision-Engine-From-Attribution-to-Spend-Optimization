import pandas as pd
import json
import os

JSON_COLUMNS = [
    'device', 'geoNetwork', 'totals', 'trafficSource'
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
        col_df.columns = [f"{col}.{subcol}" for subcol in col_df.columns]

        df = df.drop(columns=[col]).join(col_df)

    return df


# ----------------------------
# SAFE NUMERIC HELPER
# ----------------------------
def _safe_numeric(series, index):
    """Ensures output is always a pandas Series"""
    return (
        pd.to_numeric(series, errors='coerce')
        .fillna(0)
        if isinstance(series, pd.Series)
        else pd.Series(0, index=index)
    )


# ----------------------------
# CLEAN + SELECT
# ----------------------------
def _clean_and_select(df: pd.DataFrame) -> pd.DataFrame:
    """Clean + select relevant columns safely"""

    cols = [
        'fullVisitorId',
        'visitId',
        'visitStartTime',
        'channelGrouping',
        'totals.transactions',
        'totals.totalTransactionRevenue'
    ]

    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()

    # ----------------------------
    # SAFE NUMERIC CONVERSIONS
    # ----------------------------
    df['totals.transactions'] = _safe_numeric(
        df.get('totals.transactions'),
        df.index
    )

    df['totals.totalTransactionRevenue'] = _safe_numeric(
        df.get('totals.totalTransactionRevenue'),
        df.index
    )

    # Revenue conversion
    df['revenue'] = df['totals.totalTransactionRevenue'] / 1e6

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
# MAIN PIPELINE FUNCTION
# ----------------------------
def preprocess_data(
    input_path: str,
    output_path: str = "data/processed/clean.parquet",
    nrows: int = None
):
    """
    Full preprocessing pipeline
    """

    print("📥 Loading raw data...")

    df = pd.read_csv(
        input_path,
        dtype={'fullVisitorId': 'str'},
        nrows=nrows
    ) if nrows else pd.read_csv(
        input_path,
        dtype={'fullVisitorId': 'str'}
    )

    print(f"   Shape after load: {df.shape}")

    print("🔄 Flattening JSON columns...")
    df = _flatten_json(df)

    print(f"   Shape after flatten: {df.shape}")

    print("🧹 Cleaning & selecting columns...")
    df = _clean_and_select(df)

    print(f"   Shape after clean: {df.shape}")

    print(f"💾 Saving processed data to {output_path}...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

    print("✅ Preprocessing complete!")

    return df