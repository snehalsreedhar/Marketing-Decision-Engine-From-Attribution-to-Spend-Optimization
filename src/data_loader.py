import pandas as pd

def load_data(path: str, nrows: int = None) -> pd.DataFrame:
    """
    Load dataset from parquet or csv
    """

    print(f"📥 Loading data from: {path}")

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path, nrows=nrows)
    else:
        raise ValueError("Unsupported file format")

    print(f"   Shape: {df.shape}")
    return df