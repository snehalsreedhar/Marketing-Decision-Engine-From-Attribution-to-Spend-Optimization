import pandas as pd
from pathlib import Path
import pandas as pd
from src.config import ROOT

def load_data(path, nrows=None):
    path = str(path) 

    full_path = ROOT / path
    print(f"📥 Loading: {full_path}")

    if path.endswith(".parquet"):
        return pd.read_parquet(full_path)

    return pd.read_csv(
        full_path,
        dtype={'fullVisitorId': 'str'},
        nrows=nrows
    )