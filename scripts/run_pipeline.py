from src.preprocessing import preprocess_data

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

input_path = ROOT / "data" / "raw" / "train.csv"
output_path = ROOT / "data" / "processed" / "clean.parquet"

def main():
    preprocess_data(
        input_path=input_path,
        output_path=output_path,
        nrows = 50000
    )

if __name__ == "__main__":
    main()