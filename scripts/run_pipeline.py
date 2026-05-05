from src.preprocessing import preprocess_data

def main():
    preprocess_data(
        input_path="/tmp/data/train.csv",
        output_path="/tmp/data/clean.parquet",
        nrows=50000   # ✅ correct parameter
    )

if __name__ == "__main__":
    main()