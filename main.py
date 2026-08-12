from src.data.data_loader import load_raw_data, get_dataset_summary


def main():
    print("=" * 60)
    print("SMART FACTORY ALLOCATION")
    print("=" * 60)

    df = load_raw_data()

    summary = get_dataset_summary(df)

    print(f"Rows              : {summary['rows']:,}")
    print(f"Columns           : {summary['columns']}")
    print(f"Duplicate rows    : {summary['duplicate_rows']:,}")
    print(f"Missing values    : {summary['missing_values']:,}")
    print(f"Memory usage      : {summary['memory_usage_mb']} MB")

    print("\nDataset loaded successfully!")


if __name__ == "__main__":
    main()