"""Minimal sanity-check loader for the Pak3H dataset splits.

Usage:
    python3 scripts/load_data.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# (filename, expected row count, Urdu column to check for completeness)
SPLITS = {
    "alpaca": ("pak_alpaca.csv", 4500, "ctx_instruction_ur"),
    "beavertails": ("pak_beavertails.csv", 12583, "ctx_prompt_ur"),
    "truthfulqa": ("pak_truthfulqa.csv", 817, "ctx_Question_ur"),
}


def main():
    for split, (filename, expected_rows, ur_col) in SPLITS.items():
        path = DATA_DIR / split / filename
        df = pd.read_csv(path)
        print(f"{split}: {len(df)} rows, columns = {list(df.columns)}")

        assert len(df) == expected_rows, (
            f"{split}: expected {expected_rows} rows, got {len(df)}"
        )
        empty_ur = df[ur_col].isna().sum()
        assert empty_ur == 0, (
            f"{split}: {empty_ur} rows have an empty '{ur_col}' column"
        )

    print("\nAll splits have the expected row counts and complete Urdu coverage.")


if __name__ == "__main__":
    main()
