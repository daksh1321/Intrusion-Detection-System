import pandas as pd
import numpy as np
import os

INPUT_FILE = (
    r"E:\IDS_Project\data\processed"
    r"\CIC_IDS2017_combined.csv"
)

OUTPUT_FILE = (
    r"E:\IDS_Project\data\processed"
    r"\CIC_IDS2017_preprocessed.csv"
)

CHUNK_SIZE = 50000

print("=" * 70)
print("CIC-IDS2017 PREPROCESSING")
print("=" * 70)

first_chunk = True
total_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    print(f"\nProcessing chunk {chunk_number}...")

    # 1. Clean column names

    chunk.columns = (
        chunk.columns
        .str.strip()
    )

    # 2. Clean label

    chunk["Label"] = (
        chunk["Label"]
        .astype(str)
        .str.strip()
    )

    # Fix encoding replacement character
    chunk["Label"] = (
        chunk["Label"]
        .str.replace("�", "-", regex=False)
    )

    # 3. Replace infinite values

    chunk.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # 4. Convert feature columns to numeric

    feature_columns = [
        col for col in chunk.columns
        if col != "Label"
    ]

    for col in feature_columns:

        chunk[col] = pd.to_numeric(
            chunk[col],
            errors="coerce"
        )

    # 5. Remove rows where label is missing

    chunk.dropna(
        subset=["Label"],
        inplace=True
    )

    # 6. Fill missing feature values

    chunk[feature_columns] = (
        chunk[feature_columns]
        .fillna(0)
    )

    # 7. Remove duplicate rows inside chunk

    chunk.drop_duplicates(
        inplace=True
    )

    # 8. Save processed chunk

    chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

    total_rows += len(chunk)

    print(
        f"Rows processed so far: "
        f"{total_rows:,}"
    )

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(
    f"Total processed rows: {total_rows:,}"
)

print("\nSaved to:")
print(OUTPUT_FILE)