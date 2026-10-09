import pandas as pd
import os

FILE = (
    r"E:\IDS_Project\data\processed"
    r"\CIC_IDS2017_preprocessed.csv"
)

CHUNK_SIZE = 50000

total_rows = 0
label_counts = {}

print("=" * 70)
print("VERIFYING PREPROCESSED CIC-IDS2017 DATASET")
print("=" * 70)

for chunk_number, chunk in enumerate(
    pd.read_csv(
        FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    chunk.columns = chunk.columns.str.strip()

    total_rows += len(chunk)

    counts = chunk["Label"].value_counts()

    for label, count in counts.items():

        label_counts[label] = (
            label_counts.get(label, 0) + count
        )

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

print(
    f"\nTotal rows: {total_rows:,}"
)

print(
    f"Number of classes: {len(label_counts)}"
)

print("\nLabel distribution:")
print("-" * 70)

for label, count in sorted(
    label_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    percentage = (
        count / total_rows
    ) * 100

    print(
        f"{label:35s}"
        f"{count:12,}"
        f"   {percentage:7.3f}%"
    )

print("\n" + "=" * 70)
print("FILE INFORMATION")
print("=" * 70)

file_size_gb = os.path.getsize(FILE) / (1024 ** 3)

print(
    f"File size: {file_size_gb:.2f} GB"
)