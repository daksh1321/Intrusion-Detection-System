import pandas as pd
import glob
import os

# PATHS
INPUT_FOLDER = r"E:\IDS_Project\data\raw\CIC-IDS2017"

OUTPUT_FILE = (
    r"E:\IDS_Project\data\processed\CIC_IDS2017_combined.csv"
)

# FIND CSV FILES

files = glob.glob(
    os.path.join(INPUT_FOLDER, "*.csv")
)

print("CSV files found:", len(files))

# Remove output file if it somehow exists in the input folder
files = [
    f for f in files
    if os.path.abspath(f) != os.path.abspath(OUTPUT_FILE)
]

# COMBINE FILES

first_file = True

for file_number, file in enumerate(files, start=1):

    print("\n" + "=" * 70)
    print(f"FILE {file_number}/{len(files)}")
    print(os.path.basename(file))
    print("=" * 70)

    chunk_number = 0

    for chunk in pd.read_csv(
        file,
        chunksize=50000,
        low_memory=False
    ):

        chunk_number += 1

        # Clean column names
        chunk.columns = chunk.columns.str.strip()

        # Remove infinity
        chunk.replace(
            [float("inf"), float("-inf")],
            pd.NA,
            inplace=True
        )

        # Remove missing rows
        chunk.dropna(inplace=True)

        # Write to combined CSV
        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_file else "a",
            header=first_file,
            index=False
        )

        first_file = False

        print(
            f"Chunk {chunk_number}: "
            f"{len(chunk):,} rows written"
        )

print("\n" + "=" * 70)
print("COMBINATION COMPLETE")
print("=" * 70)

print("Output:")
print(OUTPUT_FILE)