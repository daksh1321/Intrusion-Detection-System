import pandas as pd
import numpy as np
import os

INPUT_FILE = (
    r"E:\IDS_Project\data\processed"
    r"\CIC_IDS2017_preprocessed.csv"
)

OUTPUT_FOLDER = (
    r"E:\IDS_Project\data\processed"
)

TRAIN_FILE = os.path.join(
    OUTPUT_FOLDER,
    "CIC_IDS2017_train.csv"
)

TEST_FILE = os.path.join(
    OUTPUT_FOLDER,
    "CIC_IDS2017_test.csv"
)

CHUNK_SIZE = 50000

TRAIN_RATIO = 0.80

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 70)
print("CIC-IDS2017 TRAIN / TEST SPLIT")
print("=" * 70)

# Remove old files if they exist

if os.path.exists(TRAIN_FILE):
    os.remove(TRAIN_FILE)

if os.path.exists(TEST_FILE):
    os.remove(TEST_FILE)

train_first = True
test_first = True

total_rows = 0
train_rows = 0
test_rows = 0

rng = np.random.default_rng(42)

# Process dataset in chunks

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    print(f"\nProcessing chunk {chunk_number}...")

    chunk.columns = chunk.columns.str.strip()

    # Random 80/20 split

    random_values = rng.random(len(chunk))

    train_mask = random_values < TRAIN_RATIO

    train_chunk = chunk[train_mask]
    test_chunk = chunk[~train_mask]

    # Save training data

    train_chunk.to_csv(
        TRAIN_FILE,
        mode="w" if train_first else "a",
        header=train_first,
        index=False
    )

    train_first = False

    # Save testing data

    test_chunk.to_csv(
        TEST_FILE,
        mode="w" if test_first else "a",
        header=test_first,
        index=False
    )

    test_first = False

    # Counters

    total_rows += len(chunk)
    train_rows += len(train_chunk)
    test_rows += len(test_chunk)

    print(
        f"Train rows: {train_rows:,} | "
        f"Test rows: {test_rows:,}"
    )

# Final result

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT COMPLETE")
print("=" * 70)

print(f"Total rows : {total_rows:,}")
print(f"Train rows : {train_rows:,}")
print(f"Test rows  : {test_rows:,}")

print(
    f"\nTrain percentage: "
    f"{train_rows / total_rows * 100:.2f}%"
)

print(
    f"Test percentage: "
    f"{test_rows / total_rows * 100:.2f}%"
)

print("\nTrain file:")
print(TRAIN_FILE)

print("\nTest file:")
print(TEST_FILE)