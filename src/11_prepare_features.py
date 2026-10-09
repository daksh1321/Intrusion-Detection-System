import pandas as pd
import numpy as np
import os
import json

TRAIN_FILE = (
    r"E:\IDS_Project\data\processed"
    r"\CIC_IDS2017_train.csv"
)

TEST_FILE = (
    r"E:\IDS_Project\data\processed"
    r"\CIC_IDS2017_test.csv"
)

OUTPUT_FOLDER = (
    r"E:\IDS_Project\data\processed"
)

CHUNK_SIZE = 50000

# Output files

TRAIN_X_FILE = os.path.join(
    OUTPUT_FOLDER,
    "X_train.npy"
)

TRAIN_Y_FILE = os.path.join(
    OUTPUT_FOLDER,
    "y_train.npy"
)

TEST_X_FILE = os.path.join(
    OUTPUT_FOLDER,
    "X_test.npy"
)

TEST_Y_FILE = os.path.join(
    OUTPUT_FOLDER,
    "y_test.npy"
)

LABEL_MAP_FILE = os.path.join(
    OUTPUT_FOLDER,
    "label_mapping.json"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# STEP 1 — Find all labels

print("=" * 70)
print("STEP 1: FINDING LABELS")
print("=" * 70)

labels = set()

for chunk in pd.read_csv(
    TRAIN_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    labels.update(
        chunk["Label"]
        .astype(str)
        .str.strip()
        .unique()
    )

labels = sorted(labels)

print("\nNumber of classes:", len(labels))

print("\nClasses:")

for i, label in enumerate(labels):
    print(f"{i:2d} -> {label}")

# STEP 2 — Create label mapping

label_to_id = {
    label: i
    for i, label in enumerate(labels)
}

id_to_label = {
    str(i): label
    for label, i in label_to_id.items()
}

with open(
    LABEL_MAP_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "label_to_id": label_to_id,
            "id_to_label": id_to_label
        },
        f,
        indent=4
    )

print("\nLabel mapping saved to:")
print(LABEL_MAP_FILE)


# STEP 3 — Determine feature columns

sample = pd.read_csv(
    TRAIN_FILE,
    nrows=5,
    low_memory=False
)

sample.columns = sample.columns.str.strip()

FEATURE_COLUMNS = [
    col
    for col in sample.columns
    if col != "Label"
]

print("\nNumber of features:", len(FEATURE_COLUMNS))


# STEP 4 — Find constant features

print("\n" + "=" * 70)
print("CHECKING CONSTANT FEATURES")
print("=" * 70)

unique_values = {
    col: set()
    for col in FEATURE_COLUMNS
}

for chunk in pd.read_csv(
    TRAIN_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    chunk.columns = chunk.columns.str.strip()

    for col in FEATURE_COLUMNS:

        values = (
            pd.to_numeric(
                chunk[col],
                errors="coerce"
            )
            .dropna()
            .unique()
        )

        unique_values[col].update(values)

constant_features = [
    col
    for col in FEATURE_COLUMNS
    if len(unique_values[col]) <= 1
]

print(
    "\nConstant features found:",
    len(constant_features)
)

for col in constant_features:
    print(" -", col)


# Remove constant features

FINAL_FEATURES = [
    col
    for col in FEATURE_COLUMNS
    if col not in constant_features
]

print(
    "\nFeatures after removing constants:",
    len(FINAL_FEATURES)
)

# Function to process dataset

def process_dataset(
    input_file,
    x_output,
    y_output,
    dataset_name
):

    print("\n" + "=" * 70)
    print(f"PROCESSING {dataset_name}")
    print("=" * 70)

    # First pass: count rows

    total_rows = 0

    for chunk in pd.read_csv(
        input_file,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        total_rows += len(chunk)

    print(
        f"Total rows: {total_rows:,}"
    )

    # Create memory-mapped arrays

    X = np.lib.format.open_memmap(
        x_output,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, len(FINAL_FEATURES))
    )

    y = np.lib.format.open_memmap(
        y_output,
        mode="w+",
        dtype=np.int8,
        shape=(total_rows,)
    )

    current_position = 0

    # Process chunks

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_file,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        print(
            f"Processing chunk {chunk_number}..."
        )

        chunk.columns = chunk.columns.str.strip()

        # Features

        features = chunk[FINAL_FEATURES].apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Replace infinity
        features.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        # Fill missing values
        features.fillna(
            0,
            inplace=True
        )

        # Convert to float32
        features = features.astype(
            np.float32
        )

        labels_numeric = (
            chunk["Label"]
            .astype(str)
            .str.strip()
            .map(label_to_id)
            .astype(np.int8)
            .to_numpy()
        )


        rows = len(chunk)

        X[
            current_position:
            current_position + rows
        ] = features.to_numpy()

        y[
            current_position:
            current_position + rows
        ] = labels_numeric

        current_position += rows


    X.flush()
    y.flush()

    del X
    del y

    print(
        f"\n{dataset_name} processing complete."
    )


# STEP 5 — Process TRAIN

process_dataset(
    TRAIN_FILE,
    TRAIN_X_FILE,
    TRAIN_Y_FILE,
    "TRAIN DATA"
)


# STEP 6 — Process TEST

process_dataset(
    TEST_FILE,
    TEST_X_FILE,
    TEST_Y_FILE,
    "TEST DATA"
)


# COMPLETE

print("\n" + "=" * 70)
print("FEATURE PREPARATION COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print("X_train:", TRAIN_X_FILE)
print("y_train:", TRAIN_Y_FILE)
print("X_test :", TEST_X_FILE)
print("y_test :", TEST_Y_FILE)

print("\nFeatures used:", len(FINAL_FEATURES))
print("Classes:", len(labels))