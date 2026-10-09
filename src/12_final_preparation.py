import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


DATA_FOLDER = r"E:\IDS_Project\data\processed"

TRAIN_FILE = os.path.join(
    DATA_FOLDER,
    "CIC_IDS2017_train.csv"
)

TEST_FILE = os.path.join(
    DATA_FOLDER,
    "CIC_IDS2017_test.csv"
)

LABEL_MAP_FILE = os.path.join(
    DATA_FOLDER,
    "label_mapping.json"
)

X_TRAIN_FILE = os.path.join(
    DATA_FOLDER,
    "X_train.npy"
)

Y_TRAIN_FILE = os.path.join(
    DATA_FOLDER,
    "y_train.npy"
)

X_TEST_FILE = os.path.join(
    DATA_FOLDER,
    "X_test.npy"
)

Y_TEST_FILE = os.path.join(
    DATA_FOLDER,
    "y_test.npy"
)

X_TRAIN_SCALED_FILE = os.path.join(
    DATA_FOLDER,
    "X_train_scaled.npy"
)

X_TEST_SCALED_FILE = os.path.join(
    DATA_FOLDER,
    "X_test_scaled.npy"
)

SCALER_FILE = os.path.join(
    DATA_FOLDER,
    "standard_scaler.pkl"
)

CHUNK_SIZE = 50000

print("\n" + "=" * 75)
print("CIC-IDS2017 FINAL DATA PREPARATION")
print("=" * 75)

# STEP 1 — CHECK REQUIRED FILES

print("\n[1/7] Checking required files...")

required_files = [
    TRAIN_FILE,
    TEST_FILE,
    LABEL_MAP_FILE,
    X_TRAIN_FILE,
    Y_TRAIN_FILE,
    X_TEST_FILE,
    Y_TEST_FILE
]

for file in required_files:

    if not os.path.exists(file):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{file}"
        )

    print("OK:", os.path.basename(file))


# STEP 2 — LOAD LABEL MAPPING

print("\n[2/7] Loading label mapping...")

with open(
    LABEL_MAP_FILE,
    "r",
    encoding="utf-8"
) as f:

    label_mapping = json.load(f)

label_to_id = label_mapping["label_to_id"]
id_to_label = label_mapping["id_to_label"]

print(
    "Number of classes:",
    len(label_to_id)
)

for label, label_id in label_to_id.items():

    print(
        f"{label_id:2d} -> {label}"
    )

# STEP 3 — LOAD MEMORY-MAPPED ARRAYS
    
print("\n[3/7] Loading feature arrays...")

X_train = np.load(
    X_TRAIN_FILE,
    mmap_mode="r"
)

y_train = np.load(
    Y_TRAIN_FILE,
    mmap_mode="r"
)

X_test = np.load(
    X_TEST_FILE,
    mmap_mode="r"
)

y_test = np.load(
    Y_TEST_FILE,
    mmap_mode="r"
)

print("\nTraining X shape:", X_train.shape)
print("Training y shape:", y_train.shape)

print("\nTesting X shape:", X_test.shape)
print("Testing y shape:", y_test.shape)

# STEP 4 — FIT STANDARD SCALER

print("\n[4/7] Fitting StandardScaler...")
print("IMPORTANT: scaler is fitted ONLY on training data.")

scaler = StandardScaler()

train_rows = X_train.shape[0]

for start in range(
    0,
    train_rows,
    CHUNK_SIZE
):

    end = min(
        start + CHUNK_SIZE,
        train_rows
    )

    chunk = X_train[start:end]

    scaler.partial_fit(chunk)

    print(
        f"Fitted rows: "
        f"{end:,}/{train_rows:,}"
    )


joblib.dump(
    scaler,
    SCALER_FILE
)

print(
    "\nScaler saved:",
    SCALER_FILE
)

# STEP 5 — SCALE TRAINING DATA

print("\n[5/7] Scaling training data...")

X_train_scaled = np.lib.format.open_memmap(
    X_TRAIN_SCALED_FILE,
    mode="w+",
    dtype=np.float32,
    shape=X_train.shape
)

for start in range(
    0,
    train_rows,
    CHUNK_SIZE
):

    end = min(
        start + CHUNK_SIZE,
        train_rows
    )

    chunk = X_train[start:end]

    scaled_chunk = scaler.transform(
        chunk
    ).astype(np.float32)

    X_train_scaled[start:end] = (
        scaled_chunk
    )

    print(
        f"Scaled train rows: "
        f"{end:,}/{train_rows:,}"
    )

X_train_scaled.flush()

del X_train_scaled

# STEP 6 — SCALE TEST DATA

print("\n[6/7] Scaling testing data...")

test_rows = X_test.shape[0]

X_test_scaled = np.lib.format.open_memmap(
    X_TEST_SCALED_FILE,
    mode="w+",
    dtype=np.float32,
    shape=X_test.shape
)

for start in range(
    0,
    test_rows,
    CHUNK_SIZE
):

    end = min(
        start + CHUNK_SIZE,
        test_rows
    )

    chunk = X_test[start:end]

    scaled_chunk = scaler.transform(
        chunk
    ).astype(np.float32)

    X_test_scaled[start:end] = (
        scaled_chunk
    )

    print(
        f"Scaled test rows: "
        f"{end:,}/{test_rows:,}"
    )

X_test_scaled.flush()

del X_test_scaled

# STEP 7 — FINAL VERIFICATION

print("\n[7/7] Final verification...")

X_train_scaled = np.load(
    X_TRAIN_SCALED_FILE,
    mmap_mode="r"
)

X_test_scaled = np.load(
    X_TEST_SCALED_FILE,
    mmap_mode="r"
)

print("\nScaled training shape:")
print(X_train_scaled.shape)

print("\nScaled testing shape:")
print(X_test_scaled.shape)

print("\nNumber of features:")
print(X_train_scaled.shape[1])

print("\nNumber of training samples:")
print(X_train_scaled.shape[0])

print("\nNumber of testing samples:")
print(X_test_scaled.shape[0])


# SAMPLE STATISTICS

sample_size = min(
    10000,
    X_train_scaled.shape[0]
)

sample = np.asarray(
    X_train_scaled[:sample_size]
)

sample_mean = np.mean(
    sample,
    axis=0
)

sample_std = np.std(
    sample,
    axis=0
)

print("\nSample mean range:")
print(
    f"{sample_mean.min():.4f} "
    f"to "
    f"{sample_mean.max():.4f}"
)

print("\nSample standard deviation range:")
print(
    f"{sample_std.min():.4f} "
    f"to "
    f"{sample_std.max():.4f}"
)


# FINAL SUMMARY

print("\n" + "=" * 75)
print("FINAL PREPARATION COMPLETE")
print("=" * 75)

print("\nDataset:")
print("CIC-IDS2017")

print("\nTraining samples:")
print(f"{X_train_scaled.shape[0]:,}")

print("\nTesting samples:")
print(f"{X_test_scaled.shape[0]:,}")

print("\nFeatures:")
print(X_train_scaled.shape[1])

print("\nClasses:")
print(len(label_to_id))

print("\nFiles ready for training:")

print(
    "\nX_train_scaled.npy"
)

print(
    "y_train.npy"
)

print(
    "X_test_scaled.npy"
)

print(
    "y_test.npy"
)

print(
    "label_mapping.json"
)

print(
    "standard_scaler.pkl"
)

print("\n" + "=" * 75)
print("READY FOR MODEL TRAINING")
print("=" * 75)
