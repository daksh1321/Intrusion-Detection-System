import os
import json
import joblib
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


DATA_DIR = r"E:\IDS_Project\data\processed"

X_TRAIN = os.path.join(
    DATA_DIR,
    "X_train_scaled.npy"
)

Y_TRAIN = os.path.join(
    DATA_DIR,
    "y_train.npy"
)

X_TEST = os.path.join(
    DATA_DIR,
    "X_test_scaled.npy"
)

Y_TEST = os.path.join(
    DATA_DIR,
    "y_test.npy"
)

LABEL_MAPPING = os.path.join(
    DATA_DIR,
    "label_mapping.json"
)

MODEL_FILE = os.path.join(
    DATA_DIR,
    "IDS_SGD_model.pkl"
)

CONFUSION_MATRIX_FILE = os.path.join(
    DATA_DIR,
    "IDS_confusion_matrix.npy"
)

BATCH_SIZE = 50_000
EPOCHS = 3


print("\n" + "=" * 75)
print("CIC-IDS2017 — IDS MODEL TRAINING")
print("=" * 75)


# 1. LOAD DATA

print("\nLoading data...")

X_train = np.load(
    X_TRAIN,
    mmap_mode="r"
)

y_train = np.load(
    Y_TRAIN,
    mmap_mode="r"
)

X_test = np.load(
    X_TEST,
    mmap_mode="r"
)

y_test = np.load(
    Y_TEST,
    mmap_mode="r"
)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# 2. LOAD LABEL MAPPING

with open(
    LABEL_MAPPING,
    "r",
    encoding="utf-8"
) as f:

    mapping = json.load(f)

id_to_label = {
    int(k): v
    for k, v in mapping["id_to_label"].items()
}

classes = np.array(
    sorted(id_to_label.keys())
)

class_names = [
    id_to_label[i]
    for i in classes
]

print("\nNumber of classes:", len(classes))

print("\nClasses:")

for i in classes:
    print(
        f"{i:2d} -> {id_to_label[i]}"
    )


# 3. CREATE MODEL

print("\n" + "=" * 75)
print("CREATING MODEL")
print("=" * 75)

model = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=0.0001,
    learning_rate="optimal",
    random_state=42
)

print("\nModel:")
print(model)

# 4. TRAIN MODEL

print("\n" + "=" * 75)
print("TRAINING MODEL")
print("=" * 75)

total_train = X_train.shape[0]

rng = np.random.default_rng(42)

for epoch in range(EPOCHS):

    print(
        f"\n========== EPOCH "
        f"{epoch + 1}/{EPOCHS} =========="
    )

    # Shuffle the training rows
    indices = rng.permutation(
        total_train
    )

    for start in range(
        0,
        total_train,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total_train
        )

        batch_indices = indices[start:end]

        X_batch = X_train[
            batch_indices
        ]

        y_batch = y_train[
            batch_indices
        ]

        model.partial_fit(
            X_batch,
            y_batch,
            classes=classes
        )

        print(
            f"Epoch {epoch + 1}: "
            f"{end:,}/{total_train:,}",
            end="\r"
        )

    print(
        f"\nEpoch {epoch + 1} completed."
    )

# 5. SAVE MODEL
joblib.dump(
    model,
    MODEL_FILE
)

print("\n" + "=" * 75)
print("MODEL SAVED")
print("=" * 75)

print(
    "\nLocation:"
)

print(MODEL_FILE)

# 6. PREDICTION

print("\n" + "=" * 75)
print("PREDICTING TEST DATA")
print("=" * 75)

total_test = X_test.shape[0]

predictions = np.empty(
    total_test,
    dtype=np.int8
)

for start in range(
    0,
    total_test,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        total_test
    )

    predictions[start:end] = model.predict(
        X_test[start:end]
    )

    print(
        f"Predicted: "
        f"{end:,}/{total_test:,}",
        end="\r"
    )

print("\nPrediction completed.")

# 7. METRICS

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

# 8. DISPLAY RESULTS

print("\n" + "=" * 75)
print("FINAL MODEL RESULTS")
print("=" * 75)

print(
    f"\nAccuracy       : {accuracy:.4f}"
)

print(
    f"Weighted F1    : {weighted_f1:.4f}"
)

print(
    f"Macro F1       : {macro_f1:.4f}"
)

print(
    f"Precision      : {precision:.4f}"
)

print(
    f"Recall         : {recall:.4f}"
)

# 9. CLASSIFICATION REPORT

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

report = classification_report(
    y_test,
    predictions,
    labels=classes,
    target_names=class_names,
    zero_division=0
)

print(report)

# 10. CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    predictions,
    labels=classes
)

np.save(
    CONFUSION_MATRIX_FILE,
    cm
)

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print(cm)

print(
    "\nConfusion matrix saved:"
)

print(CONFUSION_MATRIX_FILE)

# COMPLETE

print("\n" + "=" * 75)
print("IDS TRAINING COMPLETE")
print("=" * 75)

print(
    "\nTraining samples used:"
)

print(
    f"{X_train.shape[0]:,}"
)

print(
    "\nTesting samples used:"
)

print(
    f"{X_test.shape[0]:,}"
)

print(
    "\nFeatures:"
)

print(
    X_train.shape[1]
)

print(
    "\nClasses:"
)

print(
    len(classes)
)

print("\nModel:")
print(MODEL_FILE)

print("\n" + "=" * 75)
print("NEXT: AUTOENCODER + CNN-LSTM")
print("=" * 75)