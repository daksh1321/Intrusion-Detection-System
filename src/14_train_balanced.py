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

X_TRAIN_FILE = os.path.join(
    DATA_DIR, "X_train_scaled.npy"
)

Y_TRAIN_FILE = os.path.join(
    DATA_DIR, "y_train.npy"
)

X_TEST_FILE = os.path.join(
    DATA_DIR, "X_test_scaled.npy"
)

Y_TEST_FILE = os.path.join(
    DATA_DIR, "y_test.npy"
)

LABEL_FILE = os.path.join(
    DATA_DIR, "label_mapping.json"
)

MODEL_FILE = os.path.join(
    DATA_DIR, "IDS_Balanced_SGD_model.pkl"
)

CONFUSION_FILE = os.path.join(
    DATA_DIR, "IDS_Balanced_confusion_matrix.npy"
)

BATCH_SIZE = 50_000
EPOCHS = 3


print("\n" + "=" * 75)
print("CIC-IDS2017 — BALANCED IDS MODEL")
print("=" * 75)

# 1. LOAD DATA

print("\nLoading memory-mapped data...")

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

print("\nTraining:", X_train.shape)
print("Testing :", X_test.shape)

# 2. LOAD LABEL MAPPING

with open(
    LABEL_FILE,
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

print(
    "\nNumber of classes:",
    len(classes)
)

# 3. CALCULATE CLASS WEIGHTS

print("\n" + "=" * 75)
print("CALCULATING CLASS WEIGHTS")
print("=" * 75)

class_counts = np.bincount(
    y_train,
    minlength=len(classes)
)

total_samples = len(y_train)
number_of_classes = len(classes)

# Balanced weighting:
#
# weight = total_samples /
#          (number_of_classes * class_count)

class_weights = {}

for class_id in classes:

    count = class_counts[class_id]

    if count > 0:

        weight = (
            total_samples /
            (number_of_classes * count)
        )

    else:

        weight = 1.0

    class_weights[class_id] = weight


print("\nClass distribution and weights:")

for class_id in classes:

    print(
        f"{class_id:2d} "
        f"{id_to_label[class_id]:35s} "
        f"count={class_counts[class_id]:10,} "
        f"weight={class_weights[class_id]:8.3f}"
    )

# 4. CREATE MODEL

print("\n" + "=" * 75)
print("CREATING BALANCED MODEL")
print("=" * 75)

model = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=0.0001,
    learning_rate="optimal",
    random_state=42
)

print(model)

# 5. TRAIN MODEL

print("\n" + "=" * 75)
print("TRAINING")
print("=" * 75)

total_train = X_train.shape[0]

rng = np.random.default_rng(42)

for epoch in range(EPOCHS):

    print(
        f"\n========== EPOCH "
        f"{epoch + 1}/{EPOCHS} =========="
    )

    # Shuffle all training indices
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

        # Create sample weights for this batch
        sample_weights = np.array(
            [
                class_weights[int(label)]
                for label in y_batch
            ],
            dtype=np.float32
        )

        model.partial_fit(
            X_batch,
            y_batch,
            classes=classes,
            sample_weight=sample_weights
        )

        print(
            f"Epoch {epoch + 1}: "
            f"{end:,}/{total_train:,}",
            end="\r"
        )

    print(
        f"\nEpoch {epoch + 1} complete."
    )

# 6. SAVE MODEL

joblib.dump(
    model,
    MODEL_FILE
)

print("\n" + "=" * 75)
print("MODEL SAVED")
print("=" * 75)

print(MODEL_FILE)


# 7. PREDICT TEST DATA

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

print("\nPrediction complete.")


# 8. CALCULATE METRICS

accuracy = accuracy_score(
    y_test,
    predictions
)

weighted_precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
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

macro_precision = precision_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

# 9. OVERALL RESULTS

print("\n" + "=" * 75)
print("BALANCED MODEL RESULTS")
print("=" * 75)

print(
    f"\nAccuracy          : {accuracy:.4f}"
)

print(
    f"Weighted Precision : {weighted_precision:.4f}"
)

print(
    f"Weighted Recall    : {weighted_recall:.4f}"
)

print(
    f"Weighted F1        : {weighted_f1:.4f}"
)

print(
    f"\nMacro Precision    : {macro_precision:.4f}"
)

print(
    f"Macro Recall       : {macro_recall:.4f}"
)

print(
    f"Macro F1           : {macro_f1:.4f}"
)


# 10. CLASSIFICATION REPORT

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

# 11. CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    predictions,
    labels=classes
)

np.save(
    CONFUSION_FILE,
    cm
)

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print(cm)

print("\nSaved:")
print(CONFUSION_FILE)


# 12. FINAL SUMMARY

print("\n" + "=" * 75)
print("BALANCED TRAINING COMPLETE")
print("=" * 75)

print(
    f"\nTraining samples used: "
    f"{X_train.shape[0]:,}"
)

print(
    f"Testing samples used: "
    f"{X_test.shape[0]:,}"
)

print(
    f"Features: "
    f"{X_train.shape[1]}"
)

print(
    f"Classes: "
    f"{len(classes)}"
)

print("\nModel:")
print(MODEL_FILE)

print("\n" + "=" * 75)
print("COMPARE THIS WITH THE BASELINE")
print("=" * 75)

print(
    "\nBaseline Macro F1 : 0.3426"
)

print(
    f"Balanced Macro F1: {macro_f1:.4f}"
)

if macro_f1 > 0.3426:

    print(
        "\nRESULT: Balanced model improved Macro F1."
    )

else:

    print(
        "\nRESULT: Balanced model did not improve Macro F1."
    )