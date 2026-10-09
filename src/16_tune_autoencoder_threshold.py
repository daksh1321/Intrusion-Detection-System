import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

DATA_DIR = r"E:\IDS_Project\data\processed"

MODEL_FILE = os.path.join(
    DATA_DIR,
    "IDS_Autoencoder.keras"
)

X_TEST_FILE = os.path.join(
    DATA_DIR,
    "X_test_scaled.npy"
)

Y_TEST_FILE = os.path.join(
    DATA_DIR,
    "y_test.npy"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "autoencoder_threshold_results.json"
)

BATCH_SIZE = 512

BENIGN_LABEL = 0


print("\n" + "=" * 75)
print("AUTOENCODER THRESHOLD TUNING")
print("=" * 75)

# 1. LOAD MODEL

print("\nLoading Autoencoder...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("Autoencoder loaded successfully.")


# 2. LOAD TEST DATA

print("\nLoading test data...")

X_test = np.load(
    X_TEST_FILE,
    mmap_mode="r"
)

y_test = np.load(
    Y_TEST_FILE,
    mmap_mode="r"
)

print(
    "X_test shape:",
    X_test.shape
)

print(
    "y_test shape:",
    y_test.shape
)

# 3. CALCULATE RECONSTRUCTION ERRORS

print("\n" + "=" * 75)
print("CALCULATING RECONSTRUCTION ERRORS")
print("=" * 75)

test_errors = np.empty(
    len(y_test),
    dtype=np.float32
)


for start in range(
    0,
    len(y_test),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(y_test)
    )

    batch = np.asarray(
        X_test[start:end],
        dtype=np.float32
    )

    reconstructed = model.predict(
        batch,
        verbose=0
    )

    errors = np.mean(
        np.square(
            batch - reconstructed
        ),
        axis=1
    )

    test_errors[start:end] = errors

    print(
        f"Processed "
        f"{end:,}/{len(y_test):,}",
        end="\r"
    )


print("\n\nReconstruction errors calculated.")


# 4. CONVERT TEST LABELS TO BINARY

# BENIGN = 0
# Any attack = 1

y_true = (
    y_test != BENIGN_LABEL
).astype(np.int8)

# 5. THRESHOLDS TO TEST

thresholds = [
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.12,
    0.15,
    0.18,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50
]

# 6. TEST EACH THRESHOLD

print("\n" + "=" * 75)
print("THRESHOLD COMPARISON")
print("=" * 75)

print(
    "\n"
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FP':<12}"
    f"{'FN':<12}"
)

print("-" * 75)


results = []


for threshold in thresholds:

    y_pred = (
        test_errors > threshold
    ).astype(np.int8)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    print(
        f"{threshold:<12.3f}"
        f"{accuracy:<12.4f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fp:<12,}"
        f"{fn:<12,}"
    )

    results.append({
        "threshold": threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    })

# 7. FIND BEST THRESHOLD BY F1

best_f1_result = max(
    results,
    key=lambda x: x["f1"]
)

# 8. FIND BEST THRESHOLD BY RECALL
best_recall_result = max(
    results,
    key=lambda x: x["recall"]
)

# 9. FIND BALANCED THRESHOLD


print("\n" + "=" * 75)
print("BEST THRESHOLD BY F1")
print("=" * 75)

print(
    "\nThreshold:",
    best_f1_result["threshold"]
)

print(
    "Accuracy:",
    f"{best_f1_result['accuracy']:.4f}"
)

print(
    "Precision:",
    f"{best_f1_result['precision']:.4f}"
)

print(
    "Recall:",
    f"{best_f1_result['recall']:.4f}"
)

print(
    "F1:",
    f"{best_f1_result['f1']:.4f}"
)

print(
    "False Positives:",
    f"{best_f1_result['false_positive']:,}"
)

print(
    "False Negatives:",
    f"{best_f1_result['false_negative']:,}"
)

# 10. BEST RECALL

print("\n" + "=" * 75)
print("BEST THRESHOLD BY RECALL")
print("=" * 75)

print(
    "\nThreshold:",
    best_recall_result["threshold"]
)

print(
    "Accuracy:",
    f"{best_recall_result['accuracy']:.4f}"
)

print(
    "Precision:",
    f"{best_recall_result['precision']:.4f}"
)

print(
    "Recall:",
    f"{best_recall_result['recall']:.4f}"
)

print(
    "F1:",
    f"{best_recall_result['f1']:.4f}"
)

# 11. ROC-AUC

roc_auc = roc_auc_score(
    y_true,
    test_errors
)

print("\n" + "=" * 75)
print("ROC-AUC")
print("=" * 75)

print(
    f"\nROC-AUC: {roc_auc:.4f}"
)

# 12. SAVE RESULTS

output = {
    "roc_auc": float(roc_auc),
    "best_f1": best_f1_result,
    "best_recall": best_recall_result,
    "all_thresholds": results
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=4
    )

print("\n" + "=" * 75)
print("THRESHOLD TUNING COMPLETE")
print("=" * 75)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)