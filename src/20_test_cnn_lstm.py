
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import gc
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE = r"E:\IDS_Project\data\processed"

MODEL_PATH = os.path.join(BASE, "IDS_CNN_LSTM.keras")
X_PATH = os.path.join(BASE, "X_test_scaled.npy")
Y_PATH = os.path.join(BASE, "y_test.npy")
LABEL_PATH = os.path.join(BASE, "label_mapping.json")

RESULT_PATH = os.path.join(BASE, "cnn_lstm_test_results.json")
CM_PATH = os.path.join(BASE, "CNN_LSTM_test_confusion_matrix.npy")

SEQ_LEN = 10
FEATURES = 70
BATCH_SIZE = 1000
print("Loading label mapping...")

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

if "id_to_label" in mapping_data:
    labels = {int(k): v for k, v in mapping_data["id_to_label"].items()}
elif "label_to_id" in mapping_data:
    labels = {int(v): k for k, v in mapping_data["label_to_id"].items()}
else:
    labels = {int(k): v for k, v in mapping_data.items()}

X = np.load(X_PATH, mmap_mode="r")
y = np.load(Y_PATH, mmap_mode="r")

if X.ndim != 2 or X.shape[1] != FEATURES:
    raise ValueError(f"Expected test features shaped (N, {FEATURES}); got {X.shape}")

if len(X) != len(y) or len(X) < SEQ_LEN:
    raise ValueError("Invalid test data dimensions.")

n_sequences = len(X) - SEQ_LEN + 1
print(f"Test flows: {len(X):,}")
print(f"Sequences to evaluate: {n_sequences:,}")

print("Loading CNN-LSTM...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Model input:", model.input_shape)
print("Model output:", model.output_shape)

if tuple(model.input_shape[1:]) != (SEQ_LEN, FEATURES):
    raise ValueError(f"Unexpected model input shape: {model.input_shape}")

y_pred = np.empty(n_sequences, dtype=np.int8)

print("\nPredicting in batches...")

for start in range(0, n_sequences, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n_sequences)
    count = end - start
    batch = np.empty((count, SEQ_LEN, FEATURES), dtype=np.float32)

    for j in range(count):
        pos = start + j
        batch[j] = X[pos:pos + SEQ_LEN]

    probabilities = model.predict(
        batch,
        batch_size=256,
        verbose=0
    )

    y_pred[start:end] = np.argmax(probabilities, axis=1).astype(np.int8)

    del batch, probabilities
    gc.collect()

    print(f"Processed {end:,}/{n_sequences:,} ({100 * end / n_sequences:.1f}%)")

y_true = np.asarray(y[SEQ_LEN - 1:], dtype=np.int8)
class_ids = sorted(labels.keys())
target_names = [labels[i] for i in class_ids]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

print("\n" + "=" * 65)
print("CNN-LSTM TEST RESULTS")
print("=" * 65)
print(f"Accuracy:          {accuracy:.4f} ({accuracy:.2%})")
print(f"Weighted precision: {precision:.4f}")
print(f"Weighted recall:    {recall:.4f}")
print(f"Weighted F1:        {weighted_f1:.4f}")
print(f"Macro F1:           {macro_f1:.4f}")

report = classification_report(
    y_true,
    y_pred,
    labels=class_ids,
    target_names=target_names,
    digits=4,
    zero_division=0
)

print("\nCLASSIFICATION REPORT\n")
print(report)

cm = confusion_matrix(y_true, y_pred, labels=class_ids)
np.save(CM_PATH, cm)

results = {
    "model": "CNN-LSTM",
    "test_flows": int(len(X)),
    "test_sequences": int(n_sequences),
    "sequence_length": SEQ_LEN,
    "features": FEATURES,
    "accuracy": float(accuracy),
    "weighted_precision": float(precision),
    "weighted_recall": float(recall),
    "weighted_f1": float(weighted_f1),
    "macro_f1": float(macro_f1),
    "classification_report": report
}

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nSaved confusion matrix:", CM_PATH)
print("Saved metrics:", RESULT_PATH)
print("\nTESTING COMPLETED.")
