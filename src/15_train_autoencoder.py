import os
import json
import joblib
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)



DATA_DIR = r"E:\IDS_Project\data\processed"

X_TRAIN_FILE = os.path.join(
    DATA_DIR,
    "X_train_scaled.npy"
)

Y_TRAIN_FILE = os.path.join(
    DATA_DIR,
    "y_train.npy"
)

X_TEST_FILE = os.path.join(
    DATA_DIR,
    "X_test_scaled.npy"
)

Y_TEST_FILE = os.path.join(
    DATA_DIR,
    "y_test.npy"
)

LABEL_FILE = os.path.join(
    DATA_DIR,
    "label_mapping.json"
)

MODEL_FILE = os.path.join(
    DATA_DIR,
    "IDS_Autoencoder.keras"
)

THRESHOLD_FILE = os.path.join(
    DATA_DIR,
    "autoencoder_threshold.npy"
)

RESULT_FILE = os.path.join(
    DATA_DIR,
    "autoencoder_results.json"
)

FEATURES = 70

BATCH_SIZE = 512

EPOCHS = 20

THRESHOLD_SAMPLES = 100_000

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)



print("\n" + "=" * 75)
print("CIC-IDS2017 — AUTOENCODER ANOMALY DETECTION")
print("=" * 75)

# 1. LOAD DATA

print("\nLoading memory-mapped arrays...")

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

print("\nX_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nX_test:", X_test.shape)
print("y_test:", y_test.shape)

# 2. FIND BENIGN TRAINING SAMPLES

print("\n" + "=" * 75)
print("FINDING BENIGN TRAINING DATA")
print("=" * 75)

BENIGN_LABEL = 0

benign_indices = np.flatnonzero(
    y_train == BENIGN_LABEL
)

print(
    "\nTotal BENIGN training samples:"
)

print(
    f"{len(benign_indices):,}"
)


# 3. CREATE AUTOENCODER

print("\n" + "=" * 75)
print("BUILDING AUTOENCODER")
print("=" * 75)


input_layer = Input(
    shape=(FEATURES,)
)

# Encoder
encoded = Dense(
    64,
    activation="relu"
)(input_layer)

encoded = Dense(
    32,
    activation="relu"
)(encoded)

encoded = Dense(
    16,
    activation="relu",
    name="latent_space"
)(encoded)

# Decoder

decoded = Dense(
    32,
    activation="relu"
)(encoded)

decoded = Dense(
    64,
    activation="relu"
)(decoded)

decoded = Dense(
    FEATURES,
    activation="linear"
)(decoded)


autoencoder = Model(
    input_layer,
    decoded
)


# COMPILE

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)


autoencoder.summary()

# 4. CREATE BENIGN GENERATOR

print("\n" + "=" * 75)
print("CREATING BENIGN DATA GENERATOR")
print("=" * 75)


def benign_generator():

    while True:

        # Shuffle benign indices every epoch
        shuffled_indices = np.random.permutation(
            benign_indices
        )

        for start in range(
            0,
            len(shuffled_indices),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(shuffled_indices)
            )

            batch_indices = (
                shuffled_indices[start:end]
            )

            batch = np.asarray(
                X_train[batch_indices],
                dtype=np.float32
            )

            # Autoencoder target = input
            yield batch, batch


# 5. CALLBACKS

early_stopping = EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_FILE,
    monitor="loss",
    save_best_only=True
)

# 6. TRAIN AUTOENCODER

print("\n" + "=" * 75)
print("TRAINING AUTOENCODER")
print("=" * 75)

steps_per_epoch = int(
    np.ceil(
        len(benign_indices) /
        BATCH_SIZE
    )
)

print(
    "\nBenign samples used per epoch:"
)

print(
    f"{len(benign_indices):,}"
)

print(
    "\nSteps per epoch:"
)

print(
    steps_per_epoch
)

print(
    "\nEpochs:"
)

print(
    EPOCHS
)


history = autoencoder.fit(
    benign_generator(),
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        checkpoint
    ],
    verbose=1
)

# 7. LOAD BEST MODEL

print("\n" + "=" * 75)
print("LOADING BEST AUTOENCODER")
print("=" * 75)

autoencoder = tf.keras.models.load_model(
    MODEL_FILE
)

# 8. CALCULATE BENIGN RECONSTRUCTION ERRORS

print("\n" + "=" * 75)
print("CALCULATING BENIGN RECONSTRUCTION ERROR")
print("=" * 75)


# Select a reproducible subset only for threshold estimation.
# The Autoencoder itself was trained on all benign samples.

rng = np.random.default_rng(
    RANDOM_SEED
)

threshold_count = min(
    THRESHOLD_SAMPLES,
    len(benign_indices)
)

threshold_indices = rng.choice(
    benign_indices,
    size=threshold_count,
    replace=False
)

benign_errors = []

for start in range(
    0,
    len(threshold_indices),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(threshold_indices)
    )

    indices = threshold_indices[
        start:end
    ]

    batch = np.asarray(
        X_train[indices],
        dtype=np.float32
    )

    reconstructed = autoencoder.predict(
        batch,
        verbose=0
    )

    errors = np.mean(
        np.square(
            batch - reconstructed
        ),
        axis=1
    )

    benign_errors.extend(
        errors
    )

benign_errors = np.asarray(
    benign_errors
)

# 9. DETERMINE THRESHOLD

print("\n" + "=" * 75)
print("DETERMINING ANOMALY THRESHOLD")
print("=" * 75)


# 99th percentile of benign reconstruction error

threshold = np.percentile(
    benign_errors,
    99
)

np.save(
    THRESHOLD_FILE,
    np.array([threshold])
)

print(
    "\nThreshold:",
    threshold
)

print(
    "\nMean benign error:",
    benign_errors.mean()
)

print(
    "Maximum benign error:",
    benign_errors.max()
)


# 10. TEST AUTOENCODER

print("\n" + "=" * 75)
print("EVALUATING AUTOENCODER")
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

    reconstructed = autoencoder.predict(
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
        f"Processed: "
        f"{end:,}/{len(y_test):,}",
        end="\r"
    )

print("\nTest reconstruction complete.")


# 11. CONVERT TO BINARY IDS

# BENIGN = 0
# ATTACK = 1

y_test_binary = (
    y_test != BENIGN_LABEL
).astype(np.int8)


y_pred_binary = (
    test_errors > threshold
).astype(np.int8)

# 12. METRICS

accuracy = accuracy_score(
    y_test_binary,
    y_pred_binary
)

precision = precision_score(
    y_test_binary,
    y_pred_binary,
    zero_division=0
)

recall = recall_score(
    y_test_binary,
    y_pred_binary,
    zero_division=0
)

f1 = f1_score(
    y_test_binary,
    y_pred_binary,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test_binary,
    test_errors
)


# 13. RESULTS

print("\n" + "=" * 75)
print("AUTOENCODER RESULTS")
print("=" * 75)

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# 14. CLASSIFICATION REPORT

print("\n" + "=" * 75)
print("BINARY CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_test_binary,
        y_pred_binary,
        target_names=[
            "BENIGN",
            "ATTACK"
        ],
        zero_division=0
    )
)

# 15. CONFUSION MATRIX

cm = confusion_matrix(
    y_test_binary,
    y_pred_binary
)

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print(cm)

# 16. ATTACK DETECTION RATE

attack_mask = (
    y_test_binary == 1
)

attack_detection_rate = (
    np.sum(
        y_pred_binary[attack_mask] == 1
    )
    /
    np.sum(attack_mask)
)

print(
    "\nAttack Detection Rate:",
    f"{attack_detection_rate:.4f}"
)


results = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "threshold": float(threshold),
    "training_benign_samples": int(
        len(benign_indices)
    ),
    "threshold_samples": int(
        threshold_count
    ),
    "test_samples": int(
        len(y_test)
    )
}


with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\n" + "=" * 75)
print("AUTOENCODER TRAINING COMPLETE")
print("=" * 75)

print("\nModel:")
print(MODEL_FILE)

print("\nThreshold:")
print(THRESHOLD_FILE)

print("\nResults:")
print(RESULT_FILE)

print("\n" + "=" * 75)
print("NEXT: CNN-LSTM TEMPORAL IDS")
print("=" * 75)