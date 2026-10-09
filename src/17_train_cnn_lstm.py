import os
import json
import gc
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

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
    "IDS_CNN_LSTM.keras"
)

RESULT_FILE = os.path.join(
    DATA_DIR,
    "cnn_lstm_results.json"
)

CONFUSION_FILE = os.path.join(
    DATA_DIR,
    "CNN_LSTM_confusion_matrix.npy"
)


N_FEATURES = 70

SEQUENCE_LENGTH = 10

BATCH_SIZE = 256

EPOCHS = 15

NUM_CLASSES = 15

RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)

tf.random.set_seed(
    RANDOM_SEED
)

try:

    tf.config.threading.set_inter_op_parallelism_threads(2)

    tf.config.threading.set_intra_op_parallelism_threads(4)

except Exception:

    pass


print("\n" + "=" * 80)
print("CIC-IDS2017 CNN-LSTM INTRUSION DETECTION SYSTEM")
print("=" * 80)

# 2. LOAD DATA USING MEMORY MAPPING

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


print("\nTraining data:")
print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print("\nTesting data:")

print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)

# 3. VERIFY FEATURES

if X_train.shape[1] != N_FEATURES:

    raise ValueError(
        f"Expected {N_FEATURES} features, "
        f"but found {X_train.shape[1]}"
    )

# 4. LOAD LABEL MAPPING

print("\nLoading label mapping...")


with open(
    LABEL_FILE,
    "r",
    encoding="utf-8"
) as f:

    label_mapping_raw = json.load(f)


if "id_to_label" in label_mapping_raw:

    label_mapping = {
        int(k): v
        for k, v in label_mapping_raw["id_to_label"].items()
    }

elif "label_to_id" in label_mapping_raw:

    label_mapping = {
        int(v): k
        for k, v in label_mapping_raw["label_to_id"].items()
    }

else:

    label_mapping = {
        int(k): v
        for k, v in label_mapping_raw.items()
    }


print("\nClasses:")

for label_id in sorted(label_mapping):

    print(
        f"{label_id:2d} -> "
        f"{label_mapping[label_id]}"
    )

# 5. CHECK NUMBER OF CLASSES

if len(label_mapping) != NUM_CLASSES:

    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"found {len(label_mapping)}"
    )

# 6. SEQUENCE GENERATOR

print("\n" + "=" * 80)
print("CREATING SEQUENCE GENERATORS")
print("=" * 80)


class SequenceGenerator(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        X,
        y,
        sequence_length,
        batch_size,
        shuffle=True
    ):

        self.X = X

        self.y = y

        self.sequence_length = (
            sequence_length
        )

        self.batch_size = batch_size

        self.shuffle = shuffle

        # Number of valid sequences
        self.num_sequences = (
            len(X)
            - sequence_length
            + 1
        )

        self.indices = np.arange(
            self.num_sequences,
            dtype=np.int64
        )

        self.on_epoch_end()


    def __len__(self):

        return int(
            np.ceil(
                self.num_sequences
                /
                self.batch_size
            )
        )


    def __getitem__(
        self,
        batch_number
    ):

        start = (
            batch_number
            *
            self.batch_size
        )

        end = min(
            start + self.batch_size,
            self.num_sequences
        )

        batch_indices = (
            self.indices[start:end]
        )

        actual_batch_size = (
            len(batch_indices)
        )

        X_batch = np.empty(
            (
                actual_batch_size,
                self.sequence_length,
                N_FEATURES
            ),
            dtype=np.float32
        )

        y_batch = np.empty(
            actual_batch_size,
            dtype=np.int8
        )


        for i, sequence_start in enumerate(
            batch_indices
        ):

            sequence_end = (
                sequence_start
                +
                self.sequence_length
            )

            X_batch[i] = np.asarray(
                self.X[
                    sequence_start:
                    sequence_end
                ],
                dtype=np.float32
            )

            # Label = final flow in sequence
            y_batch[i] = self.y[
                sequence_end - 1
            ]


        return (
            X_batch,
            y_batch
        )


    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )

# 7. CREATE TRAIN/VALIDATION SPLIT

print("\nCreating train/validation split")
total_train_sequences = (
    len(X_train)
    -
    SEQUENCE_LENGTH
    +
    1
)

validation_ratio = 0.10

validation_sequences = int(
    total_train_sequences
    *
    validation_ratio
)

training_sequences = (
    total_train_sequences
    -
    validation_sequences
)


print(
    "\nTotal possible training sequences:",
    f"{total_train_sequences:,}"
)

print(
    "Training sequences:",
    f"{training_sequences:,}"
)

print(
    "Validation sequences:",
    f"{validation_sequences:,}"
)

# 8. CUSTOM GENERATOR FOR RANGE

class RangeSequenceGenerator(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        X,
        y,
        start_sequence,
        number_sequences,
        sequence_length,
        batch_size,
        shuffle=True
    ):

        self.X = X

        self.y = y

        self.start_sequence = (
            start_sequence
        )

        self.number_sequences = (
            number_sequences
        )

        self.sequence_length = (
            sequence_length
        )

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.indices = np.arange(
            self.number_sequences,
            dtype=np.int64
        )

        self.on_epoch_end()


    def __len__(self):

        return int(
            np.ceil(
                self.number_sequences
                /
                self.batch_size
            )
        )


    def __getitem__(
        self,
        batch_number
    ):

        start = (
            batch_number
            *
            self.batch_size
        )

        end = min(
            start + self.batch_size,
            self.number_sequences
        )

        local_indices = (
            self.indices[start:end]
        )

        actual_batch_size = (
            len(local_indices)
        )

        X_batch = np.empty(
            (
                actual_batch_size,
                self.sequence_length,
                N_FEATURES
            ),
            dtype=np.float32
        )

        y_batch = np.empty(
            actual_batch_size,
            dtype=np.int8
        )


        for i, local_index in enumerate(
            local_indices
        ):

            sequence_start = (
                self.start_sequence
                +
                local_index
            )

            sequence_end = (
                sequence_start
                +
                self.sequence_length
            )

            X_batch[i] = np.asarray(
                self.X[
                    sequence_start:
                    sequence_end
                ],
                dtype=np.float32
            )

            y_batch[i] = self.y[
                sequence_end - 1
            ]


        return (
            X_batch,
            y_batch
        )


    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )

# 9. CREATE GENERATORS

train_generator = RangeSequenceGenerator(
    X=X_train,
    y=y_train,
    start_sequence=0,
    number_sequences=training_sequences,
    sequence_length=SEQUENCE_LENGTH,
    batch_size=BATCH_SIZE,
    shuffle=True
)


validation_generator = RangeSequenceGenerator(
    X=X_train,
    y=y_train,
    start_sequence=training_sequences,
    number_sequences=validation_sequences,
    sequence_length=SEQUENCE_LENGTH,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nTrain batches:")
print(
    len(train_generator)
)

print("\nValidation batches:")
print(
    len(validation_generator)
)

# 10. BUILD CNN-LSTM

print("\n" + "=" * 80)
print("BUILDING CNN-LSTM MODEL")
print("=" * 80)


inputs = Input(
    shape=(
        SEQUENCE_LENGTH,
        N_FEATURES
    )
)

# CNN FEATURE EXTRACTION

x = Conv1D(
    filters=64,
    kernel_size=3,
    padding="same",
    activation="relu"
)(inputs)

x = BatchNormalization()(x)

x = Conv1D(
    filters=64,
    kernel_size=3,
    padding="same",
    activation="relu"
)(x)

x = MaxPooling1D(
    pool_size=2
)(x)

# LSTM TEMPORAL LEARNING

x = LSTM(
    64,
    return_sequences=False
)(x)

# CLASSIFIER

x = Dense(
    64,
    activation="relu"
)(x)

x = Dropout(
    0.3
)(x)

outputs = Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = Model(
    inputs=inputs,
    outputs=outputs
)

# 11. COMPILE

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)


model.summary()

# 12. CALLBACKS

checkpoint = ModelCheckpoint(
    MODEL_FILE,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    min_lr=1e-6,
    verbose=1
)

# 13. TRAIN

print("\n" + "=" * 80)
print("TRAINING CNN-LSTM")
print("=" * 80)

print(
    "\nSequence length:",
    SEQUENCE_LENGTH
)

print(
    "Features per timestep:",
    N_FEATURES
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Training sequences:",
    f"{training_sequences:,}"
)


history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],
    verbose=1
)

# 14. LOAD BEST MODEL

print("\n" + "=" * 80)
print("LOADING BEST CNN-LSTM")
print("=" * 80)


model = tf.keras.models.load_model(
    MODEL_FILE
)

# 15. TEST GENERATOR

test_sequences = (
    len(X_test)
    -
    SEQUENCE_LENGTH
    +
    1
)


test_generator = RangeSequenceGenerator(
    X=X_test,
    y=y_test,
    start_sequence=0,
    number_sequences=test_sequences,
    sequence_length=SEQUENCE_LENGTH,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print(
    "\nTest sequences:",
    f"{test_sequences:,}"
)

# 16. PREDICTION

print("\n" + "=" * 80)
print("PREDICTING TEST DATA")
print("=" * 80)


predicted_probabilities = model.predict(
    test_generator,
    verbose=1
)


y_pred = np.argmax(
    predicted_probabilities,
    axis=1
)


# Actual label corresponds to final flow
y_true = np.asarray(
    y_test[
        SEQUENCE_LENGTH - 1:
    ],
    dtype=np.int8
)

# 17. METRICS

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_macro = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

precision_macro = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


# 18. PRINT RESULTS

print("\n" + "=" * 80)
print("CNN-LSTM RESULTS")
print("=" * 80)


print(
    f"\nAccuracy          : {accuracy:.4f}"
)

print(
    f"Weighted Precision: {precision:.4f}"
)

print(
    f"Weighted Recall   : {recall:.4f}"
)

print(
    f"Weighted F1       : {f1_weighted:.4f}"
)

print(
    f"Macro Precision   : {precision_macro:.4f}"
)

print(
    f"Macro Recall      : {recall_macro:.4f}"
)

print(
    f"Macro F1          : {f1_macro:.4f}"
)

# 19. CLASSIFICATION REPORT

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)


class_names = [
    label_mapping[i]
    for i in range(NUM_CLASSES)
]


print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=class_names,
        zero_division=0
    )
)

# 20. CONFUSION MATRIX

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)


print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(cm)


np.save(
    CONFUSION_FILE,
    cm
)

# 21. SAVE RESULTS

results = {

    "model": "CNN-LSTM",

    "sequence_length": int(
        SEQUENCE_LENGTH
    ),

    "features": int(
        N_FEATURES
    ),

    "batch_size": int(
        BATCH_SIZE
    ),

    "epochs_requested": int(
        EPOCHS
    ),

    "training_sequences": int(
        training_sequences
    ),

    "validation_sequences": int(
        validation_sequences
    ),

    "test_sequences": int(
        test_sequences
    ),

    "accuracy": float(
        accuracy
    ),

    "weighted_precision": float(
        precision
    ),

    "weighted_recall": float(
        recall
    ),

    "weighted_f1": float(
        f1_weighted
    ),

    "macro_precision": float(
        precision_macro
    ),

    "macro_recall": float(
        recall_macro
    ),

    "macro_f1": float(
        f1_macro
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

# 22. CLEAN MEMORY

del predicted_probabilities

del train_generator

del validation_generator

del test_generator

gc.collect()

# COMPLETE

print("\n" + "=" * 80)
print("CNN-LSTM TRAINING COMPLETE")
print("=" * 80)

print("\nModel:")
print(
    MODEL_FILE
)

print("\nResults:")
print(
    RESULT_FILE
)

print("\nConfusion Matrix:")
print(
    CONFUSION_FILE
)

print("\n" + "=" * 80)
print("IDS PIPELINE COMPLETE")
print("=" * 80)