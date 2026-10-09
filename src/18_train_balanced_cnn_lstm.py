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

from sklearn.utils.class_weight import compute_class_weight


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
    "IDS_Balanced_CNN_LSTM.keras"
)

RESULT_FILE = os.path.join(
    DATA_DIR,
    "balanced_cnn_lstm_results.json"
)

CONFUSION_FILE = os.path.join(
    DATA_DIR,
    "Balanced_CNN_LSTM_confusion_matrix.npy"
)


N_FEATURES = 70

SEQUENCE_LENGTH = 10

BATCH_SIZE = 256

EPOCHS = 12

NUM_CLASSES = 15

RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

try:

    tf.config.threading.set_inter_op_parallelism_threads(2)

    tf.config.threading.set_intra_op_parallelism_threads(4)

except Exception:

    pass


print("\n" + "=" * 80)
print("CLASS-WEIGHTED CNN-LSTM — CIC-IDS2017")
print("=" * 80)


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


print("\nTraining:")
print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print("\nTesting:")

print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)

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
        for k, v in
        label_mapping_raw["id_to_label"].items()
    }

elif "label_to_id" in label_mapping_raw:

    label_mapping = {
        int(v): k
        for k, v in
        label_mapping_raw["label_to_id"].items()
    }

else:

    label_mapping = {
        int(k): v
        for k, v in
        label_mapping_raw.items()
    }


print("\nClasses:")

for i in range(NUM_CLASSES):

    print(
        f"{i:2d} -> {label_mapping[i]}"
    )


print("\n" + "=" * 80)
print("CALCULATING CLASS WEIGHTS")
print("=" * 80)

class_counts = np.bincount(
    np.asarray(y_train),
    minlength=NUM_CLASSES
)


print("\nTraining class distribution:")

for i in range(NUM_CLASSES):

    print(
        f"{i:2d} | "
        f"{label_mapping[i]:30s} | "
        f"{class_counts[i]:,}"
    )


classes = np.arange(
    NUM_CLASSES
)


class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=np.asarray(y_train)
)


class_weights = {
    int(i): float(
        class_weights_array[i]
    )
    for i in range(NUM_CLASSES)
}


print("\nCalculated class weights:")

for i in range(NUM_CLASSES):

    print(
        f"{i:2d} | "
        f"{label_mapping[i]:30s} | "
        f"{class_weights[i]:.6f}"
    )


class WeightedSequenceGenerator(
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
        class_weights,
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

        self.class_weights = (
            class_weights
        )

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


        sample_weights = np.empty(
            actual_batch_size,
            dtype=np.float32
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


            # Label of final flow
            label = int(
                self.y[
                    sequence_end - 1
                ]
            )


            y_batch[i] = label


            sample_weights[i] = (
                self.class_weights[label]
            )


        return (
            X_batch,
            y_batch,
            sample_weights
        )


    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


print("\n" + "=" * 80)
print("CREATING TRAIN / VALIDATION SEQUENCES")
print("=" * 80)


total_sequences = (
    len(X_train)
    -
    SEQUENCE_LENGTH
    +
    1
)


validation_ratio = 0.10


validation_sequences = int(
    total_sequences
    *
    validation_ratio
)


training_sequences = (
    total_sequences
    -
    validation_sequences
)


print(
    "\nTotal sequences:",
    f"{total_sequences:,}"
)

print(
    "Training sequences:",
    f"{training_sequences:,}"
)

print(
    "Validation sequences:",
    f"{validation_sequences:,}"
)


train_generator = WeightedSequenceGenerator(
    X=X_train,
    y=y_train,
    start_sequence=0,
    number_sequences=training_sequences,
    sequence_length=SEQUENCE_LENGTH,
    batch_size=BATCH_SIZE,
    class_weights=class_weights,
    shuffle=True
)


validation_generator = WeightedSequenceGenerator(
    X=X_train,
    y=y_train,
    start_sequence=training_sequences,
    number_sequences=validation_sequences,
    sequence_length=SEQUENCE_LENGTH,
    batch_size=BATCH_SIZE,
    class_weights=class_weights,
    shuffle=False
)


print(
    "\nTraining batches:",
    len(train_generator)
)

print(
    "Validation batches:",
    len(validation_generator)
)


print("\n" + "=" * 80)
print("BUILDING CLASS-WEIGHTED CNN-LSTM")
print("=" * 80)


inputs = Input(
    shape=(
        SEQUENCE_LENGTH,
        N_FEATURES
    )
)


# CNN

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


# LSTM

x = LSTM(
    64,
    return_sequences=False
)(x)


# Dense classifier

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

print("\n" + "=" * 80)
print("TRAINING CLASS-WEIGHTED CNN-LSTM")
print("=" * 80)


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

print("\n" + "=" * 80)
print("LOADING BEST BALANCED CNN-LSTM")
print("=" * 80)


model = tf.keras.models.load_model(
    MODEL_FILE
)

test_sequences = (
    len(X_test)
    -
    SEQUENCE_LENGTH
    +
    1
)

class TestSequenceGenerator(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        X,
        y,
        sequence_length,
        batch_size
    ):

        self.X = X

        self.y = y

        self.sequence_length = (
            sequence_length
        )

        self.batch_size = batch_size

        self.number_sequences = (
            len(X)
            -
            sequence_length
            +
            1
        )


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

        batch_indices = np.arange(
            start,
            end
        )


        X_batch = np.empty(
            (
                len(batch_indices),
                self.sequence_length,
                N_FEATURES
            ),
            dtype=np.float32
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


        return X_batch


test_generator = TestSequenceGenerator(
    X=X_test,
    y=y_test,
    sequence_length=SEQUENCE_LENGTH,
    batch_size=BATCH_SIZE
)

print("\n" + "=" * 80)
print("PREDICTING TEST DATA")
print("=" * 80)


probabilities = model.predict(
    test_generator,
    verbose=1
)


y_pred = np.argmax(
    probabilities,
    axis=1
)


y_true = np.asarray(
    y_test[
        SEQUENCE_LENGTH - 1:
    ],
    dtype=np.int8
)

accuracy = accuracy_score(
    y_true,
    y_pred
)


weighted_precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


weighted_recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 80)
print("BALANCED CNN-LSTM RESULTS")
print("=" * 80)


print(
    f"\nAccuracy          : {accuracy:.4f}"
)

print(
    f"Weighted Precision: {weighted_precision:.4f}"
)

print(
    f"Weighted Recall   : {weighted_recall:.4f}"
)

print(
    f"Weighted F1       : {weighted_f1:.4f}"
)

print(
    f"Macro Precision   : {macro_precision:.4f}"
)

print(
    f"Macro Recall      : {macro_recall:.4f}"
)

print(
    f"Macro F1          : {macro_f1:.4f}"
)


class_names = [
    label_mapping[i]
    for i in range(NUM_CLASSES)
]


print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)


print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=class_names,
        zero_division=0
    )
)


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



results = {

    "model":
        "Class-Weighted CNN-LSTM",

    "sequence_length":
        int(SEQUENCE_LENGTH),

    "features":
        int(N_FEATURES),

    "batch_size":
        int(BATCH_SIZE),

    "training_sequences":
        int(training_sequences),

    "validation_sequences":
        int(validation_sequences),

    "test_sequences":
        int(test_sequences),

    "accuracy":
        float(accuracy),

    "weighted_precision":
        float(weighted_precision),

    "weighted_recall":
        float(weighted_recall),

    "weighted_f1":
        float(weighted_f1),

    "macro_precision":
        float(macro_precision),

    "macro_recall":
        float(macro_recall),

    "macro_f1":
        float(macro_f1),

    "class_weights":
        class_weights
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



del probabilities

del train_generator

del validation_generator

del test_generator

gc.collect()

print("\n" + "=" * 80)
print("BALANCED CNN-LSTM TRAINING COMPLETE")
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
print("NEXT: COMPARE CNN-LSTM MODELS")
print("=" * 80)