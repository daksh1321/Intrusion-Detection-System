import pandas as pd

TRAIN_FILE = (
    r"E:\IDS_Project\data\processed"
    r"\CIC_IDS2017_train.csv"
)

TEST_FILE = (
    r"E:\IDS_Project\data\processed"
    r"\CIC_IDS2017_test.csv"
)

CHUNK_SIZE = 50000


def get_distribution(file_path):

    counts = {}
    total = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk.columns = chunk.columns.str.strip()

        total += len(chunk)

        label_counts = chunk["Label"].value_counts()

        for label, count in label_counts.items():

            counts[label] = (
                counts.get(label, 0) + count
            )

    return total, counts


# TRAIN DATA

train_total, train_counts = get_distribution(
    TRAIN_FILE
)

print("=" * 75)
print("TRAIN DATA DISTRIBUTION")
print("=" * 75)

for label, count in sorted(
    train_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    percentage = count / train_total * 100

    print(
        f"{label:35s}"
        f"{count:12,}"
        f"   {percentage:7.3f}%"
    )


# TEST DATA

test_total, test_counts = get_distribution(
    TEST_FILE
)

print("\n" + "=" * 75)
print("TEST DATA DISTRIBUTION")
print("=" * 75)

for label, count in sorted(
    test_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    percentage = count / test_total * 100

    print(
        f"{label:35s}"
        f"{count:12,}"
        f"   {percentage:7.3f}%"
    )


# SUMMARY

print("\n" + "=" * 75)
print("SPLIT SUMMARY")
print("=" * 75)

print(f"Total dataset : {train_total + test_total:,}")
print(f"Training      : {train_total:,}")
print(f"Testing       : {test_total:,}")

print("\nClasses in training:", len(train_counts))
print("Classes in testing :", len(test_counts))

missing_from_train = set(test_counts) - set(train_counts)
missing_from_test = set(train_counts) - set(test_counts)

print("\nMissing from training:", missing_from_train)
print("Missing from testing :", missing_from_test)

print("\n" + "=" * 75)