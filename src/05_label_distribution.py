import pandas as pd

FILE = r"E:\IDS_Project\data\processed\CIC_IDS2017_combined.csv"

label_counts = {}

for chunk in pd.read_csv(
    FILE,
    chunksize=50000,
    low_memory=False
):

    chunk.columns = chunk.columns.str.strip()

    counts = chunk["Label"].value_counts()

    for label, count in counts.items():

        label_counts[label] = (
            label_counts.get(label, 0) + count
        )

print("\nComplete CIC-IDS2017 label distribution")
print("=" * 60)

for label, count in sorted(
    label_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(f"{label:35s} {count:,}")

print("=" * 60)

print(
    "Total rows:",
    sum(label_counts.values())
)

print(
    "Number of classes:",
    len(label_counts)
)