import pandas as pd

FILE = r"E:\IDS_Project\data\processed\CIC_IDS2017_combined.csv"

# Read only a sample for quick inspection
df = pd.read_csv(
    FILE,
    nrows=100000,
    low_memory=False
)

print("Sample shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabels in sample:")
print(df["Label"].value_counts())