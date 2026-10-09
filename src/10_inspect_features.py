import pandas as pd

FILE = (
    r"E:\IDS_Project\data\processed"
    r"\CIC_IDS2017_train.csv"
)

print("=" * 70)
print("CIC-IDS2017 FEATURE INSPECTION")
print("=" * 70)

# Read only the first few rows
df = pd.read_csv(
    FILE,
    nrows=5,
    low_memory=False
)

df.columns = df.columns.str.strip()

print("\nTotal columns:", len(df.columns))

print("\nColumns:")
print("-" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:3d}. {column}")

print("\n" + "=" * 70)
print("LABEL COLUMN")
print("=" * 70)

print("Label" in df.columns)

print("\nFeature count:", len(df.columns) - 1)