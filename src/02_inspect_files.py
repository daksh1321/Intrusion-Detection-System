import pandas as pd
import glob
import os

DATA_PATH = r"E:\IDS_Project\data\raw\CIC-IDS2017"

files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

for file in files:

    print("\n" + "=" * 70)
    print(os.path.basename(file))
    print("=" * 70)

    df = pd.read_csv(file, nrows=5)

    df.columns = df.columns.str.strip()

    print("Columns:", len(df.columns))
    print("Shape of first 5 rows:", df.shape)

    print("\nLast column:")
    print(df.columns[-1])