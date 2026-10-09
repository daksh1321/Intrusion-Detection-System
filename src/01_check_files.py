import os
import glob

DATA_PATH = r"E:\IDS_Project\data\raw\CIC-IDS2017"

files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

print("Number of CSV files found:", len(files))
print()

for i, file in enumerate(files, start=1):
    print(f"{i}. {os.path.basename(file)}")