from pathlib import Path

# This points to your raw data folder
ROOT = Path(r"C:\Users\azhao\PycharmProjects\PEAgent")
RAW_DIR = ROOT / "data" / "raw"

if RAW_DIR.exists():
    print("--- FILES FOUND IN DATA/RAW ---")
    for file in RAW_DIR.iterdir():
        print(f"'{file.name}'")
    print("-------------------------------")
else:
    print(f"Directory not found at: {RAW_DIR}")