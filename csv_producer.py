from pathlib import Path
import pandas as pd
import regex as re

def sn(name):
    max_len = 40
    s = str(name).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)   # commas, brackets, spaces, slashes, parentheses -> _
    s = s.strip("_")[:max_len].rstrip("_")
    return s

directory = Path("data/log_files/data_3")
csv_file = Path("molecules/csv_files/mega_data_86400.csv")
name_column = "safeName"  # CSV column containing file names
remove_part = "data_3_"  # Set to "" if nothing should be removed


# Collect file names. A set handles duplicate names.
file_names = {
    path.name.replace(remove_part, "").replace(".log", "")
    for path in directory.iterdir()
    if path.is_file()
}

df = pd.read_csv(csv_file)
df["safeName"] = df.apply(
    lambda row: sn(row["IUPAC_Name"]),
    axis=1,
)
print(df.head())

# Apply the same change to the names in the CSV before comparing.
csv_names = df[name_column].astype(str).str.replace(
    remove_part, "", regex=False
)

filtered_df = df[csv_names.isin(file_names)].copy()

print(f"Filtered rows: {len(filtered_df)}")
print(filtered_df)
filtered_df.to_csv("data/log_data/data_3.csv", index = False)