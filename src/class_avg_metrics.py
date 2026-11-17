import pandas as pd
import numpy as np
import ast
import argparse
from pathlib import Path

# -----------------------------------------------
# Usage: python class_avg_metrics.py --input <results>/class_metrics.csv [--output <file>.csv]
# -----------------------------------------------
parser = argparse.ArgumentParser(description="Append the per-class mean AP[50-95] row to a class_metrics.csv file")
parser.add_argument("--input", type=str, required=True, help="Path to class_metrics.csv")
parser.add_argument("--output", type=str, default=None, help="Output CSV (default: <input>_with_means.csv)")
args = parser.parse_args()

# Load file
file_path = Path(args.input)
df = pd.read_csv(file_path)

# Parse the 'cl_ap_IoU[50-95]' column into lists of floats
def parse_list(s):
    if pd.isna(s):
        return np.nan
    s = s.strip("[] ")
    parts = [p for p in s.split() if p.strip()]
    return [float(p) for p in parts]

df["cl_ap_IoU[50-95]_parsed"] = df["cl_ap_IoU[50-95]"].apply(parse_list)

# Convert to a DataFrame of subcolumns
sub_df = pd.DataFrame(df["cl_ap_IoU[50-95]_parsed"].tolist())

# Compute mean of each subcolumn
means = sub_df.mean().round(5).tolist()

# Find first empty row (where the target column is NaN)
empty_row_idx = df["cl_ap_IoU[50-95]"].isna().idxmax() if df["cl_ap_IoU[50-95]"].isna().any() else len(df)

# If no empty row, append a new one
if empty_row_idx == len(df):
    df.loc[len(df)] = [np.nan] * len(df.columns)

# Insert the means back as a list string
df.at[empty_row_idx, "cl_ap_IoU[50-95]"] = str(means)

# Drop helper column
df.drop(columns=["cl_ap_IoU[50-95]_parsed"], inplace=True)

# Save the updated file
output_path = Path(args.output) if args.output else file_path.with_name(file_path.stem + "_with_means.csv")
df.to_csv(output_path, index=False)

print(f"Saved: {output_path}")
