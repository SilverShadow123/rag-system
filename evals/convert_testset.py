import pandas as pd
import json
from pathlib import Path

# Input and output files
input_file = Path("synthetic_testset.csv")
output_file = Path("synthetic_testset.json")

# Read CSV
df = pd.read_csv(input_file)

# Rename columns for the evaluation dataset
df = df.rename(columns={
    "question": "user_input",
    "ground_truth": "reference",
    "contexts": "reference_contexts"
})

# Convert context strings into lists when they contain JSON arrays
def parse_contexts(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value

    value = str(value).strip()

    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        return [value] if value else []

df["reference_contexts"] = df["reference_contexts"].apply(parse_contexts)

# Replace missing values with None
df = df.astype(object).where(pd.notna(df), None)

# Export as JSON
records = df.to_dict(orient="records")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Converted {len(records)} records to {output_file}")