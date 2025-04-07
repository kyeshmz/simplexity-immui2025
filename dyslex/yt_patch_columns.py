import os
import pandas as pd

# Your source directory with the 840 ETDD70 CSVs
source_dir = "data/experiment-final/v3.0/original"
patched_dir = "data/experiment-final/v3.0/original-patched"
os.makedirs(patched_dir, exist_ok=True)

# Basic column renaming mapping
metrics_col_rename = {
    'sid': 'subject_id',
    'aoi': 'aoi_id',
    # Add more if necessary
}

fixations_col_rename = {
    'sid': 'subject_id',
    'task': 'task_id',
    # Add more if needed
}

for file_name in os.listdir(source_dir):
    if not file_name.endswith('.csv'):
        continue

    path = os.path.join(source_dir, file_name)
    df = pd.read_csv(path)

    if '_metrics' in file_name:
        df.rename(columns=metrics_col_rename, inplace=True)
    elif '_fixations' in file_name:
        df.rename(columns=fixations_col_rename, inplace=True)

    df.to_csv(os.path.join(patched_dir, file_name), index=False)
    print(f"✅ Patched {file_name}")
