import os
import json
import pandas as pd
import numpy as np
from feature_extractor import (
    load_task_feature_definition_dict,
    load_and_transform_subject_characteristics,
    create_subject_characteristics_profile
)

# === Paths ===
BASE_DIR = os.getcwd()
DATA_VERSION = "v3.0"
DATA_BASE_DIR = os.path.join(BASE_DIR, "data", "experiment-final", DATA_VERSION)

in_data_dir = os.path.join(DATA_BASE_DIR, "original")
out_data_extracted_features_dir = os.path.join(DATA_BASE_DIR, "extracted_features")
meta_file = os.path.join(BASE_DIR, "properties.json")

# === Settings ===
fill_null_values_by_zero = True
summarize_characteristics_by_mean = True
normalize_values = False
DATA_FILE_EXT = ".csv"

# === Load properties ===
with open(meta_file) as property_file:
    properties = json.load(property_file)

meta_dir = properties["meta_dir"]
subject_id_col_name = properties["subject_id_col_name"]
aoi_id_col_name = properties["aoi_id_col_name"]

# === Task Definitions ===
tasks_def = eval(properties["tasks"])
task_feature_def_dict = load_task_feature_definition_dict(tasks_def, meta_dir)
task_definitions = [(task["type_id"], task["desc"], task["id"]) for task in tasks_def]

# === Extract features for T4 only ===
for task_type_id, task_desc, task_id in task_definitions:
    if task_id != "T4":
        continue

    print(f"🔍 Extracting features for Task {task_id} - {task_desc}")
    subject_metrics_files = [
        f for f in os.listdir(in_data_dir)
        if f.endswith(f"_{task_id}_metrics{DATA_FILE_EXT}")
    ]

    data_dict = {}
    for file_name in subject_metrics_files:
        full_path = os.path.join(in_data_dir, file_name)
        subject_id, subject_data_dict = load_and_transform_subject_characteristics(
            task_type_id,
            task_feature_def_dict,
            fill_null_values_by_zero,
            subject_id_col_name,
            aoi_id_col_name,
            summarize_characteristics_by_mean,
            full_path
        )
        data_dict[subject_id] = subject_data_dict

    # === Normalize if enabled ===
    if normalize_values:
        for (characteristics_name, columns) in [
            ("characteristics_global", task_feature_def_dict[task_type_id]["characteristics_global"]),
            ("characteristics", task_feature_def_dict[task_type_id]["characteristics"]),
            ("AOIs_characteristics", task_feature_def_dict[task_type_id]["aoi_characteristics"])
        ]:
            if columns is not None:
                for column in data_dict[subject_id][characteristics_name].columns:
                    all_vals = [
                        data_dict[sub_id][characteristics_name][column].values
                        for sub_id in data_dict.keys()
                    ]
                    all_vals = [v for sublist in all_vals for v in sublist]
                    mean, std = np.mean(all_vals), np.std(all_vals)
                    for sub_id in data_dict.keys():
                        data_dict[sub_id][characteristics_name].loc[:, column] = \
                            data_dict[sub_id][characteristics_name][column].apply(lambda x: (x - mean) / std)

    # === Flatten and save ===
    df_profiles = [
        create_subject_characteristics_profile(subject_id, subject_data)
        for subject_id, subject_data in data_dict.items()
    ]

    if df_profiles:
        df_final = pd.concat(df_profiles, axis=0)
        os.makedirs(out_data_extracted_features_dir, exist_ok=True)
        out_file = os.path.join(out_data_extracted_features_dir, f"{task_id}_{task_desc}_features.csv")
        df_final.to_csv(out_file, sep=";", index=False)
        print(f"✅ Features saved to: {out_file}")
    else:
        print("⚠️ No feature data found.")
