import os
import shutil

source_dir = "/Users/heyutian/Documents/For transfer to hard disk/11_Grad School/MIT_2025_Spring_Semester/6_8510/TermProject/Eye_Tracking/Git_Repo_Downloads/dyslex/experiment-final/v3.0/ETDD70"
dest_dir = "/Users/heyutian/Documents/For transfer to hard disk/11_Grad School/MIT_2025_Spring_Semester/6_8510/TermProject/Eye_Tracking/Git_Repo_Downloads/dyslex/data/experiment-final/v3.0/original/"

os.makedirs(dest_dir, exist_ok=True)

for filename in os.listdir(source_dir):
    if not filename.lower().endswith(".csv"):
        continue

    parts = filename.split('_')
    if len(parts) < 4:
        continue  # skip files with unexpected names

    subject = parts[1]
    task = parts[2]
    task_type = parts[3].lower()

    # Infer suffix
    if 'fixation' in filename.lower():
        suffix = 'fixations.csv'
    elif 'saccade' in filename.lower() or 'metric' in filename.lower():
        suffix = 'metrics.csv'
    else:
        print(f"⚠️ Skipping unrecognized file: {filename}")
        continue

    new_filename = f"{subject}_{task}_{suffix}"
    src = os.path.join(source_dir, filename)
    dst = os.path.join(dest_dir, new_filename)

    print(f"✅ Renaming: {filename} → {new_filename}")
    shutil.copy(src, dst)  # or use shutil.move() to move instead of copy

# for f in os.listdir(raw_dir):
#     if not f.endswith(".csv"):
#         continue
#
#     parts = f.split('_')
#     if len(parts) < 4:
#         continue  # unexpected filename
#
#     subject_id = parts[1]
#     task_id = parts[2]
#     task_type = parts[3].lower()  # 'Syllables', 'MeaningfulText', etc.
#
#     # Define what type of data this is
#     if 'saccade' in f.lower() or 'metric' in f.lower():
#         suffix = 'metrics.csv'
#     elif 'fixation' in f.lower():
#         suffix = 'fixations.csv'
#     else:
#         suffix = 'unknown.csv'  # fallback
#
#     # New filename
#     new_name = f"{subject_id}_{task_id}_{suffix}"
#     old_path = os.path.join(raw_dir, f)
#     new_path = os.path.join(raw_dir, new_name)
#     print(f"Renaming: {f} → {new_name}")
#     os.rename(old_path, new_path)
