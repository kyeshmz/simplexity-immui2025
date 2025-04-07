import os

in_data_dir = "data/experiment-final/v3.0/original"
print([f for f in os.listdir(in_data_dir) if "T4" in f and "metrics" in f])
