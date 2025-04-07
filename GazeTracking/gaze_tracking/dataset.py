import os
import pandas as pd
from pathlib import Path


class ETDD70Loader:
    def __init__(self, data_path='ETDD70'):
        self.data_path = Path(__file__).parent.parent / data_path
        if not self.data_path.exists():
            raise FileNotFoundError(f"ETDD70 directory not found at {self.data_path}")

    def load_task_data(self, task_filters=None):
        """Load CSV files with optional task filtering"""
        csv_files = [
            f for f in self.data_path.glob('*.csv')
            if not task_filters or any(task in f.name for task in task_filters)
        ]

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")

        required_columns = ['avg_fixation_duration', 'num_regressions',
                            'pupil_variability', 'saccade_ratio', 'label']

        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            if not all(col in df.columns for col in required_columns):
                missing = set(required_columns) - set(df.columns)
                raise ValueError(f"Missing columns {missing} in {file.name}")
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True).dropna()


# Example usage
if __name__ == "__main__":
    loader = ETDD70Loader()
    df = loader.load_task_data(['Pseudo_Text'])
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
