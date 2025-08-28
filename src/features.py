import pandas as pd

FEATURES_ALL = [f'V{i}' for i in range(1,29)] + ['Amount', 'Time']
TARGET = 'Class'

def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def split_X_y(df: pd.DataFrame):
    X = df[FEATURES_ALL].copy()
    y = df[TARGET].copy()
    return X, y
