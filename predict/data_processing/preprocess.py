import pandas as pd

def clean_data(df):
    """Clean data by handling missing values."""
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def merge_data(data_sources):
    """Merge different datasets into a single DataFrame."""
    return pd.concat(data_sources, axis=1)
