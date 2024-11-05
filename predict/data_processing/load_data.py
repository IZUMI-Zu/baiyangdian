import xarray as xr
import pandas as pd

from .column_mappings import RESERVOIR_COLUMNS_MAP, NUMERIC_COLUMNS
from ..utils import load_csv_data

def load_data(file_path):
    """Load data from file and return as xarray Dataset or pandas DataFrame."""
    if file_path.endswith('.nc'):
        return load_nc_data(file_path)
    elif file_path.endswith('.csv'):
        return process_reservoir_data(load_csv_data(file_path))
    elif file_path.endswith('.xls'):
        return load_xls_data(file_path)

def load_nc_data(file_path):
    """Load nc file and return as xarray Dataset."""
    return xr.open_dataset(file_path)

def process_reservoir_data(df):
    """Process reservoir data by translating headers and converting data types.
    
    Args:
        df (pandas.DataFrame): Input DataFrame with Chinese headers
        
    Returns:
        pandas.DataFrame: Processed DataFrame with English headers and correct data types
    """
    # Rename columns using the mapping
    df = df.rename(columns=RESERVOIR_COLUMNS_MAP)
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Convert numeric columns to appropriate types
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def load_xls_data(file_path):
    """Load XLS file and return as pandas DataFrame."""
    return pd.read_excel(file_path)
