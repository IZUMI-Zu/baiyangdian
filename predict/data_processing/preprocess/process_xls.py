import logging
import pandas as pd

logger = logging.getLogger(__name__)

def process_xls(df: pd.DataFrame) -> pd.Series:
    """
    Process water level data from a single DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing water level data
            with months as columns and days as rows.
            
    Returns:
        pd.Series: Processed one-dimensional Series with NaN values removed.
    """
    # Remove '日/月' (day/month) column if it exists
    if '日/月' in df.columns:
        df = df.drop('日/月', axis=1)
    
    # Transpose the DataFrame so months are in rows and days are in columns
    df_transposed = df.T
    # TODO: process indsx with year.
    # Flatten the DataFrame into a one-dimensional Series
    series_flattened = df_transposed.stack()
    
    # Remove NaN values
    series_clean = series_flattened.dropna()
    
    return series_clean

def process_xlsx(df: pd.DataFrame) -> pd.Series:
    """
    Process the DataFrame to retain 'data' values indexed by 'ymd' and
    output as a one-dimensional Series.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing 'time', 'data', and 'ymd'.
    
    Returns:
        pd.Series: Flattened Series indexed by 'ymd'.
    """
    # Drop the 'time' column as it is not needed
    if 'time' in df.columns:
        df = df.drop('time', axis=1)
    
    # Use 'ymd' as the index and extract the 'data' column
    df_processed = df.set_index('ymd')['data']
    
    return df_processed
