import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_csv_data(file_path: str):
    """Load CSV file and return as pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    try:
        return pd.read_csv(file_path, 
                          encoding='utf-8',
                          na_values=['NA', 'missing', ''],  # Common NA values
                          low_memory=False,  # Avoid mixed type inference warnings
                          parse_dates=True)  # Automatically parse date columns
    except FileNotFoundError:
        logger.error(f"CSV file not found at: {file_path}")
        raise FileNotFoundError(f"CSV file not found at: {file_path}")
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {file_path}")
        raise pd.errors.EmptyDataError(f"CSV file is empty: {file_path}")
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        raise Exception(f"Error loading CSV file: {str(e)}")

def process_reservoir_data(df: pd.DataFrame):
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

# Chinese to English column mappings for reservoir data
RESERVOIR_COLUMNS_MAP = {
    '日期': 'date',
    '库(淀)': 'reservoir',
    '库站': 'station',
    '正常蓄水位(m)': 'normal_water_level_m',
    '水位08h(m)': 'water_level_08h_m',
    '蓄水量08h(百万m³)': 'storage_08h_million_m3',
    '出库流量08h(m³/s)': 'outflow_08h_m3s',
    '闸门启闭-孔数': 'gate_holes',
    '闸门启闭-开启高度': 'gate_opening_height',
    '日均入库流量08h(m³/s)': 'avg_daily_inflow_08h_m3s',
    '日均出库流量08h(m³/s)': 'avg_daily_outflow_08h_m3s'
}

# List of numeric columns for type conversion
NUMERIC_COLUMNS = [
    'normal_water_level_m',
    'water_level_08h_m',
    'storage_08h_million_m3',
    'outflow_08h_m3s',
    'gate_holes',
    'gate_opening_height',
    'avg_daily_inflow_08h_m3s',
    'avg_daily_outflow_08h_m3s'
] 