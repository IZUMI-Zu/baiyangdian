import pandas as pd

def load_csv_data(file_path):
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
        raise FileNotFoundError(f"CSV file not found at: {file_path}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"CSV file is empty: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")