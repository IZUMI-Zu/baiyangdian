import pandas as pd

def load_xls_data(file_path):
    """Load XLS file and return as pandas DataFrame."""
    try:
        return pd.read_excel(file_path, engine='xlrd')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
def load_xlsx_data(file_path):
    try:
        return pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None