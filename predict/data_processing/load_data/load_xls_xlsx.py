import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_xls_data(file_path: str):
    """Load XLS file and return as pandas DataFrame."""
    try:
        return pd.read_excel(file_path, engine='xlrd')
    except Exception as e:
        logger.error(f"Error loading XLS file: {str(e)}")
        return None
    
def load_xlsx_data(file_path: str):
    try:
        return pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        logger.error(f"Error loading XLSX file: {str(e)}")
        return None