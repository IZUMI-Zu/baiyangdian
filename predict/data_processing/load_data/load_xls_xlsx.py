import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_xls_data(file_path):
    """Load XLS file and return as pandas DataFrame."""
    try:
        return pd.read_excel(file_path, engine='xlrd')
    except Exception as e:
        logger.error(f"Error loading XLS file: {str(e)}")
        return None
    
def load_xlsx_data(file_path):
    try:
        return pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        logger.error(f"Error loading XLSX file: {str(e)}")
        return None
    
def load_excel_data(file_path):
    if str(file_path).endswith('.xls'):
        return load_xls_data(file_path)
    elif str(file_path).endswith('.xlsx'):
        return load_xlsx_data(file_path)
    else:
        logger.error(f"Unsupported file type: {file_path}")
        return None
