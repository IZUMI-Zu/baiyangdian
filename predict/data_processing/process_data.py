from .load_data import load_csv_data, load_xls_data, load_xlsx_data, load_nc_data

def load_data(file_path: str):
    """Load data from file and return as xarray Dataset or pandas DataFrame."""
    if file_path.endswith('.nc'):
        return load_nc_data(file_path)
    elif file_path.endswith('.csv'):
        return load_csv_data(file_path)
    elif file_path.endswith('.xls'):
        return load_xls_data(file_path)
    elif file_path.endswith('.xlsx'):
        return load_xlsx_data(file_path)
