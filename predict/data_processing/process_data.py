from .load_data import load_csv_data, load_excel_data, load_nc_data

def load_data(file_path: str):
    """Load data from file and return as xarray Dataset or pandas DataFrame."""
    if str(file_path).endswith('.nc'):
        return load_nc_data(file_path)
    elif str(file_path).endswith('.csv'):
        return load_csv_data(file_path)
    elif str(file_path).endswith('.xls'):
        return load_excel_data(file_path)
    elif str(file_path).endswith('.xlsx'):
        return load_excel_data(file_path)
