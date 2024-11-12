import xarray as xr
import pandas as pd
import numpy as np


from .column_mappings import RESERVOIR_COLUMNS_MAP, NUMERIC_COLUMNS
from ..utils import load_csv_data
from netCDF4 import Dataset
from prettytable import PrettyTable

def load_data(file_path):
    """Load data from file and return as xarray Dataset or pandas DataFrame."""
    if file_path.endswith('.nc'):
        return load_nc_data(file_path)
    elif file_path.endswith('.csv'):
        return process_reservoir_data(load_csv_data(file_path))
    elif file_path.endswith('.xls'):
        return load_xls_data(file_path)


def load_nc_data(file_path):
    """
    从 NetCDF 文件读取数据并转化为 Pandas DataFrame。
    
    :param file_path: NetCDF 文件路径
    :return: 包含每个变量数据的字典，键为变量名，值为对应的 DataFrame。
    """
    # 打开 NetCDF 文件
    nc_file = Dataset(file_path, 'r')
    
    # 打印变量列表
    print(f"Variables in the netCDF file: {list(nc_file.variables)}")
    
    # 创建一个字典来存储每个变量的数据
    dataframes = {}
    
    # 遍历所有变量，读取数据并转换为 DataFrame
    for var_name in nc_file.variables:
        var_data = nc_file.variables[var_name][:]
        
        # 处理变量数据，生成 DataFrame
        if len(var_data.shape) == 1:  # 如果数据是一维
            dataframes[var_name] = pd.DataFrame(var_data, columns=[var_name])
        elif len(var_data.shape) == 2:  # 如果数据是二维
            dataframes[var_name] = pd.DataFrame(var_data, columns=[f'{var_name}_col_{i}' for i in range(var_data.shape[1])])
        elif len(var_data.shape) == 3:  # 如果数据是三维
            dataframes[var_name] = pd.DataFrame(var_data.reshape(-1, var_data.shape[-1]), columns=[f'{var_name}_dim_{i}' for i in range(var_data.shape[-1])])
    
    # 关闭文件
    nc_file.close()
    
    return dataframes


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
