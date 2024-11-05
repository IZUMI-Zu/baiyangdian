import pandas as pd
import xarray as xr

def read_data(file_path: str) -> pd.DataFrame:
    """读取不同格式的数据文件并统一转换为DataFrame格式
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        pd.DataFrame: 统一格式的数据框
    """
    file_type = file_path.split('.')[-1].lower()
    
    if file_type == 'nc':
        # 读取nc文件
        ds = xr.open_dataset(file_path)
        df = ds.to_dataframe()
        
    elif file_type == 'csv':
        # 读取csv文件
        df = pd.read_csv(file_path)
        
    elif file_type in ['xls', 'xlsx']:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # 确保数据包含必要的列
    required_columns = ['timestamp', 'water_level']  # 根据实际需求修改
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # 设置时间索引
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    return df

