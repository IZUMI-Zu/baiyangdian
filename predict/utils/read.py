import pandas as pd
import netCDF4 as nc
import numpy as np
from typing import Dict, Union

def read_weather_data(file_path: str) -> pd.DataFrame:
    """读取气象NC数据
    
    Args:
        file_path: NC格式气象数据文件路径
        
    Returns:
        pd.DataFrame: 处理后的气象数据，包含所有气象参数
    """
    # 打开NC文件
    dataset = nc.Dataset(file_path)
    
    # 获取时间数据并转换
    time = dataset.variables['Time'][:]
    # 通常NC文件的时间是相对于某个基准时间的小时数或天数
    # 需要根据实际数据调整基准时间
    base_time = pd.Timestamp('2000-01-01')  # 示例基准时间，需要根据实际调整
    timestamps = [base_time + pd.Timedelta(hours=int(t)) for t in time]
    
    # 创建数据字典
    data_dict = {
        'Lwr': dataset.variables['Lwr'][:].flatten(),  # 长波辐射
        'Swr': dataset.variables['Swr'][:].flatten(),  # 短波辐射
        'Sp': dataset.variables['Sp'][:].flatten(),    # 大气气压
        'Prec': dataset.variables['Prec'][:].flatten(),  # 降水量
        'Rh2m': dataset.variables['Rh2m'][:].flatten(),  # 相对湿度
        'Q2m': dataset.variables['Q2m'][:].flatten(),    # 比湿值
        'T2m': dataset.variables['T2m'][:].flatten(),    # 温度
        'U10': dataset.variables['U10'][:].flatten(),    # 风速u分量
        'V10': dataset.variables['V10'][:].flatten(),    # 风速v分量
    }
    
    # 计算风速和风向
    data_dict['wind_speed'] = np.sqrt(data_dict['U10']**2 + data_dict['V10']**2)
    data_dict['wind_direction'] = np.arctan2(data_dict['V10'], data_dict['U10']) * 180 / np.pi
    
    # 创建DataFrame
    df = pd.DataFrame(data_dict, index=timestamps)
    
    # 重命名列以更直观
    columns_map = {
        'Lwr': 'longwave_radiation',
        'Swr': 'shortwave_radiation',
        'Sp': 'atmospheric_pressure',
        'Prec': 'precipitation',
        'Rh2m': 'relative_humidity_2m',
        'Q2m': 'specific_humidity_2m',
        'T2m': 'temperature_2m',
        'U10': 'wind_u_10m',
        'V10': 'wind_v_10m'
    }
    df = df.rename(columns=columns_map)
    
    # 关闭NC文件
    dataset.close()
    
    return df

def read_water_level_data(file_path: str) -> pd.DataFrame:
    """读取水位数据
    
    Args:
        file_path: 水位数据文件路径
        
    Returns:
        pd.DataFrame: 处理后的水位数据，包含水位、时间等
    """
    file_type = file_path.split('.')[-1].lower()
    
    if file_type in ['xls', 'xlsx']:
        df = pd.read_excel(file_path)
        # 处理水位特有的列名和格式
        columns_map = {
            '水位': 'water_level',
            '时间': 'timestamp',
            # 添加其他水位数据的列映射
        }
        df = df.rename(columns=columns_map)
        
    elif file_type == 'csv':
        df = pd.read_csv(file_path)
    
    return df

def read_hydrological_data(file_path: str) -> pd.DataFrame:
    """读取水文数据
    
    Args:
        file_path: 水文数据文件路径
        
    Returns:
        pd.DataFrame: 处理后的水文数据，包含流量、水质等
    """
    file_type = file_path.split('.')[-1].lower()
    
    if file_type == 'csv':
        df = pd.read_csv(file_path)
        # 处理水文特有的列名和格式
        columns_map = {
            '流量': 'flow_rate',
            '水质': 'water_quality',
            # 添加其他水文数据的列映射
        }
        df = df.rename(columns=columns_map)
        
    # 添加其他格式的处理...
    
    return df

def process_data(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """统一处理数据格式
    
    Args:
        df: 输入数据框
        data_type: 数据类型 ('weather', 'water_level', 'hydrological')
        
    Returns:
        pd.DataFrame: 处理后的数据框
    """
    # 统一时间格式
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # 根据数据类型进行特定处理
    if data_type == 'weather':
        # 处理气象数据特有的逻辑
        df = df.resample('H').mean()  # 按小时重采样
        
    elif data_type == 'water_level':
        # 处理水位数据特有的逻辑
        df = df.sort_index()
        
    elif data_type == 'hydrological':
        # 处理水文数据特有的逻辑
        pass
    
    return df

def read_data(file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """主函数：读取所有类型的数据
    
    Args:
        file_paths: 包含各类数据文件路径的字典
            例如: {
                'weather': 'path/to/weather.nc',
                'water_level': 'path/to/water_level.xlsx',
                'hydrological': 'path/to/hydro.csv'
            }
            
    Returns:
        Dict[str, pd.DataFrame]: 包含所有处理后数据的字典
    """
    result = {}
    
    if 'weather' in file_paths:
        df_weather = read_weather_data(file_paths['weather'])
        result['weather'] = process_data(df_weather, 'weather')
        
    if 'water_level' in file_paths:
        df_water = read_water_level_data(file_paths['water_level'])
        result['water_level'] = process_data(df_water, 'water_level')
        
    if 'hydrological' in file_paths:
        df_hydro = read_hydrological_data(file_paths['hydrological'])
        result['hydrological'] = process_data(df_hydro, 'hydrological')
    
    return result

