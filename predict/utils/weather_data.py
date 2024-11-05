import pandas as pd
import netCDF4 as nc
import numpy as np

def read_weather_data(file_path: str) -> pd.DataFrame:
    """读取气象NC数据
    
    Args:
        file_path: NC格式气象数据文件路径
        
    Returns:
        pd.DataFrame: 处理后的气象数据，包含所有气象参数
    """
    # 打开NC文件
    dataset = nc.Dataset(file_path)
    
    # 获取时间数据并转换 - 处理不同的时间变量名称
    time_vars = ['Time', 'time', 'TIME']
    time_var = None
    for var in time_vars:
        if var in dataset.variables:
            time_var = var
            break
    
    if time_var is None:
        raise ValueError(f"No time variable found in NC file. Available variables: {list(dataset.variables.keys())}")
    
    # 获取时间单位和基准时间信息
    time_units = dataset.variables[time_var].units
    if not hasattr(dataset.variables[time_var], 'units'):
        raise ValueError(f"Time variable does not have units attribute")
    
    # 使用netCDF4的内置时间转换，并确保时间在有效范围内
    times = nc.num2date(dataset.variables[time_var][:], 
                       time_units,
                       only_use_cftime_datetimes=True,  # 使用CFTime对象来处理超出范围的日期
                       only_use_python_datetimes=False)
    
    # 转换为字符串后再转换为pandas时间戳
    timestamps = pd.to_datetime([t.strftime('%Y-%m-%d %H:%M:%S') for t in times], 
                              format='%Y-%m-%d %H:%M:%S')
    
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


def process_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """处理气象数据的特定逻辑
    
    Args:
        df: 原始气象数据DataFrame
        
    Returns:
        pd.DataFrame: 处理后的气象数据
    """
    # 确保时间索引
    df.index.name = 'timestamp'
    
    # 单位转换（根据需要调整）
    # 例如：将温度从K转换为℃
    if 'temperature_2m' in df.columns:
        df['temperature_2m'] = df['temperature_2m'] - 273.15
    
    # 添加一些可能有用的派生特征
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['season'] = df.index.month % 12 // 3 + 1
    
    # 处理异常值
    for col in df.columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # 使用3个标准差作为异常值界限
        mean = df[col].mean()
        std = df[col].std()
        df[col] = df[col].clip(mean - 3*std, mean + 3*std)
    
    # 填充缺失值
    df = df.interpolate(method='time')
    
    return df