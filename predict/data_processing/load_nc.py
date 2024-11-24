from netCDF4 import Dataset
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_nc_data(file_path):
    """
    从 NetCDF 文件读取数据并转化为 Pandas DataFrame。
    
    :param file_path: NetCDF 文件路径
    :return: 包含每个变量数据的字典，键为变量名，值为对应的 DataFrame。
    """
    # 打开 NetCDF 文件
    nc_file = Dataset(file_path, 'r')
    
    # 打印变量列表
    logger.info(f"Variables in the netCDF file: {list(nc_file.variables)}")
    
    # 创建一个字典来存储每个变量的数据
    dataFrames = {}
    
    # 遍历所有变量，读取数据并转换为 DataFrame
    for var_name in nc_file.variables:
        var_data = nc_file.variables[var_name][:]
        
        # 处理变量数据，生成 DataFrame
        if len(var_data.shape) == 1:  # 如果数据是一维
            dataFrames[var_name] = pd.DataFrame(var_data, columns=[var_name])
        elif len(var_data.shape) == 2:  # 如果数据是二维
            dataFrames[var_name] = pd.DataFrame(var_data, columns=[f'{var_name}_col_{i}' for i in range(var_data.shape[1])])
        elif len(var_data.shape) == 3:  # 如果数据是三维
            dataFrames[var_name] = pd.DataFrame(var_data.reshape(-1, var_data.shape[-1]), columns=[f'{var_name}_dim_{i}' for i in range(var_data.shape[-1])])
    
    # 关闭文件
    nc_file.close()
    
    return dataFrames