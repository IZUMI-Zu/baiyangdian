import pandas as df
from ..predict.data_processing.load_data import load_nc_data

# df = load_nc_data('./data/20210101.nc')
# print(df)

file_path = './data/20210101.nc'
dataframes = load_nc_data(file_path)


for var_name, df in dataframes.items():
    print(f"Data for {var_name}:")
    print(df.head())  # 打印前几行数据

