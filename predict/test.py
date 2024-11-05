import netCDF4 as nc
import numpy as np
from netCDF4 import num2date
# 打开文件
ds = nc.Dataset('data/20210101.nc', 'r')

# 读取各个变量
# 1. 坐标和时间变量
time = ds.variables['time']  # 时间
x = ds.variables['x'][:]        # x坐标
y = ds.variables['y'][:]        # y坐标

# 2. 气象变量
# 长波辐射
lwr = ds.variables['lwr'][:]    # Long Wave Radiation
# 短波辐射
swr = ds.variables['swr'][:]    # Short Wave Radiation
# 表面气压
sp = ds.variables['sp'][:]      # Surface Pressure
# 降水
prec = ds.variables['prec'][:]  # Precipitation
# 2米相对湿度
rh2m = ds.variables['rh2m'][:]  # 2m Relative Humidity
# 2米比湿
q2m = ds.variables['q2m'][:]    # 2m Specific Humidity
# 2米温度
t2m = ds.variables['t2m'][:]    # 2m Temperature
# 10米u风速
u10 = ds.variables['u10'][:]    # 10m U-component wind
# 10米v风速
v10 = ds.variables['v10'][:]    # 10m V-component wind


print(f"时间变量: {time}")

# 查看每个变量的基本信息
for var_name in ['lwr', 'swr', 'sp', 'prec', 'rh2m', 'q2m', 't2m', 'u10', 'v10']:
    var = ds.variables[var_name]
    print(f"\n{var_name}:")
    print(f"形状: {var.shape}")
    print(f"数据类型: {var.dtype}")
    print(f"最小值: {np.min(var)}")
    print(f"最大值: {np.max(var)}")
    print(f"单位: {var.units if hasattr(var, 'units') else '未指定'}")
    print(f"描述: {var.long_name if hasattr(var, 'long_name') else '未指定'}")

# 检查变量的维度后再进行切片
t2m_shape = ds.variables['t2m'].shape
print(f"\nt2m shape: {t2m_shape}")

if len(t2m_shape) == 2:
    t2m_first = ds.variables['t2m'][:, :]  # 如果是2维数据
    t2m_subset = ds.variables['t2m'][:, :]  # 读取所有数据
elif len(t2m_shape) == 3:
    t2m_first = ds.variables['t2m'][0, :, :]  # 如果是3维数据
    t2m_subset = ds.variables['t2m'][0:24, :, :]  # 读取前24个时间点

# 计算一些基本统计量
print("\n温度统计信息:")
print(f"平均温度: {np.mean(t2m):.2f}")
print(f"最高温度: {np.max(t2m):.2f}")
print(f"最低温度: {np.min(t2m):.2f}")

# 获取时间变量
time_var = ds.variables['time']

# 直接使用 num2date 转换时间
dates = num2date(time_var[:], 
                units=time_var.units, 
                calendar=time_var.calendar)


# 方法1：直接打印时间变量的所有值
print("\n原始时间数据:")
print(time_var[:])  # 这会显示所有时间点的原始值

# 方法2：使用numpy转换后打印
print("\n使用numpy数组形式显示:")
time_data = np.array(time_var[:])
print(time_data)

# 方法3：逐个打印并显示索引
print("\n带索引的时间数据:")
for i, t in enumerate(time_var[:]):
    print(f"时间点 {i}: {t} {time_var.units}")

# 关闭文件
ds.close()

# Example usage
# if __name__ == "__main__":
#     weather_data_path = '../data/20210101.nc'
#     weather_data = read_weather_data(weather_data_path)
#     processed_weather_data = process_weather_data(weather_data)
#     print(processed_weather_data.info())
#     print(processed_weather_data.describe())

