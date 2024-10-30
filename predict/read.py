import xlrd
import numpy as np
import math
def excel2matrix(path):
    t=0
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数 32
    ncols = table.ncols  # 列数 13
    datamatrix = np.zeros((ncols-1, nrows-1)) #(12,31)
    # datamatrix = np.zeros((nrows-1, ncols-1)) #(31,12)
    for i in range(ncols-1):  #12 0----11
        cols = table.col_values(i+1,start_rowx=1) # 1----12
        for j in range(len(cols)):
            try:
                datamatrix[i, j] = float(cols[j])
            except ValueError:
                datamatrix[i, j] = math.nan
    return datamatrix


path1 = '2019shuiwei.xls'  #  113.xlsx 在当前文件夹下
path2 = '2020shuiwei.xls'  #  113.xlsx 在当前文件夹下
path3 = '2021shuiwei.xls'  #  113.xlsx 在当前文件夹下
x2019 = excel2matrix(path1)
x2020 = excel2matrix(path2)
x2021 = excel2matrix(path3)

# x2019=x2019.reshape(1,31*12)
# print(x2019)
x2019_without_nan = x2019[~np.isnan(x2019)]
print(x2019_without_nan.shape)
# print(x2019_without_nan)
# print(x2019_without_nan.shape)
# x2020=x2019.reshape(1,31*12)
x2020_without_nan = x2020[~np.isnan(x2020)]
print(x2020_without_nan.shape)
# x2021=x2019.reshape(1,31*12)
x2021_without_nan = x2021[~np.isnan(x2021)]
print(x2021_without_nan.shape)


shuiwei = np.concatenate([x2019_without_nan, x2020_without_nan, x2021_without_nan],axis=0)
shuiwei=np.expand_dims(shuiwei, 1)
np.save('', shuiwei)
# print(combined)
print(shuiwei.shape)
# print(x2019_without_nan)
# print(x2019_without_nan.shape)
# print(x)
# print(x.shape)

