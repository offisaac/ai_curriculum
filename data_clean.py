#数据清洗
#1. 删除缺少特征样本
#2. 删除无用列 选取有用列
#3. 修改特征格式()
# 对本 设置工业用房为0号 商业1号 住宅2号
# 赊钱报名为0号 放心付1号
#4. 保存数据
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime

# 读取Excel文件，确保安装了 openpyxl
df = pd.read_excel("datas/new_resource_pred/风电场风功率.xlsx", engine='openpyxl')
# 1.
df = df.dropna(axis=0, how='any')
# 2.
# 3.转化时间的方式 1.使用to_datatime读取存在固定间隔符号的时间序列 2.使用df的dt属性中的时间属性调用
if "日期" in df.columns:
    df['日期'] = pd.to_datetime(df['日期'], format='%Y/%m/%d')
    df['月份'] = df['日期'].dt.month
    df['日'] = df['日期'].dt.day
if "时间" in df.columns:
    df['时间'] = pd.to_datetime(df['时间'], format='%H:%M:%S')
    df['分钟'] = df['时间'].dt.hour * 60 + df['时间'].dt.minute
df.to_excel("datas/new_resource_pred/风电场风功率.xlsx", engine='openpyxl')