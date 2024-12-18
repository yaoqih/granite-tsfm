import pandas as pd
import os
import akshare as ak
data_path='./basic_data/'
df=pd.read_parquet(data_path+"SH603088.parquet")
print(list(df.columns))
print(df.tail())
# print(len(os.listdir(data_path)))