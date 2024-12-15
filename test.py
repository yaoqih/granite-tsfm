import pandas as pd
import os
import akshare as ak
data_path='./basic_data/'
df=pd.read_parquet(data_path+"SH600925.parquet")
print(df.head())
# print(len(os.listdir(data_path)))