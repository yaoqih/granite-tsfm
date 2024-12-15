import pandas as pd
import os
import akshare as ak
data_path='./factor_data/'
df=pd.read_parquet(data_path+"SZ300151.parquet")
print(list(df.columns))
print(df.head())
# print(len(os.listdir(data_path)))