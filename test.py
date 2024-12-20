import pandas as pd
import os
import akshare as ak
import sys
predict_path='./predict_result/'
file_list=os.listdir(predict_path)
if 'loss.csv' not in os.listdir('./'):
    open('loss.csv','w').write('epoch,12-12_all,12-12,12-13_all,12-13_all,12-16,12-16\n')
open('loss.csv','a').write(f'{sys.argv[1]}')
for file_name in sorted(file_list):
    df1=pd.read_csv(predict_path+file_name)
    df1.sort_values('predict',ascending=False,inplace=True)
    df1.reset_index(drop=True,inplace=True)
    # df1=df1[:20]
    if 'true' in df1.columns and df1['true'].isnull().sum()==0:
        loss=(abs(df1['true']-df1['predict'])).mean()
        open('loss.csv','a').write(f',{loss}')
        # print(f'{file_name}的平均误差为{loss}')
    df1=df1[:20]
    if 'true' in df1.columns and df1['true'].isnull().sum()==0:
        loss=(abs(df1['true']-df1['predict'])).mean()
        open('loss.csv','a').write(f',{loss}')
        print(f'{file_name}的平均误差为{loss}')
open('loss.csv','a').write(f'\n')
    
    