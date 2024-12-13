import pandas as pd
for file_name in ['test','valid']:
    df=pd.read_csv(f'zero_shot_{file_name}.csv')
    grouped = df.groupby('date')
    ids=[]
    for id,group in grouped:
        if len(group)>1000:
            id_=group['predict'].idxmax()
            ids.append(id_)
    result = df.loc[ids]
    # 将结果保存到Excel文件
    result.to_excel(f'highest_predict_{file_name}.xlsx', index=False)