import json
import os
import pandas as pd
result_data={'loss':{},'eval_loss':{},'grad_norm':{},'eval_runtime':{}}
for file in os.listdir('./exp/'):
    data=json.load(open('./exp/'+file))['log_history']
    file=file.split('.')[0]
    result_data['loss'][file]=data[len(data)-2]['loss']
    result_data['grad_norm'][file]=data[len(data)-2]['grad_norm']
    
    result_data['eval_loss'][file]=data[len(data)-1]['eval_loss']
    result_data['eval_runtime'][file]=data[len(data)-1]['eval_runtime']
df=pd.DataFrame(result_data)
df.to_csv('exp.csv')