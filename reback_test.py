import logging
import math
import os
import tempfile
import comet_ml
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.models.tinytimemixer import (
    TinyTimeMixerConfig,
    TinyTimeMixerForPrediction,
)
from tsfm_public.models.tinytimemixer.utils import get_ttm_args
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions
from pathlib import Path
from tqdm import tqdm
from transformers.trainer_utils import get_last_checkpoint
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from ttm_pretrain_stock import calculate_max_drawdown_simple,BatchIterator,custom_predict
from scipy import stats

data_path='./origin_data'
timestamp_column = "date"
id_columns = ['stock_id']  # mention the ids that uniquely identify a time-series.

target_columns = ['change_rate']
conditional_columns=['open','high','low','close','volume','amount','amplitude','pct_chg','change','turnover_rate']

parquet_files = list(Path(data_path).glob('*.parquet'))

# 创建一个空列表来存储所有数据框
dfs = []

# 读取每个parquet文件并处理
for file in tqdm(parquet_files,'reading parquet files'):
    # 读取parquet文件
    df = pd.read_parquet(file)
    
    # 将date列转换为datetime类型
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    while df[['open', 'high', 'low', 'close']].min().min() < 1:
        df['open'] += 1
        df['high'] += 1
        df['low'] += 1
        df['close'] += 1
    df['change_rate']=(df['open'].shift(-2)-df['open'].shift(-1))/(df['open'].shift(-1))
    df.dropna(inplace=True)
    df['stock_id']=file.stem
    # 将处理后的数据框添加到列表中
    # df = df[df['date'] >= datetime(2009, 1, 1)]

    dfs.append(df)

# 合并所有数据框
final_df = pd.concat(dfs, ignore_index=True)
args = get_ttm_args()

column_specifiers = {
    "timestamp_column": timestamp_column,
    "id_columns": id_columns,
    "target_columns": target_columns,
    "control_columns": [],
    "conditional_columns":conditional_columns,
}

tsp = TimeSeriesPreprocessor(
    **column_specifiers,
    context_length=args.context_length,
    prediction_length=args.forecast_length,
    scaling=True,
    encode_categorical=False,
    scaler_type="standard",
)

dset_train, dset_valid, dset_test = get_datasets(tsp, final_df,split_config = {"train": '2022-01-01', "test": '2023-01-01'})
TTM_MODEL_PATH = "/root/granite-tsfm/tmp/TTM_cl-32_fl-1_pl-16_apl-6_ne-30_es-False/checkpoint/checkpoint-314563"

zeroshot_model = get_model(
    TTM_MODEL_PATH,
    context_length=args.context_length,
    prediction_length=args.forecast_length,
    prediction_channel_indices=tsp.prediction_channel_indices,
    num_input_channels=tsp.num_input_channels,
)
temp_dir = tempfile.mkdtemp()
trainer = Trainer(
    model=zeroshot_model,
    args=TrainingArguments(
        output_dir=temp_dir,
        per_device_eval_batch_size=1000,
    ),
)
batch_size = 1000
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from datetime import datetime
result_save=[]
for dataset_spilt,save_name in zip([dset_valid,dset_test],['valid','test']):
    dfs=[]
    # print(trainer.evaluate(dataset))
    context_length=trainer.model.config.context_length
    # predictions_dict = trainer.predict(dataset_spilt)
    # flattened_array = predictions_dict.predictions[0][:,:,0].flatten()  # 变成一维数组，长度为324
    for dataset in tqdm(dataset_spilt.datasets,save_name):
        iterator = BatchIterator(dataset, batch_size)
        flattened_array = custom_predict(trainer.model, iterator)
        # flattened_array = predictions_dict.predictions[0][:,:,0].flatten()  # 变成一维数组，长度为324
        df=dataset.revert_scaling(tsp,dataset.group_id[0])
        additional_elements = np.zeros(context_length)  # 或者其他你想要的值
        final_array = np.concatenate([additional_elements, flattened_array])
        df['predict'] = final_array*tsp.target_scaler_dict[dataset.group_id[0]].scale_+tsp.target_scaler_dict[dataset.group_id[0]].mean_
        # df = df.drop(columns=['group'])
        df= df.iloc[context_length:]
        # reset_index 可以将结果的索引重置为默认的整数索引
        df.reset_index(drop=True, inplace=True)
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)
    # final_df.to_csv(f"zero_shot_{save_name}.csv", index=False)  # 保存为Excel文件

    grouped = final_df.groupby('date')
    ids=[]
    for id,group in grouped:
        if len(group)>1000:
            id_=group['predict'].idxmax()
            ids.append(id_)
    result = final_df.loc[ids]
    # 将结果保存到Excel文件
    # result.to_excel(f'highest_predict_{save_name}.xlsx', index=False)
    print(f'{save_name}_mean',result['change_rate'].mean())
    result_save.append(result['change_rate'].mean())
    prices=[1]
    for i in range(len(result)):
        prices.append(prices[-1]*(1+result.iloc[i]['change_rate']))
    print(f'{save_name}_max_drawdown',calculate_max_drawdown_simple(prices))
    print(f'{save_name}_std',result['change_rate'].std())
    print(f'{save_name}_end_price',prices[-1])
    result_save.append(calculate_max_drawdown_simple(prices))
    result_save.append(result['change_rate'].std())
    result_save.append(prices[-1])

    df_sorted = result.sort_values(by='predict', ascending=False)
    # 2. 最小二乘法拟合
    x = np.arange(len(df_sorted))  # x轴为索引
    y = df_sorted['change_rate'].values  # y轴为change_rate值
    # 进行线性拟合
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # 拟合方程: y = slope * x + intercept
    # 3. 计算y=0时的x值
    x_zero = -intercept / slope

    # 4. 找到最接近的且大于0的点
    x_nearest = int(x_zero) if int(x_zero)*slope+intercept>0 else int(x_zero) - 1
    x_nearest = min(max(0, x_nearest), len(df_sorted) - 1)

    # 获取对应的predict值
    result_predict = df_sorted.iloc[x_nearest]['predict']
    print(f'{save_name}_trand_point',result_predict)
    result_save.append(result_predict)
print(result_save)
