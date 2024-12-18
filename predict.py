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
import akshare as ak
from ttm_pretrain_stock import custom_predict
data_path='./basic_data'
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
    if len(df)<250:
        continue
    # 将date列转换为datetime类型
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    while df[['open', 'high', 'low', 'close']].min().min() < 1:
        df['open'] += 1
        df['high'] += 1
        df['low'] += 1
        df['close'] += 1
    df['change_rate']=(df['open'].shift(-2)-df['open'].shift(-1))/(df['open'].shift(-1))
    # df.dropna(inplace=True)
    df.fillna(0,inplace=True)
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

dset_train, dset_valid, dset_test = get_datasets(tsp, final_df,split_config = {"train": '2024-01-01', "test": '2024-06-01'})
TTM_MODEL_PATH = "/root/granite-tsfm/tmp/TTM_cl-32_fl-1_pl-16_apl-6_ne-30_es-False/checkpoint/checkpoint-344472"

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


for last in range(2,6):
    for dataset_spilt,save_name in zip([dset_test],['test']):
        # print(trainer.evaluate(dataset))
        context_length=trainer.model.config.context_length
        # predictions_dict = trainer.predict(dataset_spilt)
        # flattened_array = predictions_dict.predictions[0][:,:,0].flatten()  # 变成一维数组，长度为324
        dfs=[]
        count=0
        dataset_spilt.datasets 
        batch = defaultdict(list)
        for dataset in dataset_spilt.datasets:
            item = dataset.get_last(-last)
            for key, value in item.items():
                batch[key].append(value)

        # Process the batch
        processed_batch = {}
        for key, values in batch.items():
            if isinstance(values[0], torch.Tensor):
                processed_batch[key] = torch.stack(values).to(trainer.model.device)
            elif isinstance(values[0], datetime):
                processed_batch[key] = values
            elif isinstance(values[0], tuple):
                processed_batch[key] = [item for sublist in values for item in sublist]
            else:
                processed_batch[key] = values

        flattened_array = custom_predict(trainer.model, [processed_batch])
        for index,dataset in tqdm(enumerate(dataset_spilt.datasets),save_name):
            # iterator = BatchIterator(dataset, batch_size)
            # flattened_array = predictions_dict.predictions[0][:,:,0].flatten()  # 变成一维数组，长度为324
            if last==1:
                df=dataset.revert_scaling(tsp,dataset.group_id[0]).iloc[-1:,:]
            else:
                df=dataset.revert_scaling(tsp,dataset.group_id[0]).iloc[-last:-last+1,:]
            
            # additional_elements = np.zeros(context_length)  # 或者其他你想要的值
            # final_array = np.concatenate([additional_elements, flattened_array[count:count+len(df)-context_length]])
            # final_array = np.concatenate([additional_elements, flattened_array])
            # count+=len(df)-context_length
            df['predict'] = flattened_array[index]*tsp.target_scaler_dict[dataset.group_id[0]].scale_+tsp.target_scaler_dict[dataset.group_id[0]].mean_
            # df = df.drop(columns=['group'])
            # reset_index 可以将结果的索引重置为默认的整数索引
            df.reset_index(drop=True, inplace=True)
            dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)
        final_df=final_df[['stock_id','date','predict']]
        final_df['code'] = final_df['stock_id'].str[-6:]
        df2=ak.stock_info_a_code_name()
        # 2. 使用merge函数将df1和df2合并
        final_df = final_df.merge(df2[['code', 'name']], 
                        on='code',  # 使用code列作为合并键
                        how='left'  # 使用左连接保留df1的所有记录
                    )

        # 3. 如果不再需要临时创建的code列，可以删除
        final_df = final_df.drop('code', axis=1)
        date=final_df['date'][0].strftime('%Y-%m-%d')
        final_df.to_csv(f"./predict_result/{date}.csv", index=False)  # 保存为Excel文件

