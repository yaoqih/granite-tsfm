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
TTM_MODEL_PATH = "/root/granite-tsfm/tmp/TTM_cl-32_fl-1_pl-16_apl-6_ne-30_es-True/checkpoint/checkpoint-207860"

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
        per_device_eval_batch_size=800,
    ),
)
batch_size = 800
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from datetime import datetime

class BatchIterator:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_index = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.dataset):
            raise StopIteration

        batch = defaultdict(list)
        for i in range(self.current_index, min(self.current_index + self.batch_size, len(self.dataset))):
            item = self.dataset[i]
            for key, value in item.items():
                batch[key].append(value)

        self.current_index += self.batch_size

        # Process the batch
        processed_batch = {}
        for key, values in batch.items():
            if isinstance(values[0], torch.Tensor):
                processed_batch[key] = torch.stack(values).to(self.device)
            elif isinstance(values[0], datetime):
                processed_batch[key] = values
            elif isinstance(values[0], tuple):
                processed_batch[key] = [item for sublist in values for item in sublist]
            else:
                processed_batch[key] = values

        return processed_batch

def custom_predict(model, dataset):
    model.eval()
    predictions = []
    
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in dataset:
            # Move batch to the same device as the model
            batch = {k: v.to(model.device) for k, v in batch.items() if type(v) == torch.Tensor}
            
            # Get predictions
            outputs = model(**batch)
            
            # Assuming you want logits, adjust as needed
            batch_predictions = outputs.prediction_outputs[:,:,0].flatten().cpu().numpy()
            predictions.extend(batch_predictions)
 
    
    return predictions
for dataset_spilt,save_name in zip([dset_valid,dset_test],['valid','test']):
    # print(trainer.evaluate(dataset))
    context_length=trainer.model.config.context_length
    # predictions_dict = trainer.predict(dataset_spilt)
    # flattened_array = predictions_dict.predictions[0][:,:,0].flatten()  # 变成一维数组，长度为324
    dfs=[]
    count=0
    for dataset in tqdm(dataset_spilt.datasets,save_name):
        iterator = BatchIterator(dataset, batch_size)
        flattened_array = custom_predict(trainer.model, iterator)
        # flattened_array = predictions_dict.predictions[0][:,:,0].flatten()  # 变成一维数组，长度为324
        df=dataset.revert_scaling(tsp,dataset.group_id[0])
        additional_elements = np.zeros(context_length)  # 或者其他你想要的值
        # final_array = np.concatenate([additional_elements, flattened_array[count:count+len(df)-context_length]])
        final_array = np.concatenate([additional_elements, flattened_array])
        # count+=len(df)-context_length
        df['predict'] = final_array*tsp.target_scaler_dict[dataset.group_id[0]].scale_+tsp.target_scaler_dict[dataset.group_id[0]].mean_
        # df = df.drop(columns=['group'])
        df= df.iloc[context_length:]
        # reset_index 可以将结果的索引重置为默认的整数索引
        df.reset_index(drop=True, inplace=True)
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv(f"zero_shot_{save_name}.csv", index=False)  # 保存为Excel文件

    grouped = final_df.groupby('date')
    # 筛选出每组数量大于1000的组
    filtered_groups = grouped.filter(lambda x: len(x) > 1000)
    # 从筛选后的组中找到'predict'列值最高的行
    result = filtered_groups.loc[filtered_groups.groupby('date')['predict'].idxmax()]
    result.to_excel(f'highest_predict_{save_name}.xlsx', index=False)
    # 将结果保存到Excel文件中
