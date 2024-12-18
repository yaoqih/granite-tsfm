#!/usr/bin/env python
# coding: utf-8

import logging
import math
import os
import tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from transformers import TrainerCallback
import numpy as np
import torch
from collections import defaultdict
from scipy import stats


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
def calculate_max_drawdown_simple(prices):
    """
    计算最大回撤的简化版本
    :param prices: 价格序列列表
    :return: 最大回撤比例
    """
    if not prices:
        return 0
        
    max_drawdown = 0
    peak = prices[0]
    
    for price in prices:
        if price > peak:
            peak = price
        else:
            drawdown = (peak - price) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
    return max_drawdown
class CustomCallback(TrainerCallback):
    def __init__(self, data_valid, data_test, model_save_path,batch_size, **kwargs):
        self.data_valid = data_valid
        self.data_test = data_test
        self.model_save_path = model_save_path
        self.batch_size=batch_size
        self.start_dropout=args.dropout
        self.num_epochs=args.num_epochs
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """
        在每个epoch开始时被调用，更新dropout rate
        """
        current_epoch = state.epoch
        if current_epoch < 0:
            # 在开始阶段保持初始dropout
            dropout = self.start_dropout
        else:
            # 计算当前epoch的dropout rate
            progress = (current_epoch) / (self.num_epochs )
            # 使用cosine函数实现先快后慢的衰减
            dropout = 0.5 * (self.start_dropout ) * (1 + math.cos(math.pi * progress))
        
        # 更新模型中所有dropout层的比例
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout
        print(f"Epoch {int(current_epoch)}: Dropout set to {dropout:.4f}")

    def on_save(self, args, state, control, **kwargs):
        # 在每个epoch结束时调用自定义函数
        results_all=[state.epoch]
        for dataset_spilt,save_name in zip([self.data_valid,self.data_test],['valid','test']):
            # print(trainer.evaluate(dataset))
            context_length=kwargs['model'].config.context_length
            # predictions_dict = trainer.predict(dataset_spilt)
            # flattened_array = predictions_dict.predictions[0][:,:,0].flatten()  # 变成一维数组，长度为324
            dfs=[]
            count=0
            for dataset in tqdm(dataset_spilt.datasets,save_name):
                iterator = BatchIterator(dataset, self.batch_size)
                flattened_array = custom_predict(kwargs['model'], iterator)
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
            grouped = final_df.groupby('date')
            ids=[]
            for id,group in grouped:
                if len(group)>1000:
                    id_=group['predict'].idxmax()
                    ids.append(id_)
            result = final_df.loc[ids]
            results_all.append(result['change_rate'].mean())
            prices=[1]
            for i in range(len(result)):
                prices.append(prices[-1]*(1+result.iloc[i]['change_rate']))
            results_all.append(calculate_max_drawdown_simple(prices))
            results_all.append(result['change_rate'].std())
            results_all.append(prices[-1])

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
            results_all.append(result_predict)

        open(os.path.join(self.model_save_path,'result.csv'),'a').write(','.join([str(i) for i in results_all])+'\n')
            # 将结果保存到Excel文件
    def on_train_begin(self, args, state, control, **kwargs):
        if not os.path.exists(os.path.join(self.model_save_path,'result.csv')):        
            open(os.path.join(self.model_save_path,'result.csv'),'w').write('epoch,valid_mean,valid_max_drawdown,valid_std,valid_end_price,valid_trand_point,test_mean,test_max_drawdown,test_std,test_end_price,test_trand_point\n')

logger = logging.getLogger(__file__)
# TTM pre-training example.
# This scrips provides a toy example to pretrain a Tiny Time Mixer (TTM) model on
# the `etth1` dataset. For pre-training TTM on a much large set of datasets, please
# have a look at our paper: https://arxiv.org/pdf/2401.03955.pdf
# If you want to directly utilize the pre-trained models. Please use them from the
# Hugging Face Hub: https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1
# Have a look at the fine-tune scripts for example usecases of the pre-trained
# TTM models.

# Basic usage:
# python ttm_pretrain_sample.py --data_root_path datasets/
# See the get_ttm_args() function to know more about other TTM arguments
resume=True

def get_base_model(args):
    # Pre-train a `TTM` forecasting model
    config = TinyTimeMixerConfig(
        context_length=args.context_length,
        prediction_length=args.forecast_length,
        patch_length=args.patch_length,
        num_input_channels=11,
        patch_stride=args.patch_length,
        d_model=args.d_model,
        num_layers=args.num_layers,  # increase the number of layers if we want more complex models
        mode="mix_channel",
        expansion_factor=2,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        scaling="std",
        gated_attn=True,
        adaptive_patching_levels=args.adaptive_patching_levels,
        # decoder params
        decoder_num_layers=args.decoder_num_layers,  # increase the number of layers if we want more complex models
        decoder_adaptive_patching_levels=0,
        decoder_mode="mix_channel",
        decoder_raw_residual=False,
        use_decoder=True,
        decoder_d_model=args.decoder_d_model,
        prediction_channel_indices=[0],
    )

    model = TinyTimeMixerForPrediction(config)
    return model


def pretrain(args, model, dset_train, dset_val):
    # Find optimal learning rate
    # Use with caution: Set it manually if the suggested learning rate is not suitable

    learning_rate, model = optimal_lr_finder(
        model,
        dset_train,
        batch_size=args.batch_size,
    )
    # learning_rate = args.learning_rate
    last_checkpoint = None
    if os.path.exists(os.path.join(args.save_dir, "checkpoint")) and resume:
        last_checkpoint = get_last_checkpoint(os.path.join(args.save_dir, "checkpoint"))
    

    
    print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)

    trainer_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "checkpoint"),
        overwrite_output_dir=True if not resume else False,
        learning_rate=learning_rate,
        num_train_epochs=args.num_epochs,
        seed=args.random_seed,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=None,
        logging_dir=os.path.join(args.save_dir, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
    )
    print("tensorboard --logdir="+os.path.join(args.save_dir, "logs")+' --bind_all')

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=args.num_epochs,
        steps_per_epoch=math.ceil(len(dset_train) / args.batch_size),
        # steps_per_epoch=math.ceil(len(dset_train) / (args.batch_size * args.num_gpus)),
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )
    customcallback=CustomCallback(data_valid=dset_val,data_test=dset_test,model_save_path=args.save_dir,batch_size=args.batch_size,args=args)
    # Set trainer
    if args.early_stopping:
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            optimizers=(optimizer, scheduler),
            callbacks=[early_stopping_callback,customcallback],
        )
    else:
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            optimizers=(optimizer, scheduler),
            callbacks=[customcallback],
        )

    # Train
    if resume:
        print("Resuming from checkpoint:",last_checkpoint)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save the pretrained model

    model_save_path = os.path.join(args.save_dir, "ttm_pretrained")
    trainer.save_model(model_save_path)
    return model_save_path


def inference(args, model_path, dset_test):
    model = get_model(model_path=model_path)

    temp_dir = tempfile.mkdtemp()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=args.batch_size,
            seed=args.random_seed,
            report_to="none",
        ),
    )
    # evaluate = zero-shot performance
    print("+" * 20, "Test MSE output:", "+" * 20)
    # output = trainer.evaluate(dset_test)
    # print(output)

    # get predictions

    # predictions_dict = trainer.predict(dset_test)

    # predictions_np = predictions_dict.predictions[0]

    # print(predictions_np.shape)

    # get backbone embeddings (if needed for further analysis)

    # backbone_embedding = predictions_dict.predictions[1]

    # print(backbone_embedding.shape)

    plot_path = os.path.join(args.save_dir, "plots")
    # plot
    plot_predictions(
        model=trainer.model,
        dset=dset_test,
        plot_dir=plot_path,
        plot_prefix="test_inference",
        channel=0,
        num_plots=100,
    )
    print("Plots saved in location:", plot_path)


if __name__ == "__main__":
    # Arguments
    args = get_ttm_args()

    # Set seed
    set_seed(args.random_seed)

    logger.info(
        f"{'*' * 20} Pre-training a TTM for context len = {args.context_length}, forecast len = {args.forecast_length} {'*' * 20}"
    )

    # Data prep
    # Dataset
    # TARGET_DATASET = "etth1"
    data_path='./basic_data'
    timestamp_column = "date"
    id_columns = ['stock_id']  # mention the ids that uniquely identify a time-series.

    target_columns = ['change_rate']
    conditional_columns=['open','high','low','close','volume','amount','amplitude','pct_chg','change','turnover_rate']
    # conditional_columns=['open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_chg', 'change', 'turnover_rate', 'CDL3OUTSIDE', 'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHOMINGPIGEON', 'CDLINVERTEDHAMMER', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMORNINGSTAR', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLTAKURI', 'CDLTHRUSTING', 'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA', 'MIDPOINT', 'MIDPRICE', 'SAR', 'SAREXT', 'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA', 'ADX', 'ADXR', 'APO', 'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX', 'MACDmacdhist', 'MACDmacdsignal', 'MACDmacd', 'MACDEXTmacdhist', 'MACDEXTmacdsignal', 'MACDEXTmacd', 'MACDFIXmacdhist', 'MACDFIXmacdsignal', 'MACDFIXmacd', 'MFI', 'MINUS_DI', 'MINUS_DM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 'TRIX', 'ULTOSC', 'WILLR', 'AD', 'ADOSC', 'OBV', 'ATR', 'NATR', 'TRANGE', 'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASORquadrature', 'HT_PHASORinphase', 'HT_SINEleadsine', 'HT_SINEsine', 'STOCHslowd', 'STOCHslowk', 'STOCHFfastd', 'STOCHFfastk', 'STOCHRSIfastd', 'STOCHRSIfastk', 'AROONaroonup', 'AROONaroondown', 'BBANDSlowerband', 'BBANDSmiddleband', 'BBANDSupperband', 'BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR']
    
    # mention the train, valid and split config.
    # split_config = {
    #     "train": [0, 8640],
    #     "valid": [8640, 11520],
    #     "test": [
    #         11520,
    #         14400,
    #     ],
    # }

    parquet_files = list(Path(data_path).glob('*.parquet'))

    # 创建一个空列表来存储所有数据框
    dfs = []

    # 读取每个parquet文件并处理
    for file in tqdm(parquet_files,'reading parquet files'):
        # 读取parquet文件
        df = pd.read_parquet(file)
        if len(df)<200:
            continue
        
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

    dset_train, dset_valid, dset_test = get_datasets(tsp, final_df,split_config = {"train": '2024-01-01', "test": '2024-06-01'},all_train=True)
    # Get model
    model = get_base_model(args)
    # open('model.txt','w').write(str(model))
    # print(model)
    # Pretrain
    model_save_path = pretrain(args, model, dset_train, dset_valid)
    print("=" * 20, "Pretraining Completed!", "=" * 20)
    print("Model saved in location:", model_save_path)

    # inference

    # inference(args=args, model_path=model_save_path, dset_test=dset_test)

    print("inference completed..")
