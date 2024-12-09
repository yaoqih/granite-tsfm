#!/usr/bin/env python
# coding: utf-8

import logging
import math
import os
import tempfile

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
    if os.path.exists(os.path.join(args.save_dir, "checkpoint") and resume):
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
        save_total_limit=3,
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

    # Set trainer
    if args.early_stopping:
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            optimizers=(optimizer, scheduler),
            callbacks=[early_stopping_callback],
        )
    else:
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            optimizers=(optimizer, scheduler),
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
    data_path='./origin_data'
    timestamp_column = "date"
    id_columns = ['stock_id']  # mention the ids that uniquely identify a time-series.

    target_columns = ['change_rate']
    conditional_columns=['open','high','low','close','volume','amount','amplitude','pct_chg','change','turnover_rate']

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

    dset_train, dset_valid, dset_test = get_datasets(tsp, final_df)

    # Get model
    model = get_base_model(args)
    print(model)
    # Pretrain
    model_save_path = pretrain(args, model, dset_train, dset_valid)
    print("=" * 20, "Pretraining Completed!", "=" * 20)
    print("Model saved in location:", model_save_path)

    # inference

    inference(args=args, model_path=model_save_path, dset_test=dset_test)

    print("inference completed..")
