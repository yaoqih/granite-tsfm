# %%


# %% [markdown]
# # Getting started with TinyTimeMixer (TTM) with Channel-Mix Finetuning
# 
# This notebooke demonstrates the usage of finetuning a pre-trained `TinyTimeMixer` model with channel-mix finetuning. In channel-mix finetuning, we forecast the target_values based on the past values of the target columns and conditional columns and mixing them.
# 
# 
# For details related to model architecture, refer to the [TTM paper](https://arxiv.org/pdf/2401.03955.pdf).
# 
# In this example, we will use a pre-trained TTM-512-96 model. That means the TTM model can take an input of 512 time points (`context_length`), and can forecast upto 96 time points (`forecast_length`) in the future. We will use the pre-trained TTM in two settings:
# 1. **Zero-shot**: The pre-trained TTM will be directly used to evaluate on the `test` split of the target data. Note that the TTM was NOT pre-trained on the target data.
# 2. **Fine-tune**: The pre-trained TTM will be quickly fine-tuned onthe `train` split of the target data, and subsequently, evaluated on the `test` part of the target data. During finetuing, we used the values mentioned in `conditional_columns` as features for channel mixing. Search for `# ch_mix:` keyword for important parameters to edit for channel mixing finetuning. Set decoder_mode to "mix_channel" for channel-mix finetuning.
# 
# 
# Pre-trained TTM models will be fetched from the [Hugging Face TTM Model Repository](https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1).

# %% [markdown]
# ## Installation

# %%
# Install the tsfm library
# ! pip install "tsfm_public[notebooks] @ git+ssh://git@github.com/ibm-granite/granite-tsfm.git"

# %% [markdown]
# ## Imports

# %%
import comet_ml
import math
import os
import tempfile

import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from tsfm_public import (
    TimeSeriesPreprocessor,
    TrackingCallback,
    count_parameters,
    get_datasets,
)
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions

# %% [markdown]
# ## Important arguments

# %%
# Set seed for reproducibility
SEED = 42
set_seed(SEED)

# DATA ROOT PATH
# Make sure to download the target data (here ettm2) on the `DATA_ROOT_PATH` folder.
# ETT is available at: https://github.com/zhouhaoyi/ETDataset/tree/main
# target_dataset = "bike_sharing"
DATA_ROOT_PATH = "origin_data"

# Results dir
OUT_DIR = "ttm_finetuned_models/"


# Forecasting parameters
context_length = 24
forecast_length = 1

# %% [markdown]
# ## Data processing pipeline

# %%
# Load the data file and see the columns

timestamp_column = "date"
# timestamp_column = "timestamp"
id_columns = []

for file in os.listdir(DATA_ROOT_PATH):
    file_path = os.path.join(DATA_ROOT_PATH, file)
    data = pd.read_parquet(
        file_path,
        # parse_dates=[timestamp_column],
    )
    # 将date列转换为datetime类型
    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    while data[['open', 'high', 'low', 'close']].min().min() < 1:
        data['open'] += 1
        data['high'] += 1
        data['low'] += 1
        data['close'] += 1
    data['change_rate']=(data['open'].shift(-2)-data['open'].shift(-1))/(data['open'].shift(-1))
    data.dropna(inplace=True)
    # data['stock_id']=file.stem

    data[timestamp_column] = pd.to_datetime(data[timestamp_column])

    # Reset the index to ensure the hours are correctly assigned, as hour information is missing in the original timestamp of this df
    data[timestamp_column] = data[timestamp_column] + pd.to_timedelta(
        data.groupby(data[timestamp_column].dt.date).cumcount(), unit="h"
    )

    target_columns = ['change_rate']
    conditional_columns=['open','high','low','close','volume','amount','amplitude','pct_chg','change','turnover_rate']


    # print(data)
    # data = pd.read_csv(
    #     "/dccstor/tsfm23/datasets/exogs_expts/bike_sharing_dataset/processed_data/bike_sharing_hourly_processed.csv",
    #     # parse_dates=[timestamp_column],
    # )

    # print(data)

    # exog: Mention Exog channels in control_columns and target in target_columns

    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "control_columns": [],
        "conditional_columns":conditional_columns,
    }

    # split_params = {"train": [0, 0.5], "valid": [0.5, 0.75], "test": [0.75, 1.0]}

    # %%
    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    train_dataset, valid_dataset, test_dataset = get_datasets(
        tsp,
        data,
        # split_params,
    )

    # %%
    # train_dataset[3]

    # %% [markdown]
    # ## Zero-shot evaluation method

    # %%
    TTM_MODEL_PATH = "/root/granite-tsfm/tmp/TTM_cl-24_fl-1_pl-16_apl-6_ne-5_es-True/checkpoint/checkpoint-43950"

    zeroshot_model = get_model(
        TTM_MODEL_PATH,
        context_length=context_length,
        prediction_length=forecast_length,
        prediction_channel_indices=tsp.prediction_channel_indices,
        num_input_channels=tsp.num_input_channels,
    )
    zeroshot_model

    # %%
    tsp.prediction_channel_indices

    # %%
    tsp.num_input_channels

    # %%
    temp_dir = tempfile.mkdtemp()
    # zeroshot_trainer
    zeroshot_trainer = Trainer(
        model=zeroshot_model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=1024,
        ),
    )

    # %%
    print(zeroshot_trainer.evaluate(test_dataset))

    # %%
    # plot
    # plot_predictions(
    #     model=zeroshot_trainer.model,
    #     dset=test_dataset,
    #     plot_dir=os.path.join(OUT_DIR, "stock"),
    #     plot_prefix="test_zeroshot",
    #     channel=0,
    # )

    # %% [markdown]
    #  ## Few-shot finetune and evaluation method

    # %% [markdown]
    # ### Load model
    # Optionally, we can change some parameters of the model, e.g., dropout of the head.

    # %%
    finetune_forecast_model = get_model(
        TTM_MODEL_PATH,
        context_length=context_length,
        prediction_length=forecast_length,
        num_input_channels=tsp.num_input_channels,
        decoder_mode="mix_channel",  # ch_mix:  set to mix_channel for mixing channels in history
        prediction_channel_indices=tsp.prediction_channel_indices,
    )
    finetune_forecast_model

    # %% [markdown]
    # ### Frezze the TTM backbone

    # %%
    print(
        "Number of params before freezing backbone",
        count_parameters(finetune_forecast_model),
    )

    # Freeze the backbone of the model
    for param in finetune_forecast_model.backbone.parameters():
        param.requires_grad = False

    # Count params
    print(
        "Number of params after freezing the backbone",
        count_parameters(finetune_forecast_model),
    )

    # %% [markdown]
    # ### Finetune model with decoder mixing and exog fusion

    # %%
    # Important parameters
    num_epochs = 50  # Ideally, we need more epochs (try offline preferably in a gpu for faster computation)
    batch_size = 1024

    learning_rate, finetune_forecast_model = optimal_lr_finder(
        finetune_forecast_model,
        train_dataset,
        batch_size=batch_size,
        enable_prefix_tuning=False,
    )
    print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)

    # %%
    print(f"Using learning rate = {learning_rate}")
    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=8,
        report_to=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(OUT_DIR, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    # Optimizer and scheduler
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(train_dataset) / (batch_size)),
    )

    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )

    # Fine tune
    finetune_forecast_trainer.train()

    # %%
    print(finetune_forecast_trainer.evaluate(test_dataset))
    exit()
    # %%
    # plot_predictions(
    #     model=finetune_forecast_trainer.model,
    #     dset=test_dataset,
    #     plot_dir=os.path.join(OUT_DIR, "stock"),
    #     plot_prefix="test_finetune",
    #     channel=0,
    # )

    # %%



