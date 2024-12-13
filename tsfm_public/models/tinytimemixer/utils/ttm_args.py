# Copyright contributors to the TSFM project
#
"""Utilities for TTM notebooks"""

import argparse
import logging
import os

import torch


logger = logging.getLogger(__name__)


def get_ttm_args():
    parser = argparse.ArgumentParser(description="TTM pretrain arguments.")
    # Adding a positional argument
    parser.add_argument(
        "--forecast_length",
        "-fl",
        type=int,
        required=False,
        default=1,
        help="Forecast length",
    )
    parser.add_argument(
        "--context_length",
        "-cl",
        type=int,
        required=False,
        default=32,
        help="History context length",
    )
    parser.add_argument(
        "--patch_length",
        "-pl",
        type=int,
        required=False,
        default=16,
        help="Patch length",
    )
    parser.add_argument(
        "--adaptive_patching_levels",
        "-apl",
        type=int,
        required=False,
        default=6,
        help="Number of adaptive patching levels of TTM",
    )
    parser.add_argument(
        "--d_model_scale",
        "-dms",
        type=int,
        required=False,
        default=12,
        help="Model hidden dimension",
    )
    parser.add_argument(
        "--decoder_d_model_scale",
        "-ddms",
        type=int,
        required=False,
        default=12,
        help="Decoder hidden dimension",
    )
    parser.add_argument(
        "--num_gpus",
        "-ng",
        type=int,
        required=False,
        default=1,
        help="Number of GPUs",
    )
    parser.add_argument("--random_seed", "-rs", type=int, required=False, default=42, help="Random seed")
    parser.add_argument("--batch_size", "-bs", type=int, required=False, default=1024, help="Batch size")
    parser.add_argument(
        "--num_epochs",
        "-ne",
        type=int,
        required=False,
        default=30,
        help="Number of epochs",
    )

    parser.add_argument(
        "--num_workers",
        "-nw",
        type=int,
        required=False,
        default=8,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        required=False,
        default=0.1,
        help="Learning rate",
    )
    parser.add_argument(
        "--dataset",
        "-ds",
        type=str,
        required=False,
        default="stock",
        help="Dataset",
    )
    parser.add_argument(
        "--data_root_path",
        "-drp",
        type=str,
        required=False,
        default="datasets/",
        help="Dataset",
    )
    parser.add_argument(
        "--save_dir",
        "-sd",
        type=str,
        required=False,
        default="tmp/",
        help="Data path",
    )
    parser.add_argument(
        "--early_stopping",
        "-es",
        type=int,
        required=False,
        default=0,
        help="Whether to use early stopping during finetuning.",
    )
    parser.add_argument(
        "--freeze_backbone",
        "-fb",
        type=int,
        required=False,
        default=1,
        help="Whether to freeze the backbone during few-shot finetuning.",
    )
    parser.add_argument(
        "--enable_prefix_tuning",
        "-ept",
        type=int,
        required=False,
        default=1,
        help="Enable prefix tuning in TTM.",
    )
    parser.add_argument(
        "--hf_model_path",
        "-hmp",
        type=str,
        required=False,
        default="ibm-granite/granite-timeseries-ttm-r2",
        help="Hugginface model card path.",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="List of datasets, comma separated",
    )

    parser.add_argument(
        "--zeroshot",
        "-zs",
        type=int,
        required=False,
        default=1,
        help="Run Zeroshot",
    )

    parser.add_argument(
        "--fewshot",
        "-fs",
        type=int,
        required=False,
        default=1,
        help="Run Zeroshot",
    )

    parser.add_argument(
        "--plot",
        type=int,
        required=False,
        default=1,
        help="Enable plotting",
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        required=False,
        default=3,
        help="Number of  layers",
    )

    parser.add_argument(
        "--decoder_num_layers",
        type=int,
        required=False,
        default=4,
        help="Number of  decoder layers",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        required=False,
        default=0.2,
        help="Dropout",
    )

    parser.add_argument(
        "--head_dropout",
        type=float,
        required=False,
        default=0.2,
        help="head_dropout",
    )

    # Parsing the arguments
    args = parser.parse_args()
    args.early_stopping = int_to_bool(args.early_stopping)
    args.freeze_backbone = int_to_bool(args.freeze_backbone)
    args.enable_prefix_tuning = int_to_bool(args.enable_prefix_tuning)
    args.d_model = args.patch_length * args.d_model_scale
    args.decoder_d_model = args.patch_length * args.decoder_d_model_scale

    # Calculate number of gpus
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
        logger.info(f"Automatically calculated number of GPUs ={args.num_gpus}")

    # Create save directory
    args.save_dir = os.path.join(
        args.save_dir,
        f"TTM_cl-{args.context_length}_fl-{args.forecast_length}_pl-{args.patch_length}_apl-{args.adaptive_patching_levels}_ne-{args.num_epochs}_es-{args.early_stopping}",
    )
    os.makedirs(args.save_dir, exist_ok=True)

    return args


def int_to_bool(value):
    if value == 0:
        return False
    elif value == 1:
        return True
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (0 or 1)")
