#!/usr/bin/env python
# -*-coding:utf-8 -*-

from mmengine.config import read_base

# Pure Python style Configuration File
with read_base():
    from .._base_.base import *

# Pure Text style Configuration File
# _base_ = ["../_base_/base.py"]


train_options.update(
    {
        # -- General configuration -- #
        "path_to_train_data": "/home/j46lei/projects/rrg-dclausi/ai4arctic/dataset/dataset_ready_to_train/train",
        "path_to_test_data": "/home/j46lei/projects/rrg-dclausi/ai4arctic/dataset/dataset_ready_to_train/test",
        "train_list_path": "datalists/dataset.json",  # Training dataset list.
        "val_path": "datalists/valset2.json",  # Validation dataset list. Disabled when cross_val_run is True.
        "test_path": "datalists/testset.json",  # Test dataset list.
        # -- Experiment configuration -- #
        "train_variables": [
            # -- Sentinel-1 variables -- #
            "nersc_sar_primary",
            "nersc_sar_secondary",
            "sar_incidenceangle",
            # -- Geographical variables -- #
            "distance_map",
            # -- AMSR2 channels -- #
            "btemp_18_7h",
            "btemp_18_7v",
            "btemp_36_5h",
            "btemp_36_5v",
            # -- Environmental variables -- #
            "u10m_rotated",
            "v10m_rotated",
            "t2m",
            "tcwv",
            "tclw",
            # -- Auxiliary Variables -- #
            "aux_time",
            "aux_lat",
            "aux_long",
        ],
        "patience": 0,  # Number of epochs to wait before early stopping. disabled when set to 0.
        "cross_val_run": False,
        "p-out": 20,  # number of scenes taken from the TRAIN SET as the validation set.
        "compute_classwise_f1score": False,
        "plot_confusion_matrix": True,
        "load_proc": 6,  # Number of parallel processes when loading data.
        "num_workers": 6,  # Number of parallel processes to fetch data.
        "num_workers_val": 6,  # Number of parallel processes during validation.
        "down_sample_scale": 10,
        "deterministic": False,
        # -- SIC weight map configuration -- #
        "sic_weight_map": {
            "train": True,
            "val": True,
            "test": False,
            "ksize": 5,  # hard-coded kernel size
            "ksize_ratio": 100,  # ratio of the (image size / kernel size), disabled when ksize is set.
            "edge_threshold": 0,  # threshold for determining the mask after sobel filter
            "edge_weights": {
                "invalid": 0,
                "inner_edges": 0.5,
                "ice_cfv_edges": 0.5,
                "ice_water_edges": 1,
                "center": 1,
            },
            "visualization": False,
            # Path to save visualization. Only works when visualization is True.
            "visualization_save_path": "output/visualization",
        },
        # -- Training configuration -- #
        "epochs": 300,
        "epoch_len": 250,  # Number of batches for each epoch.
        "patch_size": 256,
        "batch_size": 32,
        "model_selection": "unet_regression",  #'unet_feature_fusion', #'unet_regression',
        "unet_conv_filters": [32, 32, 64, 64],
        "optimizer": {
            "type": "SGD",
            "lr": 0.001,  # Optimizer learning rate.
            "momentum": 0.9,
            "dampening": 0,
            "nesterov": False,
            "weight_decay": 0.01,
        },
        "scheduler": {
            "type": "CosineAnnealingWarmRestartsLR",
            "EpochsPerRestart": 20,  # Number of epochs for the first restart
            # This number will be used to increase or decrease the number of epochs to restart after each restart.
            "RestartMult": 1,
            "lr_min": 0,
        },
        "data_augmentations": {
            "Random_h_flip": 0.5,
            "Random_v_flip": 0.5,
            "Random_rotation_prob": 0.5,
            "Random_rotation": 90,
            "Random_scale_prob": 0.5,
            "Random_scale": (0.9, 1.1),
            "Cutmix_beta": 1.0,
            "Cutmix_prob": 0.5,
        },
        # -- Loss configuration -- #
        "task_weights": [1, 3, 3],
        "chart_loss": {
            "SIC": {
                "type": "MSELossWithIgnoreIndex",
                "ignore_index": 255,
            },
            "SOD": {
                "type": "CrossEntropyLoss",
                "ignore_index": 255,
            },
            "FLOE": {
                "type": "CrossEntropyLoss",
                "ignore_index": 255,
            },
        },
        # Metric functions for each ice parameter and the associated weight.
        "chart_metric": {
            "SIC": {
                "func": r2_metric,
                "weight": 2,
            },
            "SOD": {
                "func": f1_metric,
                "weight": 2,
            },
            "FLOE": {
                "func": f1_metric,
                "weight": 1,
            },
        },
    }
)
