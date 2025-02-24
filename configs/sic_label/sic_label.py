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
        "val_path": "datalists/valset2.json",  # Validation dataset list.
        "test_path": "datalists/dataset_test_gt_embedded.json",  # Test dataset list.
        # -- Experiment configuration -- #
        "cross_val_run": True,
        # TODO: what does this means?
        "p-out": 12,  # number of scenes taken from the TRAIN SET. Must change the datalist to move validation scenes to train if using
        "compute_classwise_f1score": True,
        "plot_confusion_matrix": True,
        "num_workers": 4,  # Number of parallel processes to fetch data.
        "num_workers_val": 4,  # Number of parallel processes during validation.
        "down_sample_scale": 10,
        # -- Training configuration -- #
        "epochs": 300,
        "batch_size": 16,
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
    }
)
