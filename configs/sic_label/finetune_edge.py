#!/usr/bin/env python
# -*-coding:utf-8 -*-

from mmengine.config import read_base

# Pure Python style Configuration File
with read_base():
    from .._base_.base import *
    from .sic_label import *

# Pure Text style Configuration File
# _base_ = ["../_base_/base.py"]


train_options.update(
    {
        "weight_map": {
            # Enable weight map for different stages.
            "train": True,
            "val": False,
            "test": False,
            # Configuration for creating the weight map.
            "ksize": 9,  # hard-coded kernel size
            "weights": {
                "SIC": {
                    "invalid": 0,
                    "inner_edges": 1,
                    "ice_cfv_edges": 1,
                    "ice_water_edges": 1,
                    "center": 0,
                },
                "SOD": {
                    "invalid": 0,
                    "inner_edges": 1,
                    "ice_cfv_edges": 1,
                    "ice_water_edges": 1,
                    "center": 0,
                },
                "FLOE": {
                    "invalid": 0,
                    "inner_edges": 1,
                    "ice_cfv_edges": 1,
                    "ice_water_edges": 1,
                    "center": 0,
                },
            },
        },
    }
)
