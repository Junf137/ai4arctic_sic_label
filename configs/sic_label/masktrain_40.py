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
            "train": True,
            "val": False,
            "test": False,
            "weights": {
                "SIC": {
                    "invalid": 0,
                    "inner_edges": 40.0,
                    "ice_cfv_edges": 40.0,
                    "ice_water_edges": 40.0,
                    "center": 1,
                },
                "SOD": {
                    "invalid": 0,
                    "inner_edges": 40.0,
                    "ice_cfv_edges": 40.0,
                    "ice_water_edges": 40.0,
                    "center": 1,
                },
                "FLOE": {
                    "invalid": 0,
                    "inner_edges": 40.0,
                    "ice_cfv_edges": 40.0,
                    "ice_water_edges": 40.0,
                    "center": 1,
                },
            },
        },
    }
)
