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
        "deterministic": True,
        "sic_label_mask": {
            "train": False,
            "val": False,
            "test": False,
        },
    }
)
