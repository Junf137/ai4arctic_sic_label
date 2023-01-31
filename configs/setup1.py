#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   setup1.py
@Time    :   2023/01/27 10:18:31
@Author  :   Muhammed Patel
@Version :   1.0
@Contact :   m32patel@uwaterloo.ca
@License :   (C)Copyright 2022-2023, VIP Lab
@Desc    :   None
'''

_base_ = ['./base.py']

SCENE_VARIABLES = [
    # -- Sentinel-1 variables -- #
    'nersc_sar_primary',
    'nersc_sar_secondary',
    'sar_incidenceangle',
]


train_options = {'train_variables': SCENE_VARIABLES,
                 'epochs': 1,
                 'num_val_scenes': 1, }
