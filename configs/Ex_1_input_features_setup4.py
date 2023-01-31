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
    # 'sar_incidenceangle',

    # -- Geographical variables -- #
    # 'distance_map',

    # -- AMSR2 channels -- #
    # 'btemp_6_9h', 'btemp_6_9v',
    # 'btemp_7_3h', 'btemp_7_3v',
    # 'btemp_10_7h', 'btemp_10_7v',
    # 'btemp_18_7h', 'btemp_18_7v',
    # 'btemp_23_8h', 'btemp_23_8v',
    # 'btemp_36_5h', 'btemp_36_5v',
    # 'btemp_89_0h', 'btemp_89_0v',

    # -- Environmental variables -- #
    # 'u10m_rotated', 'v10m_rotated',
    't2m', 
    # 'skt', 'tcwv', 'tclw'
]


train_options = {'train_variables': SCENE_VARIABLES,
                 'epochs': 40,
                 'num_workers': 0,  # Number of parallel processes to fetch data.
                 'num_workers_val': 0,  # Number of parallel processes during validation.
                }
