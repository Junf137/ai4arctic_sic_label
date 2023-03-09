#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helping functions for 'introduction' and 'quickstart' notebooks."""

# -- File info -- #
__author__ = 'Andreas R. Stokholm'
__contributors__ = ''
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk']
__version__ = '1.0.0'
__date__ = '2022-10-17'

# -- Built-in modules -- #
import os

# -- Third-party modules -- #
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
# from sklearn.metrics import r2_score, f1_score
from torchmetrics.functional import r2_score, f1_score

# -- Proprietary modules -- #
from utils import ICE_STRINGS, GROUP_NAMES


def chart_cbar(ax, n_classes, chart, cmap='vridis'):
    """
    Create discrete colourbar for plot with the sea ice parameter class names.

    Parameters
    ----------
    n_classes: int
        Number of classes for the chart parameter.
    chart: str
        The relevant chart.
    """
    arranged = np.arange(0, n_classes)
    cmap = plt.get_cmap(cmap, n_classes - 1)
    # Get colour boundaries. -0.5 to center ticks for each color.
    norm = mpl.colors.BoundaryNorm(arranged - 0.5, cmap.N)
    arranged = arranged[:-1]  # Discount the mask class.
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=arranged, fraction=0.0485, pad=0.049, ax=ax)
    cbar.set_label(label=ICE_STRINGS[chart])
    cbar.set_ticklabels(list(GROUP_NAMES[chart].values()))


def compute_metrics(true, pred, charts, metrics, num_classes):
    """
    Calculates metrics for each chart and the combined score. true and pred must be 1d arrays of equal length.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels. Must be numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must be numpy array.
    charts : List
        List of charts.
    metrics : Dict
        Stores metric calculation function and weight for each chart.

    Returns
    -------
    combined_score: float
        Combined weighted average score.
    scores: list
        List of scores for each chart.
    """
    scores = {}
    for chart in charts:
        if true[chart].ndim == 1 and pred[chart].ndim == 1:
            scores[chart] = torch.round(metrics[chart]['func'](
                true=true[chart], pred=pred[chart], num_classes=num_classes[chart]) * 100, decimals=3)

        else:
            print(f"true and pred must be 1D numpy array, got {true['SIC'].ndim} \
                and {pred['SIC'].ndim} dimensions with shape {true['SIC'].shape} and {pred.shape}, respectively")

    combined_score = compute_combined_score(scores=scores, charts=charts, metrics=metrics)

    return combined_score, scores


def r2_metric(true, pred, num_classes=None):
    """
    Calculate the r2 metric.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels. Must by numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must by numpy array.
    num_classes :
        Num of classes in the dataset, this value is not used in this function but used in f1_metric function
        which requires num_classes argument. The reason it was included here was to keep the same structure.  
    

    Returns
    -------
    r2 : float
        The calculated r2 score.

    """
    r2 = r2_score(preds=pred, target=true)

    return r2


def f1_metric(true, pred, num_classes):
    """
    Calculate the weighted f1 metric.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels.
    pred :
        ndarray, 1d contains all predicted pixels.

    Returns
    -------
    f1 : float
        The calculated f1 score.

    """
    f1 = f1_score(target=true, preds=pred, average='weighted', task='multiclass', num_classes=num_classes)

    return f1


def compute_combined_score(scores, charts, metrics):
    """
    Calculate the combined weighted score.

    Parameters
    ----------
    scores : List
        Score for each chart.
    charts : List
        List of charts.
    metrics : Dict
        Stores metric calculation function and weight for each chart.

    Returns
    -------
    : float
        The combined weighted score.

    """
    combined_metric = 0
    sum_weight = 0
    for chart in charts:
        combined_metric += scores[chart] * metrics[chart]['weight']
        sum_weight += metrics[chart]['weight']

    return torch.round(combined_metric / sum_weight, decimals=3)


# -- functions to save models -- #
def save_best_model(cfg, train_options: dict, net, optimizer, scheduler, epoch: int):
    '''
    Saves the input model in the inside the directory "/work_dirs/"experiment_name"/
    The models with be save as best_model.pth.
    The following are stored inside best_model.pth
        model_state_dict
        optimizer_state_dict
        epoch
        train_options


    Parameters
    ----------
    cfg : mmcv.Config
        The config file object of mmcv
    train_options : Dict
        The dictory which stores the train_options from quickstart
    net :
        The pytorch model
    optimizer :
        The optimizer that the model uses.
    epoch: int
        The epoch number

    '''
    print('saving model....')
    config_file_name = os.path.basename(cfg.work_dir)
    # print(config_file_name)
    torch.save(obj={'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'train_options': train_options
                    },
               f=os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth'))
    print(f"model saved successfully at {os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth')}")

    return os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth')


def load_model(net, checkpoint_path, optimizer=None, scheduler=None):
    """
    Loads a PyTorch model from a checkpoint file and returns the model, optimizer, and scheduler.
    :param model: PyTorch model to load
    :param checkpoint_path: Path to the checkpoint file
    :param optimizer: PyTorch optimizer to load (optional)
    :param scheduler: PyTorch scheduler to load (optional)
    :return: If optimizer and scheduler are provided, return the model, optimizer, and scheduler.
    """

    checkpoint = torch.load(checkpoint_path)   
    net.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']

    return epoch

