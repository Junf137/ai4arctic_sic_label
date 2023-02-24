#!/usr/bin/env python
# coding: utf-8

# # Quick start guide
# This notebook serves as an example of how to train a simple model using pytorch and the ready-to-train AI4Arctic
# challenge dataset. Initially, a dictionary, 'train_options', is set up with relevant options for both the example
# U-Net Convolutional Neural Network model and the dataloader. Note that the weights of the U-Net will be initialised
# at random and therefore not deterministic - results will vary for every training run. Two lists (dataset.json and
# testset.json) include the names of the scenes relevant to training and testing, where the former can be altered
# if desired. Training data is loaded in parallel using the build-in torch Dataset and Dataloader classes, and
# works by randomly sampling a scene and performing a random crop to extract a patch. Each batch will then be compiled
# of X number of these patches with the patch size in the 'train_options'. An obstacle is different grid resolution
# sizes, which is overcome by upsampling low resolution variables, e.g. AMSR2, ERA5, to match the SAR pixels.
# A number of batches will be prepared in parallel and stored until use, depending on the number of workers (processes)
# spawned (this can be changed in 'num_workers' in 'train_options').
# The model is trained on a fixed number of steps according to the number of batches in an epoch,
# defined by the 'epoch_len' parameter, and will run for a total number of epochs depending on the 'epochs' parameter.
# After each epoch, the model is evaluated. In this example, a random number of scenes are sampled among the training
# scenes (and removed from the list of training scenes) to act as a validation set used for the evaluation.
# The model is evaluated with the metrics, and if the current validation attempt is superior to the previous,
# then the model parameters are stored in the 'best_model' file in the directory.
#
# The models are scored on the three sea ice parameters; Sea Ice Concentration (SIC), Stage of Development (SOD) and
# the Floe size (FLOE) with the $R²$ metric for the SIC, and the weighted F1 metric for the SOD and FLOE. The 3 scores
# are combined into a single metric by taking the weighted average with SIC and SOD being weighted with 2 and the FLOE
# with 1.
#
# Finally, once you are ready to test your model on the test scenes (without reference data), the 'test_upload'
# notebook will produce model outputs with your model of choice and save the output as a netCDF file, which can be
# uploaded to the AI4EO.eu website. The model outputs will be evaluated and then you will receive a score.
#
# This quick start notebook is by no means necessary to utilize, and you are more than welcome to develop your own
# data pipeline. We do however require that the model output is stored in a netcdf file with xarray.dataarrays titled
# '{scene_name}_{chart}', i.e. 3 charts per scene / file (see how in 'test_upload'). In addition, you are more than
# welcome to create your own preprocessing scheme to prepare the raw AI4Arctic challenge dataset. However, we ask that
# the model output is in 80 m pixel spacing (original is 40 m), and that you follow the class numberings from the
# lookup tables in 'utils' - at least you will be evaluated in this way. Furthermore, we have included a function to
# convert the polygon_icechart to SIC, SOD and FLOE, you will have to incorporate it yourself.
#
# The first cell imports the necessary Python packages, initializes the 'train_options' dictionary
# the sample U-Net options, loads the dataset list and select validation scenes.

import argparse
import json
import random
import os
import os.path as osp
import shutil
from icecream import ic

import numpy as np
import torch
from mmcv import Config, mkdir_or_exist
from tqdm import tqdm  # Progress bar

import wandb
# Functions to calculate metrics and show the relevant chart colorbar.
from functions import compute_metrics, save_best_model
# Custom dataloaders for regular training and validation.
from loaders import (AI4ArcticChallengeDataset, AI4ArcticChallengeTestDataset,
                     get_variable_options)
#  get_variable_options
from unet import UNet  # Convolutional Neural Network model
# -- Built-in modules -- #
from utils import colour_str

from test_upload_function import test


def parse_args():
    parser = argparse.ArgumentParser(description='Train Default U-NET segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--wandb-project', help='Name of wandb project')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    args = parser.parse_args()

    return args

# Load training list.


def create_train_and_validation_scene_list(train_options):
    '''
    Creates the a train and validation scene list. Adds these two list to the config file train_options

    '''
    with open(train_options['path_to_env'] + 'datalists/dataset.json') as file:
        train_options['train_list'] = json.loads(file.read())

    # Convert the original scene names to the preprocessed names.
    train_options['train_list'] = [file[17:32] + '_' + file[77:80] +
                                   '_prep.nc' for file in train_options['train_list']]

    # # Select a random number of validation scenes with the same seed. Feel free to change the seed.et
    # # np.random.seed(0)
    # train_options['validate_list'] = np.random.choice(np.array(
    #     train_options['train_list']), size=train_options['num_val_scenes'], replace=False)

    # load validation list
    with open(train_options['path_to_env'] + train_options['val_path']) as file:
        train_options['validate_list'] = json.loads(file.read())
    # Convert the original scene names to the preprocessed names.
    train_options['validate_list'] = [file[17:32] + '_' + file[77:80] +
                                      '_prep.nc' for file in train_options['validate_list']]

    # from icecream import ic
    # ic(train_options['validate_list'])
    # Remove the validation scenes from the train list.
    train_options['train_list'] = [scene for scene in train_options['train_list']
                                   if scene not in train_options['validate_list']]
    print('Options initialised')


def create_dataloaders(train_options):
    '''
    Create train and validation dataloader based on the train and validation list inside train_options.

    '''
    # Custom dataset and dataloader.
    dataset = AI4ArcticChallengeDataset(
        files=train_options['train_list'], options=train_options)

    dataloader_train = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=True, num_workers=train_options['num_workers'], pin_memory=True)
    # - Setup of the validation dataset/dataloader. The same is used for model testing in 'test_upload.ipynb'.

    dataset_val = AI4ArcticChallengeTestDataset(
        options=train_options, files=train_options['validate_list'], mode='train_val')

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)

    return dataloader_train, dataloader_val


def train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer):
    '''
    Trains the model.

    '''
    best_combined_score = -np.Inf  # Best weighted model score.

    loss_functions = {chart: torch.nn.CrossEntropyLoss(ignore_index=train_options['class_fill_values'][chart])
                      for chart in train_options['charts']}

    print('Training...')
    # -- Training Loop -- #
    for epoch in tqdm(iterable=range(train_options['epochs'])):
        # gc.collect()  # Collect garbage to free memory.
        train_loss_sum = torch.tensor([0.])  # To sum the training batch losses during the epoch.
        val_loss_sum = torch.tensor([0.])  # To sum the validation batch losses during the epoch.
        net.train()  # Set network to evaluation mode.

        # Loops though batches in queue.
        for i, (batch_x, batch_y) in enumerate(tqdm(iterable=dataloader_train, total=train_options['epoch_len'], colour='red')):
            # torch.cuda.empty_cache()  # Empties the GPU cache freeing up memory.
            train_loss_batch = 0  # Reset from previous batch.

            # - Transfer to device.
            batch_x = batch_x.to(device, non_blocking=True)

            # - Mixed precision training. (Saving memory)
            with torch.cuda.amp.autocast():
                # - Forward pass.
                output = net(batch_x)
                # breakpoint()
                # - Calculate loss.
                for chart in train_options['charts']:
                    train_loss_batch += loss_functions[chart](
                        input=output[chart], target=batch_y[chart].to(device))

            # - Reset gradients from previous pass.
            optimizer.zero_grad()

            # - Backward pass.
            train_loss_batch.backward()

            # - Optimizer step
            optimizer.step()

            # - Add batch loss.
            train_loss_sum += train_loss_batch.detach().item()

            # - Average loss for displaying
            train_loss_epoch = torch.true_divide(train_loss_sum, i + 1).detach().item()
            print('\rMean training loss: ' + f'{train_loss_epoch:.3f}', end='\r')
            # del output, batch_x, batch_y # Free memory.
        # del loss_sum

        # -- Validation Loop -- #
        # For printing after the validation loop.

        # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
        outputs_flat = {chart: np.array([]) for chart in train_options['charts']}
        inf_ys_flat = {chart: np.array([]) for chart in train_options['charts']}

        net.eval()  # Set network to evaluation mode.
        print('Validating...')
        # - Loops though scenes in queue.
        for i, (inf_x, inf_y, masks, name, original_size) in enumerate(tqdm(iterable=dataloader_val, total=len(train_options['validate_list']), colour='green')):
            torch.cuda.empty_cache()
            # Reset from previous batch.
            val_loss_batch = 0
            # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
            with torch.no_grad(), torch.cuda.amp.autocast():
                inf_x = inf_x.to(device, non_blocking=True)
                output = net(inf_x)
                for chart in train_options['charts']:
                    val_loss_batch += loss_functions[chart](input=output[chart],
                                                            target=inf_y[chart].unsqueeze(0).long().to(device))

            # - Final output layer, and storing of non masked pixels.
            for chart in train_options['charts']:
                output[chart] = torch.argmax(
                    output[chart], dim=1).squeeze().cpu().numpy()
                outputs_flat[chart] = np.append(
                    outputs_flat[chart], output[chart][~masks[chart]])
                inf_ys_flat[chart] = np.append(
                    inf_ys_flat[chart], inf_y[chart][~masks[chart]].numpy())

            # - Add batch loss.
            val_loss_sum += val_loss_batch.detach().item()

            # - Average loss for displaying
            val_loss_epoch = torch.true_divide(val_loss_sum, i + 1).detach().item()

        # - Compute the relevant scores.
        print('Computing Metrics on Val dataset')
        combined_score, scores = compute_metrics(true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'],
                                                 metrics=train_options['chart_metric'])

        print("")
        print(f"Epoch {epoch} score:")

        for chart in train_options['charts']:
            print(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%")

            # Log in wandb the SIC r2_metric, SOD f1_metric and FLOE f1_metric
            wandb.log({f"{chart} {train_options['chart_metric'][chart]['func'].__name__}": scores[chart]}, step=epoch)

        print(f"Combined score: {combined_score}%")
        print(f"Train Epoch Loss: {train_loss_epoch:.3f}")
        print(f"Validation Epoch Loss: {val_loss_epoch:.3f}")

        # Log combine score and epoch loss to wandb
        wandb.log({"Combined score": combined_score,
                   "Train Epoch Loss": train_loss_epoch,
                   "Validation Epoch Loss": val_loss_epoch}, step=epoch)

        # If the scores is better than the previous epoch, then save the model and rename the image to best_validation.

        if combined_score > best_combined_score:
            best_combined_score = combined_score

            # Log the best combine score, and the metrics that make that best combine score in summary in wandb.
            wandb.run.summary["Best Combined Score"] = best_combined_score
            for chart in train_options['charts']:
                wandb.run.summary[f"{chart} {train_options['chart_metric'][chart]['func'].__name__}"] = scores[chart]
            wandb.run.summary["Train Epoch Loss"] = train_loss_epoch

            # Save the best model in work_dirs
            model_path = save_best_model(cfg, train_options, net, optimizer, epoch)
            wandb.save(model_path)
    return model_path
    # del inf_ys_flat, outputs_flat  # Free memory.


def main():
    args = parse_args()
    ic(args.config)
    cfg = Config.fromfile(args.config)
    train_options = cfg.train_options
    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(train_options)
    # cfg['experiment_name']=
    # cfg.env_dict = {}

    # set seed for everything
    seed = train_options['seed']
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True

    # To be used in test_upload.
    # get_ipython().run_line_magic('store', 'train_options')

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dir',
                                osp.splitext(osp.basename(args.config))[0])\

    ic(cfg.work_dir)
    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    # ### CUDA / GPU Setup
    # Get GPU resources.
    if torch.cuda.is_available():
        print(colour_str('GPU available!', 'green'))
        print('Total number of available devices: ',
              colour_str(torch.cuda.device_count(), 'orange'))
        device = torch.device(f"cuda:{train_options['gpu_id']}")

    else:
        print(colour_str('GPU not available.', 'red'))
        device = torch.device('cpu')
    print('GPU setup completed!')

    net = UNet(options=train_options).to(device)
    # net = UNet_sep_dec(options=train_options).to(device)

    # optimizer = torch.optim.Adam(list(net.parameters()), lr=train_options['lr'])
    optimizer = torch.optim.AdamW(list(net.parameters()), lr=train_options['lr'])

    # generate wandb run id, to be used to link the run with test_upload
    id = wandb.util.generate_id()
    # subprocess.run(['export'])

    # cfg.env_dict['WANDB_RUN_ID'] = id
    # cfg.env_dict['RESUME'] = 'allow'

    # os.environ['WANDB_RUN_ID'] = id
    # os.environ['RESUME'] = 'allow'

    # This sets up the 'device' variable containing GPU information, and the custom dataset and dataloader.
    with wandb.init(name=osp.splitext(osp.basename(args.config))[0], project=args.wandb_project,
                    entity="ai4arctic", config=train_options, id=id, resume="allow"):

        # Define the metrics and make them such that they are not added to the summary
        wandb.define_metric("Train Epoch Loss", summary="none")
        wandb.define_metric("Validation Epoch Loss", summary="none")
        wandb.define_metric("Combined score", summary="none")
        wandb.define_metric("SIC r2_metric", summary="none")
        wandb.define_metric("SOD f1_metric", summary="none")
        wandb.define_metric("FLOE f1_metric", summary="none")

        create_train_and_validation_scene_list(train_options)

        dataloader_train, dataloader_val = create_dataloaders(train_options)

        print('Data setup complete.')

        # ## Example of model training and validation loop
        # A simple model training loop following by a simple validation loop. Validation is carried out on full scenes,
        #  i.e. no cropping or stitching. If there is not enough space on the GPU, then try to do it on the cpu.
        #  This can be done by using 'net = net.cpu()'.

        checkpoint_path = train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer)
        print('Training Complete')
        print('Testing...')
        test(False, net, checkpoint_path, device, cfg)
        test(True, net, checkpoint_path, device, cfg)
        print('Testing Complete')


if __name__ == '__main__':
    main()
