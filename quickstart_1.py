#!/usr/bin/env python
# coding: utf-8

# # Quick start guide
# This notebook serves as an example of how to train a simple model using pytorch and the ready-to-train AI4Arctic challenge dataset. Initially, a dictionary, 'train_options', is set up with relevant options for both the example U-Net Convolutional Neural Network model and the dataloader. Note that the weights of the U-Net will be initialised at random and therefore not deterministic - results will vary for every training run. Two lists (dataset.json and testset.json) include the names of the scenes relevant to training and testing, where the former can be altered if desired. Training data is loaded in parallel using the build-in torch Dataset and Dataloader classes, and works by randomly sampling a scene and performing a random crop to extract a patch. Each batch will then be compiled of X number of these patches with the patch size in the 'train_options'. An obstacle is different grid resolution sizes, which is overcome by upsampling low resolution variables, e.g. AMSR2, ERA5, to match the SAR pixels. A number of batches will be prepared in parallel and stored until use, depending on the number of workers (processes) spawned (this can be changed in 'num_workers' in 'train_options'). The model is trained on a fixed number of steps according to the number of batches in an epoch, defined by the 'epoch_len' parameter, and will run for a total number of epochs depending on the 'epochs' parameter. After each epoch, the model is evaluated. In this example, a random number of scenes are sampled among the training scenes (and removed from the list of training scenes) to act as a validation set used for the evaluation. The model is evaluated with the metrics, and if the current validation attempt is superior to the previous, then the model parameters are stored in the 'best_model' file in the directory.
#
# The models are scored on the three sea ice parameters; Sea Ice Concentration (SIC), Stage of Development (SOD) and the Floe size (FLOE) with the $R²$ metric for the SIC, and the weighted F1 metric for the SOD and FLOE. The 3 scores are combined into a single metric by taking the weighted average with SIC and SOD being weighted with 2 and the FLOE with 1.
#
# Finally, once you are ready to test your model on the test scenes (without reference data), the 'test_upload' notebook will produce model outputs with your model of choice and save the output as a netCDF file, which can be uploaded to the AI4EO.eu website. The model outputs will be evaluated and then you will receive a score.
#
# This quick start notebook is by no means necessary to utilize, and you are more than welcome to develop your own data pipeline. We do however require that the model output is stored in a netcdf file with xarray.dataarrays titled '{scene_name}_{chart}', i.e. 3 charts per scene / file (see how in 'test_upload'). In addition, you are more than welcome to create your own preprocessing scheme to prepare the raw AI4Arctic challenge dataset. However, we ask that the model output is in 80 m pixel spacing (original is 40 m), and that you follow the class numberings from the lookup tables in 'utils' - at least you will be evaluated in this way. Furthermore, we have included a function to convert the polygon_icechart to SIC, SOD and FLOE, you will have to incorporate it yourself.
#
# The first cell imports the necessary Python packages, initializes the 'train_options' dictionary, the sample U-Net options, loads the dataset list and select validation scenes.

# In[1]:


# -- Built-in modules -- #
from utils import CHARTS, SIC_LOOKUP, SOD_LOOKUP, FLOE_LOOKUP, SCENE_VARIABLES1, colour_str
from unet import UNet  # Convolutional Neural Network model
# Custom dataloaders for regular training and validation.
from loaders import AI4ArcticChallengeDataset, AI4ArcticChallengeTestDataset, get_variable_options
# Functions to calculate metrics and show the relevant chart colorbar.
from functions import chart_cbar, r2_metric, f1_metric, compute_metrics, save_best_model
from tqdm import tqdm  # Progress bar
import xarray as xr
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import gc
import os
import sys

# -- Environmental variables -- #
# Fill in directory for data location.
os.environ['AI4ARCTIC_DATA'] = '/home/fer96/projects/def-dclausi/share/ai4arctic/dataset/train'
# Fill in directory for environment with Ai4Arctic get-started package.
os.environ['AI4ARCTIC_ENV'] = './'


# In[2]:


# -- Third-part modules -- #

# --Proprietary modules -- #

# TODO replace with config file later
train_options = {
    'experiment_name': 'HH_HV_UNet', # To be replaced once config file is working. 
    # -- Training options -- #
    # Replace with data directory path.
    'path_to_processed_data': os.environ['AI4ARCTIC_DATA'],
    # Replace with environmment directory path.
    'path_to_env': os.environ['AI4ARCTIC_ENV'],
    'lr': 0.0001,  # Optimizer learning rate.
    'epochs': 2,  # Number of epochs before training stop.
    'epoch_len': 1,  # Number of batches for each epoch.
    # Size of patches sampled. Used for both Width and Height.
    'patch_size': 256,
    'batch_size': 16,  # Number of patches for each batch.
    # How to upscale low resolution variables to high resolution.
    'loader_upsampling': 'nearest',

    # -- Data prepraration lookups and metrics.
    # Contains the relevant variables in the scenes.
    'train_variables': SCENE_VARIABLES1,
    'charts': CHARTS,  # Charts to train on.
    'n_classes': {  # number of total classes in the reference charts, including the mask.
        'SIC': SIC_LOOKUP['n_classes'],
        'SOD': SOD_LOOKUP['n_classes'],
        'FLOE': FLOE_LOOKUP['n_classes']
    },
    # SAR pixel spacing. 80 for the ready-to-train AI4Arctic Challenge dataset.
    'pixel_spacing': 80,
    'train_fill_value': 0,  # Mask value for SAR training data.
    'class_fill_values': {  # Mask value for class/reference data.
        'SIC': SIC_LOOKUP['mask'],
        'SOD': SOD_LOOKUP['mask'],
        'FLOE': FLOE_LOOKUP['mask'],
    },

    # -- Validation options -- #
    'chart_metric': {  # Metric functions for each ice parameter and the associated weight.
        'SIC': {
            'func': r2_metric,
            'weight': 2,
        },
        'SOD': {
            'func': f1_metric,
            'weight': 2,
        },
        'FLOE': {
            'func': f1_metric,
            'weight': 1,
        },
    },
    # Number of scenes randomly sampled from train_list to use in validation.
    'num_val_scenes': 10,

    # -- GPU/cuda options -- #
    'gpu_id': 0,  # Index of GPU. In case of multiple GPUs.
    'num_workers': 4,  # Number of parallel processes to fetch data.
    'num_workers_val': 4,  # Number of parallel processes during validation.

    # -- U-Net Options -- #
    'unet_conv_filters': [16, 32, 64, 64],  # Number of filters in the U-Net.
    'conv_kernel_size': (3, 3),  # Size of convolutional kernels.
    'conv_stride_rate': (1, 1),  # Stride rate of convolutional kernels.
    'conv_dilation_rate': (1, 1),  # Dilation rate of convolutional kernels.
    'conv_padding': (1, 1),  # Number of padded pixels in convolutional layers.
    'conv_padding_style': 'zeros',  # Style of padding.
}
# Get options for variables, amsrenv grid, cropping and upsampling.
get_variable_options = get_variable_options(train_options)
# To be used in test_upload.
# get_ipython().run_line_magic('store', 'train_options')

# Load training list.
with open(train_options['path_to_env'] + 'datalists/dataset.json') as file:
    train_options['train_list'] = json.loads(file.read())
# Convert the original scene names to the preprocessed names.
train_options['train_list'] = [file[17:32] + '_' + file[77:80] +
                               '_prep.nc' for file in train_options['train_list']]
# Select a random number of validation scenes with the same seed. Feel free to change the seed.et
np.random.seed(0)
train_options['validate_list'] = np.random.choice(np.array(
    train_options['train_list']), size=train_options['num_val_scenes'], replace=False)
# Remove the validation scenes from the train list.
train_options['train_list'] = [scene for scene in train_options['train_list']
                               if scene not in train_options['validate_list']]
print('Options initialised')


# ### CUDA / GPU Setup
# This sets up the 'device' variable containing GPU information, and the custom dataset and dataloader.

# In[3]:


# Get GPU resources.
if torch.cuda.is_available():
    print(colour_str('GPU available!', 'green'))
    print('Total number of available devices: ',
          colour_str(torch.cuda.device_count(), 'orange'))
    device = torch.device(f"cuda:{train_options['gpu_id']}")

else:
    print(colour_str('GPU not available.', 'red'))
    device = torch.device('cpu')

# Custom dataset and dataloader.
dataset = AI4ArcticChallengeDataset(
    files=train_options['train_list'], options=train_options)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=None, shuffle=True, num_workers=train_options['num_workers'], pin_memory=True)
# - Setup of the validation dataset/dataloader. The same is used for model testing in 'test_upload.ipynb'.
dataset_val = AI4ArcticChallengeTestDataset(
    options=train_options, files=train_options['validate_list'])
dataloader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)

print('GPU and data setup complete.')


# ### Example of Model, optimiser and loss function setup

# In[4]:


# Setup U-Net model, adam optimizer, loss function and dataloader.
net = UNet(options=train_options).to(device)
optimizer = torch.optim.Adam(list(net.parameters()), lr=train_options['lr'])
# Selects the kernel with the best performance for the GPU and given input size.
torch.backends.cudnn.benchmark = True

# Loss functions to use for each sea ice parameter.
# The ignore_index argument discounts the masked values, ensuring that the model is not using these pixels to train on.
# It is equivalent to multiplying the loss of the relevant masked pixel with 0.
loss_functions = {chart: torch.nn.CrossEntropyLoss(ignore_index=train_options['class_fill_values'][chart])
                  for chart in train_options['charts']}
print('Model setup complete')


# ## Example of model training and validation loop
# A simple model training loop following by a simple validation loop. Validation is carried out on full scenes, i.e. no cropping or stitching. If there is not enough space on the GPU, then try to do it on the cpu. This can be done by using 'net = net.cpu()'.

# In[ ]:


best_combined_score = -np.Inf  # Best weighted model score.

# -- Training Loop -- #
for epoch in tqdm(iterable=range(train_options['epochs']), position=0):
    # gc.collect()  # Collect garbage to free memory.
    loss_sum = torch.tensor([0.])  # To sum the batch losses during the epoch.
    net.train()  # Set network to evaluation mode.

    # Loops though batches in queue.
    for i, (batch_x, batch_y) in enumerate(tqdm(iterable=dataloader, total=train_options['epoch_len'], colour='red', position=0)):
        # torch.cuda.empty_cache()  # Empties the GPU cache freeing up memory.
        loss_batch = 0  # Reset from previous batch.

        # - Transfer to device.
        batch_x = batch_x.to(device, non_blocking=True)

        # - Mixed precision training. (Saving memory)
        with torch.cuda.amp.autocast():
            # - Forward pass.
            output = net(batch_x)

            # - Calculate loss.
            for chart in train_options['charts']:
                loss_batch += loss_functions[chart](
                    input=output[chart], target=batch_y[chart].to(device))

        # - Reset gradients from previous pass.
        optimizer.zero_grad()

        # - Backward pass.
        loss_batch.backward()

        # - Optimizer step
        optimizer.step()

        # - Add batch loss.
        loss_sum += loss_batch.detach().item()

        # - Average loss for displaying
        loss_epoch = torch.true_divide(loss_sum, i + 1).detach().item()
        print('\rMean training loss: ' + f'{loss_epoch:.3f}', end='\r')
        # del output, batch_x, batch_y # Free memory.
    # del loss_sum

    # -- Validation Loop -- #
    # For printing after the validation loop.
    loss_batch = loss_batch.detach().item()

    # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
    outputs_flat = {chart: np.array([]) for chart in train_options['charts']}
    inf_ys_flat = {chart: np.array([]) for chart in train_options['charts']}

    net.eval()  # Set network to evaluation mode.
    # - Loops though scenes in queue.
    for inf_x, inf_y, masks, name in tqdm(iterable=dataloader_val, total=len(train_options['validate_list']), colour='green', position=0):
        torch.cuda.empty_cache()

        # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
        with torch.no_grad(), torch.cuda.amp.autocast():
            inf_x = inf_x.to(device, non_blocking=True)
            output = net(inf_x)

        # - Final output layer, and storing of non masked pixels.
        for chart in train_options['charts']:
            output[chart] = torch.argmax(
                output[chart], dim=1).squeeze().cpu().numpy()
            outputs_flat[chart] = np.append(
                outputs_flat[chart], output[chart][~masks[chart]])
            inf_ys_flat[chart] = np.append(
                inf_ys_flat[chart], inf_y[chart][~masks[chart]].numpy())

        # del inf_x, inf_y, masks, output  # Free memory.

    # - Compute the relevant scores.
    combined_score, scores = compute_metrics(true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'],
                                             metrics=train_options['chart_metric'])

    print("")
    print(f"Final batch loss: {loss_batch:.3f}")
    print(f"Epoch {epoch} score:")
    for chart in train_options['charts']:
        print(
            f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%")
    print(f"Combined score: {combined_score}%")

    # If the scores is better than the previous epoch, then save the model and rename the image to best_validation.
    
    if combined_score > best_combined_score:
        best_combined_score = combined_score
        # torch.save(obj={'model_state_dict': net.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #                 'epoch': epoch},
        #            f='best_model.pth')
        save_best_model(train_options,net,optimizer,epoch)
    # del inf_ys_flat, outputs_flat  # Free memory.

