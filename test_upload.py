# %% [markdown]
# # AutoICE - test model and prepare upload package
# This notebook tests the 'best_model', created in the quickstart notebook,
# with the tests scenes exempt of reference data.
# The model outputs are stored per scene and chart in an xarray Dataset in individual Dataarrays.
# The xarray Dataset is saved and compressed in an .nc file ready to be uploaded to the AI4EO.eu platform.
# Finally, the scene chart inference is shown.
#
# The first cell imports necessary packages:

# %%
# -- Built-in modules -- #
import pickle
import os

# -- Third-part modules -- #
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from tqdm.notebook import tqdm

# --Proprietary modules -- #
from functions import chart_cbar
from loaders import AI4ArcticChallengeTestDataset
from unet import UNet
from utils import colour_str

# TODO : integrate cfg file with test_upload
os.environ['AI4ARCTIC_DATA'] = '../dataset/train'  # Fill in directory for data location.
os.environ['AI4ARCTIC_ENV'] = './'  # Fill in directory for environment with Ai4Arctic get-started package.

# Load data (deserialize)
with open('filename.pickle', 'rb') as handle:
    train_options = pickle.load(handle)

# %% [markdown]
# ### Setup of the GPU resources

# %%
# Get GPU resources.
if torch.cuda.is_available():
    print(colour_str('GPU available!', 'green'))
    print('Total number of available devices: ', colour_str(torch.cuda.device_count(), 'orange'))
    device = torch.device(f"cuda:{train_options['gpu_id']}")

else:
    print(colour_str('GPU not available.', 'red'))
    device = torch.device('cpu')

# %% [markdown]
# ### Load the model and stored parameters

# %%
weights = torch.load('best_model')['model_state_dict']

# %%
weights_2 = {}
for key, value in weights.items():
    weights_2[key[7:]] = value

# %%
print('Loading model.')
# Setup U-Net model, adam optimizer, loss function and dataloader.
net = UNet(options=train_options).to(device)
net.load_state_dict(weights_2)
print('Model successfully loaded.')


# %% [markdown]
# ### Prepare the scene list, dataset and dataloaders

# %%
with open(train_options['path_to_env'] + 'datalists/testset.json') as file:
    train_options['test_list'] = json.loads(file.read())
train_options['test_list'] = [file[17:32] + '_' + file[77:80] + '_prep.nc' for file in train_options['test_list']]
# The test data is stored in a separate folder inside the training data.
train_options['path_to_processed_data'] = '../dataset/test'
upload_package = xr.Dataset()  # To store model outputs.
dataset = AI4ArcticChallengeTestDataset(options=train_options, files=train_options['test_list'], test=True)
asid_loader = torch.utils.data.DataLoader(
    dataset, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)
print('Setup ready')

# %%
train_options

# %%
print('Testing.')
os.makedirs('inference', exist_ok=True)
net.eval()
for inf_x, _, masks, scene_name in tqdm(iterable=asid_loader,
                                        total=len(train_options['test_list']), colour='green', position=0):
    scene_name = scene_name[:19]  # Removes the _prep.nc from the name.
    torch.cuda.empty_cache()
    inf_x = inf_x.to(device, non_blocking=True)

    with torch.no_grad(), torch.cuda.amp.autocast():
        output = net(inf_x)

    for chart in train_options['charts']:
        output[chart] = torch.argmax(output[chart], dim=1).squeeze().cpu().numpy()
        upload_package[f"{scene_name}_{chart}"] = xr.DataArray(name=f"{scene_name}_{chart}", data=output[chart].astype('uint8'),
                                                               dims=(f"{scene_name}_{chart}_dim0", f"{scene_name}_{chart}_dim1"))

    # - Show the scene inference.
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    for idx, chart in enumerate(train_options['charts']):
        ax = axs[idx]
        output[chart] = output[chart].astype(float)
        output[chart][masks] = np.nan
        ax.imshow(output[chart], vmin=0, vmax=train_options['n_classes']
                  [chart] - 2, cmap='jet', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        chart_cbar(ax=ax, n_classes=train_options['n_classes'][chart], chart=chart, cmap='jet')

    plt.suptitle(f"Scene: {scene_name}", y=0.65)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=-0)
    fig.savefig(f"inference/{scene_name}.png", format='png', dpi=128, bbox_inches="tight")
    plt.close('all')


# - Save upload_package with zlib compression.
print('Saving upload_package. Compressing data with zlib.')
compression = dict(zlib=True, complevel=1)
encoding = {var: compression for var in upload_package.data_vars}
upload_package.to_netcdf('upload_package.nc', mode='w', format='netcdf4', engine='h5netcdf', encoding=encoding)
print('Testing completed.')
