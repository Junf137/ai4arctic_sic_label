
# # AutoICE - test model and prepare upload package
# This notebook tests the 'best_model', created in the quickstart notebook,
# with the tests scenes exempt of reference data.
# The model outputs are stored per scene and chart in an xarray Dataset in individual Dataarrays.
# The xarray Dataset is saved and compressed in an .nc file ready to be uploaded to the AI4EO.eu platform.
# Finally, the scene chart inference is shown.
#
# The first cell imports necessary packages:

# -- Built-in modules -- #
import json
import os
import os.path as osp

# -- Third-part modules -- #
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm
# --Proprietary modules -- #
from functions import chart_cbar, class_decider
from loaders import AI4ArcticChallengeTestDataset, get_variable_options

import wandb


def test(net: torch.nn.modules, checkpoint: str, device: str, cfg):
    """_summary_

    Args:
        net (torch.nn.modules): The model
        checkpoint (str): The checkpoint to the model
        device (str): The device to run the inference on
        cfg (Config): mmcv based Config object, Can be considered dict
    """
    train_options = cfg.train_options
    train_options = get_variable_options(train_options)
    weights = torch.load(checkpoint)['model_state_dict']
    # weights_2 = {}

    # for key, value in weights.items():
    #     weights_2[key[7:]] = value

    # Setup U-Net model, adam optimizer, loss function and dataloader.
    # net = UNet(options=train_options).to(device)
    net.load_state_dict(weights)
    print('Model successfully loaded.')
    experiment_name = osp.splitext(osp.basename(cfg.work_dir))[0]
    artifact = wandb.Artifact(experiment_name, 'dataset')
    table = wandb.Table(columns=['ID', 'Image'])

    # ### Prepare the scene list, dataset and dataloaders
    with open(train_options['path_to_env'] + 'datalists/testset.json') as file:
        train_options['test_list'] = json.loads(file.read())
        train_options['test_list'] = [file[17:32] + '_' + file[77:80] + '_prep.nc'
                                      for file in train_options['test_list']]
        # The test data is stored in a separate folder inside the training data.
        upload_package = xr.Dataset()  # To store model outputs.
        dataset = AI4ArcticChallengeTestDataset(options=train_options, files=train_options['test_list'], test=True)
        asid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)
        print('Setup ready')

    os.makedirs(osp.join(cfg.work_dir, 'inference'), exist_ok=True)
    net.eval()
    for inf_x, _, masks, scene_name, original_size in tqdm(iterable=asid_loader,
                                            total=len(train_options['test_list']), colour='green', position=0):
        scene_name = scene_name[:19]  # Removes the _prep.nc from the name.
        torch.cuda.empty_cache()
        inf_x = inf_x.to(device, non_blocking=True)

        with torch.no_grad(), torch.cuda.amp.autocast():
            output = net(inf_x)

            masks_int = masks.to(torch.uint8)
            masks_int = torch.nn.functional.interpolate(masks_int.unsqueeze(0).unsqueeze(0), size = original_size, mode = 'nearest').squeeze().squeeze()
            masks = torch.gt(masks_int, 0)
            
            # masks = torch.nn.functional.interpolate(masks.unsqueeze(0).unsqueeze(0), size = original_size, mode = 'nearest').squeeze().squeeze()
            # Upsample to match the correct size
            if train_options['down_sample_scale'] != 1:
                for chart in train_options['charts']:
                    output[chart] = torch.nn.functional.interpolate(output[chart], size = original_size, mode = 'nearest')



        for chart in train_options['charts']:
            output[chart] = class_decider(output[chart], train_options).squeeze().cpu().numpy()
            upload_package[f"{scene_name}_{chart}"] = xr.DataArray(name=f"{scene_name}_{chart}", data=output[chart].astype('uint8'),
                                                                   dims=(f"{scene_name}_{chart}_dim0", f"{scene_name}_{chart}_dim1"))

        # - Show the scene inference.
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20, 20))
        for idx, chart in enumerate(train_options['charts']):
            for j in range(0, 2):
                ax = axs[j]
                img = torch.squeeze(inf_x, dim=0).cpu().numpy()[j]
                if j == 0:
                    ax.set_title('HH')
                else:
                    ax.set_title('HV')
                ax.imshow(img, cmap='gray')
            ax = axs[idx+2]
            output[chart] = output[chart].astype(float)
            output[chart][masks] = np.nan
            ax.imshow(output[chart], vmin=0, vmax=train_options['n_classes']
                      [chart] - 2, cmap='jet', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            chart_cbar(ax=ax, n_classes=train_options['n_classes'][chart], chart=chart, cmap='jet')

        plt.suptitle(f"Scene: {scene_name}", y=0.65)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=-0)
        fig.savefig(f"{osp.join(cfg.work_dir,'inference',scene_name)}.png", format='png', dpi=128, bbox_inches="tight")
        plt.close('all')
        table.add_data(scene_name, wandb.Image(f"{osp.join(cfg.work_dir,'inference',scene_name)}.png"))

    artifact.add(table, experiment_name)
    # wandb.log_artifact(artifact)
    # - Save upload_package with zlib compression.
    print('Saving upload_package. Compressing data with zlib.')
    compression = dict(zlib=True, complevel=1)
    encoding = {var: compression for var in upload_package.data_vars}
    upload_package.to_netcdf(osp.join(cfg.work_dir, f'{experiment_name}_upload_package.nc'),
                             # f'{osp.splitext(osp.basename(cfg))[0]}
                             mode='w', format='netcdf4', engine='h5netcdf', encoding=encoding)
    print('Testing completed.')
    print("File saved succesfully at", osp.join(cfg.work_dir, f'{experiment_name}_upload_package.nc'))
    wandb.init()
    wandb.save(osp.join(cfg.work_dir, f'{experiment_name}_upload_package.nc'))
