#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pytorch Dataset class for training. Function used in train.py."""

# -- File info -- #
__author__ = "Muhammed Patel"
__contributor__ = "Xinwwei chen, Fernando Pena Cantu,Javier Turnes, Eddie Park"
__copyright__ = ["university of waterloo"]
__contact__ = ["m32patel@uwaterloo.ca", "xinweic@uwaterloo.ca"]
__version__ = "1.0.0"
__date__ = "2024-04-05"

# -- Built-in modules -- #
import os
import datetime
from dateutil import relativedelta
import re
import math
from tqdm import tqdm
import multiprocessing

# -- Third-party modules -- #
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# -- Proprietary modules -- #
from functions import rand_bbox


class AI4ArcticChallengeDataset(Dataset):
    """PyTorch dataset for loading patches from ASID V2 with optimized preprocessing."""

    def __init__(self, options, files, do_transform=False):
        self.options = options
        self.files = files
        self.do_transform = do_transform

        # Initialize data containers
        self.scenes = []

        # Precompute common parameters
        self.downsample = options["down_sample_scale"] != 1
        self.patch_size = options["patch_size"]

        # Downsample dataset
        if self.downsample:
            self._down_sample_dataset()

    def _process_single_file(self, file):
        """Process a single file and return the processed scene."""
        file_path = os.path.join(self.options["path_to_train_data"], file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training file missing: {file_path}")

        with xr.open_dataset(file_path, engine="h5netcdf") as scene:
            # Process main variables
            sar_data = torch.from_numpy(scene[self.options["full_variables"]].to_array().values)
            size = sar_data.shape[-2:]

            temp_scene = sar_data

            # Process AMSR variables
            if self.options["amsrenv_variables"]:
                amsrenv_data = torch.nn.functional.interpolate(
                    input=torch.from_numpy(scene[self.options["amsrenv_variables"]].to_array().values).unsqueeze(0),
                    size=size,
                    mode=self.options["loader_upsampling"],
                ).squeeze(0)

                temp_scene = torch.cat([temp_scene, amsrenv_data], dim=0)

            # Process auxiliary variables
            if self.options["auxiliary_variables"]:
                aux_data = self._process_auxiliary(scene, size)

                temp_scene = torch.cat([temp_scene, aux_data], dim=0)

            temp_scene = self._downsample_and_pad(temp_scene).squeeze(0)

            return temp_scene

    def _down_sample_dataset(self):
        """Initialize dataset with optimized parallel preprocessing."""
        # Determine number of processes to use (use all available cores by default)
        num_processes = min(self.options["load_proc"], len(self.files))

        # Create a multiprocessing pool
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Process files in parallel with progress bar
            processed_scenes = list(
                tqdm(pool.imap(self._process_single_file, self.files), total=len(self.files), desc="Loading scenes in parallel")
            )

        # Store processed scenes
        self.scenes = processed_scenes

    def _downsample_and_pad(self, data):
        """Handle downsampling and padding with optimized tensor ops."""

        data = F.interpolate(
            input=data.unsqueeze(0), scale_factor=1 / self.options["down_sample_scale"], mode=self.options["loader_downsampling"]
        )

        # Calculate padding needs
        h, w = data.shape[-2:]
        pad_h = max(self.patch_size - h, 0)
        pad_w = max(self.patch_size - w, 0)

        if pad_h or pad_w:
            data = F.pad(
                data,
                (0, pad_w, 0, pad_h),
                mode="constant",
                value=(255 if data.dtype == torch.uint8 else 0),  # Handle y/x differently
            )

        return data

    def _process_auxiliary(self, scene, target_shape):
        """Process auxiliary variables with batched operations."""
        aux_tensors = []

        # Create processing map for auxiliary variables
        aux_processor = {
            "aux_time": lambda: self._create_time_feature(scene, target_shape),
            "aux_lat": lambda: self._create_geo_feature(scene, "latitude", target_shape),
            "aux_long": lambda: self._create_geo_feature(scene, "longitude", target_shape),
        }

        for var in self.options["auxiliary_variables"]:
            if var in aux_processor:
                aux_tensors.append(aux_processor[var]())

        return torch.cat(aux_tensors, dim=0) if aux_tensors else None

    def _create_time_feature(self, scene, target_shape):
        """Create normalized time feature tensor."""
        norm_time = get_norm_month(scene.attrs["scene_id"])
        return torch.full((1, *target_shape), norm_time)

    def _create_geo_feature(self, scene, coord_type, target_shape):
        """Create normalized geo feature tensor."""
        coord_values = scene[f"sar_grid2d_{coord_type}"].values
        coord_values = (coord_values - self.options[coord_type]["mean"]) / self.options[coord_type]["std"]

        return F.interpolate(
            input=torch.from_numpy(coord_values).view(1, 1, *coord_values.shape),
            size=target_shape,
            mode=self.options["loader_upsampling"],
        ).squeeze(0)

    def __len__(self):
        """
        Provide number of iterations per epoch. Function required by Pytorch
        dataset.
        Returns
        -------
        Number of iterations per epoch.
        """
        return self.options["epoch_len"]

    def random_crop(self, scene):
        """
        Perform random cropping in scene.

        Parameters
        ----------
        scene :
            Xarray dataset; a scene from ASID3 ready-to-train challenge
            dataset.

        Returns
        -------
        x_patch :
            torch array with shape (len(train_variables),
            patch_height, patch_width). None if empty patch.
        y_patch :
            torch array with shape (len(charts),
            patch_height, patch_width). None if empty patch.
        """
        patch = np.zeros(
            (
                len(self.options["full_variables"])
                + len(self.options["amsrenv_variables"])
                + len(self.options["auxiliary_variables"]),
                self.options["patch_size"],
                self.options["patch_size"],
            )
        )

        # Get random index to crop from.
        row_rand = np.random.randint(low=0, high=scene["SIC"].values.shape[0] - self.options["patch_size"])
        col_rand = np.random.randint(low=0, high=scene["SIC"].values.shape[1] - self.options["patch_size"])
        # Equivalent in amsr and env variable grid.
        amsrenv_row = row_rand / self.options["amsrenv_delta"]
        # Used in determining the location of the crop in between pixels.
        amsrenv_row_dec = int(amsrenv_row - int(amsrenv_row))
        amsrenv_row_index_crop = amsrenv_row_dec * self.options["amsrenv_delta"] * amsrenv_row_dec
        amsrenv_col = col_rand / self.options["amsrenv_delta"]
        amsrenv_col_dec = int(amsrenv_col - int(amsrenv_col))
        amsrenv_col_index_crop = amsrenv_col_dec * self.options["amsrenv_delta"] * amsrenv_col_dec

        # - Discard patches with too many meaningless pixels (optional).
        if (
            np.sum(
                scene["SIC"].values[
                    row_rand : row_rand + self.options["patch_size"], col_rand : col_rand + self.options["patch_size"]
                ]
                != self.options["class_fill_values"]["SIC"]
            )
            > 1
        ):

            # Crop full resolution variables.
            patch[0 : len(self.options["full_variables"]), :, :] = (
                scene[self.options["full_variables"]]
                .isel(
                    sar_lines=range(row_rand, row_rand + self.options["patch_size"]),
                    sar_samples=range(col_rand, col_rand + self.options["patch_size"]),
                )
                .to_array()
                .values
            )
            if len(self.options["amsrenv_variables"]) > 0:
                # Crop and upsample low resolution variables.
                patch[
                    len(self.options["full_variables"]) : len(self.options["full_variables"])
                    + len(self.options["amsrenv_variables"]) :,
                    :,
                    :,
                ] = (
                    torch.nn.functional.interpolate(
                        input=torch.from_numpy(
                            scene[self.options["amsrenv_variables"]]
                            .to_array()
                            .values[
                                :,
                                int(amsrenv_row) : int(amsrenv_row + np.ceil(self.options["amsrenv_patch"])),
                                int(amsrenv_col) : int(amsrenv_col + np.ceil(self.options["amsrenv_patch"])),
                            ]
                        ).unsqueeze(0),
                        size=self.options["amsrenv_upsample_shape"],
                        mode=self.options["loader_upsampling"],
                    )
                    .squeeze(0)[
                        :,
                        int(np.around(amsrenv_row_index_crop)) : int(
                            np.around(amsrenv_row_index_crop + self.options["patch_size"])
                        ),
                        int(np.around(amsrenv_col_index_crop)) : int(
                            np.around(amsrenv_col_index_crop + self.options["patch_size"])
                        ),
                    ]
                    .numpy()
                )
            # Only add auxiliary_variables if they are called
            if len(self.options["auxiliary_variables"]) > 0:

                aux_feat_list = []

                if "aux_time" in self.options["auxiliary_variables"]:
                    # Get Scene time
                    scene_id = scene.attrs["scene_id"]
                    # Convert Scene time to number data
                    norm_time = get_norm_month(scene_id)

                    #
                    time_array = np.full((self.options["patch_size"], self.options["patch_size"]), norm_time)

                    aux_feat_list.append(time_array)

                if "aux_lat" in self.options["auxiliary_variables"]:
                    # Get Latitude
                    lat_array = scene["sar_grid2d_latitude"].values

                    lat_array = (lat_array - self.options["latitude"]["mean"]) / self.options["latitude"]["std"]

                    # Interpolate to size of original scene
                    inter_lat_array = torch.nn.functional.interpolate(
                        input=torch.from_numpy(lat_array).view((1, 1, lat_array.shape[0], lat_array.shape[1])),
                        size=scene["nersc_sar_primary"].values.shape,
                        mode=self.options["loader_upsampling"],
                    ).numpy()
                    # Crop to correct patch size
                    crop_inter_lat_array = inter_lat_array[
                        0, 0, row_rand : row_rand + self.options["patch_size"], col_rand : col_rand + self.options["patch_size"]
                    ]
                    # Append to array
                    aux_feat_list.append(crop_inter_lat_array)

                if "aux_long" in self.options["auxiliary_variables"]:
                    # Get Longitude
                    long_array = scene["sar_grid2d_longitude"].values

                    long_array = (long_array - self.options["longitude"]["mean"]) / self.options["longitude"]["std"]

                    # Interpolate to size of original scene
                    inter_long_array = torch.nn.functional.interpolate(
                        input=torch.from_numpy(long_array).view((1, 1, lat_array.shape[0], lat_array.shape[1])),
                        size=scene["nersc_sar_primary"].values.shape,
                        mode=self.options["loader_upsampling"],
                    ).numpy()
                    # Crop to correct patch size
                    crop_inter_long_array = inter_long_array[
                        0, 0, row_rand : row_rand + self.options["patch_size"], col_rand : col_rand + self.options["patch_size"]
                    ]
                    # Append to array
                    aux_feat_list.append(crop_inter_long_array)

                aux_np_array = np.stack(aux_feat_list, axis=0)

                patch[len(self.options["full_variables"]) + len(self.options["amsrenv_variables"]) :, :, :] = aux_np_array

            # Separate in to x (train variables) and y (targets) and downscale if needed

            x_patch = torch.from_numpy(patch[len(self.options["charts"]) :, :]).type(torch.float).unsqueeze(0)

            # The following code was commented because down_scale no longer happens here
            # if (self.options['down_sample_scale'] != 1):
            #     x_patch = torch.nn.functional.interpolate(
            #         x, scale_factor=1/self.options['down_sample_scale'], mode=self.options['loader_downsampling'])

            y_patch = torch.from_numpy(patch[: len(self.options["charts"]), :, :]).unsqueeze(0)

            # The following code was commented because down_scale no longer happens here
            # if (self.options['down_sample_scale'] != 1):
            #     y_patch = torch.nn.functional.interpolate(
            #         y, scale_factor=1/self.options['down_sample_scale'], mode='nearest')

        # In case patch does not contain any valid pixels - return None.
        else:
            x_patch = None
            y_patch = None

        return x_patch, y_patch

    def random_crop_downsample(self, idx):
        """
        Perform random cropping in scene.

        Parameters
        ----------
        idx : int
            Index of the scene to crop from

        Returns
        -------
        tuple (torch.Tensor, torch.Tensor)
            x_patch (input features), y_patch (target labels)
        """

        # Initialize constants
        patch_size = self.options["patch_size"]

        # Get scene dimensions
        _, height, width = self.scenes[idx].shape

        # Random crop coordinates
        assert height >= patch_size and width >= patch_size, "Scene too small for patch size"
        row_rand = 0 if height == patch_size else np.random.randint(low=0, high=height - patch_size)
        col_rand = 0 if width == patch_size else np.random.randint(low=0, high=width - patch_size)

        # Invalid patch if no valid sic label (get sic patch as idx=0)
        sic_patch = self.scenes[idx][0, row_rand : row_rand + patch_size, col_rand : col_rand + patch_size]
        if (sic_patch != self.options["class_fill_values"]["SIC"]).sum() <= 1:
            return None, None

        patch = self.scenes[idx][:, row_rand : row_rand + patch_size, col_rand : col_rand + patch_size]

        # Split into inputs and targets
        x_patch = patch[len(self.options["charts"]) :].unsqueeze(0).float()
        y_patch = patch[: len(self.options["charts"])].unsqueeze(0)

        return x_patch, y_patch

    def prep_dataset(self, x_patches, y_patches):
        """
        Convert patches from 4D numpy array to 4D torch tensor.

        Parameters
        ----------
        x_patches : ndarray
            Patches sampled from ASID3 ready-to-train challenge dataset scenes [PATCH, CHANNEL, H, W] containing only the trainable variables.
        y_patches : ndarray
            Patches sampled from ASID3 ready-to-train challenge dataset scenes [PATCH, CHANNEL, H, W] containing only the targets.

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """

        # Convert training data to tensor float.
        x = x_patches.type(torch.float)

        # Store charts in y dictionary.

        y = {}
        for idx, chart in enumerate(self.options["charts"]):
            y[chart] = y_patches[:, idx].type(torch.long)

        return x, y

    def transform(self, x_patch, y_patch):
        data_aug_options = self.options["data_augmentations"]
        if torch.rand(1) < data_aug_options["Random_h_flip"]:
            x_patch = TF.hflip(x_patch)
            y_patch = TF.hflip(y_patch)

        if torch.rand(1) < data_aug_options["Random_v_flip"]:
            x_patch = TF.vflip(x_patch)
            y_patch = TF.vflip(y_patch)

        assert data_aug_options["Random_rotation"] <= 180
        if data_aug_options["Random_rotation"] != 0 and torch.rand(1) < data_aug_options["Random_rotation_prob"]:
            random_degree = np.random.randint(-data_aug_options["Random_rotation"], data_aug_options["Random_rotation"])
        else:
            random_degree = 0

        scale_diff = data_aug_options["Random_scale"][1] - data_aug_options["Random_scale"][0]
        assert scale_diff >= 0
        if scale_diff != 0 and torch.rand(1) < data_aug_options["Random_scale_prob"]:
            random_scale = (
                np.random.rand() * (data_aug_options["Random_scale"][1] - data_aug_options["Random_scale"][0])
                + data_aug_options["Random_scale"][0]
            )
        else:
            random_scale = data_aug_options["Random_scale"][1]

        x_patch = TF.affine(x_patch, angle=random_degree, translate=(0, 0), shear=0, scale=random_scale, fill=0)
        y_patch = TF.affine(y_patch, angle=random_degree, translate=(0, 0), shear=0, scale=random_scale, fill=255)

        return x_patch, y_patch

    def __getitem__(self, idx):
        """
        Efficient batch generation.
        Returns:
            tuple: (x: 4D torch tensor, y: 4D torch tensor)
        """
        # Initialize batch containers as lists
        x_list, y_list = [], []
        max_attempts = self.options["batch_size"] * 5  # Prevent infinite loops
        attempt_count = 0

        while len(x_list) < self.options["batch_size"] and attempt_count < max_attempts:
            attempt_count += 1

            # Random scene selection using PyTorch
            scene_id = torch.randint(0, len(self.files), (1,)).item()

            try:
                if self.downsample:
                    x_patch, y_patch = self.random_crop_downsample(scene_id)
                else:
                    with xr.open_dataset(
                        os.path.join(self.options["path_to_train_data"], self.files[scene_id]), engine="h5netcdf"
                    ) as scene:
                        x_patch, y_patch = self.random_crop(scene)

                if x_patch is None:
                    continue

                if self.do_transform:
                    x_patch, y_patch = self.transform(x_patch, y_patch)

                x_list.append(x_patch.squeeze(0))
                y_list.append(y_patch.squeeze(0))

            except Exception as e:
                self._handle_scene_error(scene_id, e, attempt_count)
                continue

        # Convert lists to tensors
        x_patches = torch.stack(x_list)
        y_patches = torch.stack(y_list)

        # Apply CutMix augmentation if needed
        if self.do_transform and torch.rand(1).item() < self.options["data_augmentations"]["Cutmix_prob"]:
            x_patches, y_patches = self._apply_cutmix(x_patches, y_patches)

        return self.prep_dataset(x_patches, y_patches)

    def _handle_scene_error(self, scene_id, error, attempt_count):
        print(f"Cropping failed in {self.files[scene_id]}: {str(error)}")
        if self.downsample:
            print(
                f"Scene size: {self.scenes[scene_id][0].shape} for crop shape: \
                ({self.options['patch_size']}, {self.options['patch_size']})"
            )
        else:
            print(f"Un-downsampled scene.")
        print(f"Skipping scene (attempt {attempt_count})")

    def _apply_cutmix(self, x_patches, y_patches):
        lam = np.random.beta(self.options["data_augmentations"]["Cutmix_beta"], self.options["data_augmentations"]["Cutmix_beta"])
        rand_index = torch.randperm(x_patches.size(0))
        bbx1, bby1, bbx2, bby2 = rand_bbox(x_patches.size(), lam)

        x_patches[:, :, bbx1:bbx2, bby1:bby2] = x_patches[rand_index, :, bbx1:bbx2, bby1:bby2]
        y_patches[:, :, bbx1:bbx2, bby1:bby2] = y_patches[rand_index, :, bbx1:bbx2, bby1:bby2]

        return x_patches, y_patches


class AI4ArcticChallengeTestDataset(Dataset):
    """Pytorch dataset for loading full scenes from the ASID ready-to-train challenge dataset for inference."""

    def __init__(self, options, files, mode="test"):
        self.options = options
        self.files = files

        if mode not in ["train", "test", "test_no_gt"]:
            raise ValueError("String variable must be one of 'train', 'test', or 'test_no_gt'")
        self.mode = mode

        self.scenes = []
        self.original_sizes = []

        for file in tqdm(files, desc="Loading scenes"):
            if self.mode == "test" or self.mode == "test_no_gt":
                scene = xr.open_dataset(os.path.join(self.options["path_to_test_data"], file), engine="h5netcdf")
            else:  # train mode
                scene = xr.open_dataset(os.path.join(self.options["path_to_train_data"], file), engine="h5netcdf")

            x, y = self.prep_scene(scene)
            self.scenes.append((x, y))
            self.original_sizes.append(scene["nersc_sar_primary"].values.shape)

            scene.close()

    def __len__(self):
        """
        Provide the number of iterations. Function required by Pytorch dataset.

        Returns
        -------
        Number of scenes per validation.
        """
        return len(self.files)

    def prep_scene(self, scene):
        """
        Upsample low resolution to match charts and SAR resolution. Convert patches
        from 4D numpy array to 4D torch tensor.

        Parameters
        ----------
        scene :
            xarray dataset containing the scene data

        Returns
        -------
        x :
            4D torch tensor, ready training data.
        y :
            Dict with 3D torch tensors for each reference chart; reference inference data for x. None if test is true.
        """
        x_feat_list = []

        sar_var_x = torch.from_numpy(scene[self.options["sar_variables"]].to_array().values).unsqueeze(0)
        x_feat_list.append(sar_var_x)

        size = scene["nersc_sar_primary"].values.shape

        if len(self.options["amsrenv_variables"]) > 0:
            asmr_env__var_x = torch.nn.functional.interpolate(
                input=torch.from_numpy(scene[self.options["amsrenv_variables"]].to_array().values).unsqueeze(0),
                size=size,
                mode=self.options["loader_upsampling"],
            )
            x_feat_list.append(asmr_env__var_x)

        if len(self.options["auxiliary_variables"]) > 0:
            if "aux_time" in self.options["auxiliary_variables"]:
                scene_id = scene.attrs["scene_id"]
                norm_time = get_norm_month(scene_id)
                time_array = torch.from_numpy(np.full(scene["nersc_sar_primary"].values.shape, norm_time)).view(
                    1, 1, size[0], size[1]
                )
                x_feat_list.append(time_array)

            if "aux_lat" in self.options["auxiliary_variables"]:
                lat_array = scene["sar_grid2d_latitude"].values
                lat_array = (lat_array - self.options["latitude"]["mean"]) / self.options["latitude"]["std"]
                inter_lat_array = torch.nn.functional.interpolate(
                    input=torch.from_numpy(lat_array).view((1, 1, lat_array.shape[0], lat_array.shape[1])),
                    size=size,
                    mode=self.options["loader_upsampling"],
                )
                x_feat_list.append(inter_lat_array)

            if "aux_long" in self.options["auxiliary_variables"]:
                long_array = scene["sar_grid2d_longitude"].values
                long_array = (long_array - self.options["longitude"]["mean"]) / self.options["longitude"]["std"]
                inter_long_array = torch.nn.functional.interpolate(
                    input=torch.from_numpy(long_array).view((1, 1, lat_array.shape[0], lat_array.shape[1])),
                    size=size,
                    mode=self.options["loader_upsampling"],
                )
                x_feat_list.append(inter_long_array)

        x = torch.cat(x_feat_list, axis=1)

        if self.options["down_sample_scale"] != 1:
            x = torch.nn.functional.interpolate(
                x, scale_factor=1 / self.options["down_sample_scale"], mode=self.options["loader_downsampling"]
            )

        if self.mode != "test_no_gt":
            y_charts = torch.from_numpy(scene[self.options["charts"]].isel().to_array().values).unsqueeze(0)
            y_charts = torch.nn.functional.interpolate(
                y_charts, scale_factor=1 / self.options["down_sample_scale"], mode="nearest"
            )

            y = {}
            for idx, chart in enumerate(self.options["charts"]):
                y[chart] = y_charts[:, idx].squeeze().numpy()
        else:
            y = None

        return x.float(), y

    def __getitem__(self, idx):
        """
        Get scene from memory. Function required by Pytorch dataset.

        Returns
        -------
        x :
            4D torch tensor; ready inference data.
        y :
            Dict with 3D torch tensors for each reference chart; reference inference data for x. None if test is true.
        masks :
            Dict with 2D torch tensors; mask for each chart for loss calculation. Contain only SAR mask if test is true.
        name : str
            Name of scene.
        """
        x, y = self.scenes[idx]
        name = self.files[idx]

        if self.mode != "test_no_gt":
            cfv_masks = {}
            for chart in self.options["charts"]:
                cfv_masks[chart] = (y[chart] == self.options["class_fill_values"][chart]).squeeze()
        else:
            cfv_masks = None

        original_size = self.original_sizes[idx]

        return x, y, cfv_masks, name, original_size


def get_variable_options(train_options: dict):
    """
    Get amsr and env grid options, crop shape and upsampling shape.

    Parameters
    ----------
    train_options: dict
        Dictionary with training options.

    Returns
    -------
    train_options: dict
        Updated with amsrenv options.
        Updated with correct true patch size
    """

    train_options["amsrenv_delta"] = train_options["amsrenv_pixel_spacing"] / (
        train_options["pixel_spacing"] * train_options["down_sample_scale"]
    )

    train_options["amsrenv_patch"] = train_options["patch_size"] / train_options["amsrenv_delta"]
    train_options["amsrenv_patch_dec"] = int(train_options["amsrenv_patch"] - int(train_options["amsrenv_patch"]))
    train_options["amsrenv_upsample_shape"] = (
        int(train_options["patch_size"] + train_options["amsrenv_patch_dec"] * train_options["amsrenv_delta"]),
        int(train_options["patch_size"] + train_options["amsrenv_patch_dec"] * train_options["amsrenv_delta"]),
    )
    train_options["sar_variables"] = [
        variable for variable in train_options["train_variables"] if "sar" in variable or "map" in variable
    ]
    train_options["full_variables"] = np.hstack((train_options["charts"], train_options["sar_variables"]))
    train_options["amsrenv_variables"] = [
        variable
        for variable in train_options["train_variables"]
        if "sar" not in variable and "map" not in variable and "aux" not in variable
    ]
    train_options["auxiliary_variables"] = [variable for variable in train_options["train_variables"] if "aux" in variable]

    return train_options


def get_norm_month(file_name):

    pattern = re.compile(r"\d{8}T\d{6}")

    # Search for the first match in the string
    match = re.search(pattern, file_name)

    first_date = match.group(0)

    # parse the date string into a datetime object
    date = datetime.datetime.strptime(first_date, "%Y%m%dT%H%M%S")

    # calculate the number of days between January 1st and the given date

    delta = relativedelta.relativedelta(date, datetime.datetime(date.year, 1, 1))

    # delta = (date - datetime.datetime(date.year, 1, 1)).days

    months = delta.months
    norm_months = 2 * months / 11 - 1

    return norm_months
