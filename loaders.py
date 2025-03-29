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


def process_auxiliary_features(scene, options, target_shape):
    def create_time_feature():
        norm_time = get_norm_month(scene.attrs["scene_id"])
        return torch.full((1, *target_shape), norm_time, dtype=torch.float32)

    def create_geo_feature(coord_type):
        coord_values = scene[f"sar_grid2d_{coord_type}"].values
        coord_values = (coord_values - options[coord_type]["mean"]) / options[coord_type]["std"]
        coord_values = torch.from_numpy(coord_values).to(torch.float32)
        return F.interpolate(
            input=coord_values.unsqueeze(0).unsqueeze(0),
            size=target_shape,
            mode=options["loader_upsampling"],
        ).squeeze(0)

    aux_processor = {
        "aux_time": create_time_feature,
        "aux_lat": lambda: create_geo_feature("latitude"),
        "aux_long": lambda: create_geo_feature("longitude"),
    }

    aux_tensors = []
    for var in options["auxiliary_variables"]:
        if var in aux_processor:
            aux_tensors.append(aux_processor[var]())

    return torch.cat(aux_tensors, dim=0) if aux_tensors else None


def downsample_and_pad(data, down_sample_scale, loader_downsampling, patch_size):
    """Handle downsampling and padding with optimized tensor ops."""

    data = F.interpolate(input=data.unsqueeze(0), scale_factor=1 / down_sample_scale, mode=loader_downsampling)

    # Calculate padding needs
    h, w = data.shape[-2:]
    pad_h = max(patch_size - h, 0)
    pad_w = max(patch_size - w, 0)

    if pad_h or pad_w:
        data = F.pad(
            data,
            (0, pad_w, 0, pad_h),
            mode="constant",
            value=(255 if data.dtype == torch.uint8 else 0),  # Handle y/x differently
        )

    return data


def process_single_scene(options, scene):
    # Process main variables
    sar_data = torch.from_numpy(scene[options["full_variables"]].to_array().values).to(torch.float32)
    size = sar_data.shape[-2:]

    temp_scene = sar_data

    # Process AMSR variables
    if options["amsrenv_variables"]:
        amsrenv_data = torch.from_numpy(scene[options["amsrenv_variables"]].to_array().values).to(torch.float32)
        amsrenv_data = F.interpolate(
            input=amsrenv_data.unsqueeze(0),
            size=size,
            mode=options["loader_upsampling"],
        ).squeeze(0)

        temp_scene = torch.cat([temp_scene, amsrenv_data], dim=0)

    # Process auxiliary variables
    if options["auxiliary_variables"]:
        aux_data = process_auxiliary_features(scene, options, size)
        if aux_data is not None:
            temp_scene = torch.cat([temp_scene, aux_data], dim=0)

    temp_scene = downsample_and_pad(
        temp_scene, options["down_sample_scale"], options["loader_downsampling"], options["patch_size"]
    ).squeeze(0)

    return temp_scene


class AI4ArcticChallengeDataset(Dataset):
    """PyTorch dataset for loading patches from ASID V2 with optimized preprocessing."""

    def __init__(self, options, files, do_transform=False):
        self.options = options
        self.files = files
        self.do_transform = do_transform

        # Initialize data containers
        self.scenes = []

        # Precompute common parameters
        self.patch_size = options["patch_size"]

        if self.options["down_sample_scale"] == 1:
            raise ValueError("Downsample has to be enabled")

        # Downsample dataset
        self._load_scenes()

    def _load_scenes(self):
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

    def _process_single_file(self, file):
        """Process a single file and return the processed scene."""
        file_path = os.path.join(self.options["path_to_train_data"], file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training file missing: {file_path}")

        with xr.open_dataset(file_path, engine="h5netcdf") as scene:
            # Process the scene
            processed_scene = process_single_scene(self.options, scene)

            scene.close()

        return processed_scene

    def __len__(self):
        """
        Provide number of iterations per epoch. Function required by Pytorch
        dataset.
        Returns
        -------
        Number of iterations per epoch.
        """
        return self.options["epoch_len"]

    def random_crop_downsample(self, idx):
        """Perform random cropping in scene.

        Returns:
            tuple: (x_patch, y_patch) where:
                - x_patch: Input features tensor
                - y_patch: Target charts tensor
        """

        patch_size = self.options["patch_size"]
        scene = self.scenes[idx]
        _, height, width = scene.shape

        # Validation check
        if height < patch_size or width < patch_size:
            return None, None

        # Random crop coordinates
        row_rand = 0 if height == patch_size else torch.randint(0, height - patch_size, (1,)).item()
        col_rand = 0 if width == patch_size else torch.randint(0, width - patch_size, (1,)).item()

        patch = scene[:, row_rand : row_rand + patch_size, col_rand : col_rand + patch_size].clone()

        # Invalid patch if no valid sic label (check SIC channel which is first channel)
        sic_patch = patch[0]
        if (sic_patch != self.options["class_fill_values"]["SIC"]).sum() <= 1:
            return None, None

        # Split into inputs and targets
        # x_patch includes all input features after target charts
        x_patch = patch[len(self.options["charts"]) :].unsqueeze(0)
        # y_patch includes target charts
        y_patch = patch[: len(self.options["charts"])].unsqueeze(0)

        return x_patch, y_patch

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
                x_patch, y_patch = self.random_crop_downsample(scene_id)

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

        # prepare dataset for training
        x = x_patches

        y = {}
        for idx, chart in enumerate(self.options["charts"]):
            y[chart] = y_patches[:, idx]

        return x, y

    def _handle_scene_error(self, scene_id, error, attempt_count):
        print(f"Cropping failed in {self.files[scene_id]}: {str(error)}")
        print(
            f"Scene size: {self.scenes[scene_id][0].shape} for crop shape: \
            ({self.options['patch_size']}, {self.options['patch_size']})"
        )
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

        if mode not in ["train", "test"]:
            raise ValueError("String variable must be one of 'train', 'test'")
        self.mode = mode

        self.scenes = []
        self.original_sizes = []

        self._load_scenes()

    def _load_scenes(self):
        """Initialize dataset with optimized parallel preprocessing."""
        # Determine number of processes to use (use all available cores by default)
        num_processes = min(self.options["load_proc"], len(self.files))

        # Create a multiprocessing pool
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Process files in parallel with progress bar
            processed_data = list(
                tqdm(pool.imap(self._process_single_file, self.files), total=len(self.files), desc="Loading scenes in parallel")
            )

        # Unpack the tuples and store in respective lists
        self.scenes = []
        self.original_sizes = []
        for processed_scene, original_size in processed_data:
            self.scenes.append(processed_scene)
            self.original_sizes.append(original_size)

    def _process_single_file(self, file):
        """Process a single file and return the processed scene."""
        if self.mode == "test":
            file_path = os.path.join(self.options["path_to_test_data"], file)
        else:  # train mode
            file_path = os.path.join(self.options["path_to_train_data"], file)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training file missing: {file_path}")

        with xr.open_dataset(file_path, engine="h5netcdf") as scene:
            # Process the scene
            processed_scene = process_single_scene(self.options, scene)

            original_size = scene["nersc_sar_primary"].values.shape

            scene.close()

        return processed_scene, original_size

    def __len__(self):
        """
        Provide the number of iterations. Function required by Pytorch dataset.

        Returns
        -------
        Number of scenes per validation.
        """
        return len(self.files)

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

        scene = self.scenes[idx].clone()

        x = scene[len(self.options["charts"]) :].unsqueeze(0)

        y_charts = scene[: len(self.options["charts"])]
        y = {}
        for i, chart in enumerate(self.options["charts"]):
            y[chart] = y_charts[i]

        cfv_masks = {}
        for chart in self.options["charts"]:
            cfv_masks[chart] = (y[chart] == self.options["class_fill_values"][chart]).squeeze()

        name = self.files[idx]
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
