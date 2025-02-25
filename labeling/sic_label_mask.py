# %%
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(""), "..")))

import xarray as xr
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


data = xr.open_dataset("../data/r2t/train/20180607T184326_dmi_prep.nc", engine="h5netcdf")

# convert to torch tensor
sic_tensor = torch.tensor(data.SIC.values, dtype=torch.float32)

# downsample SIC tensor
down_sample_scale = 10
sic_tensor = torch.nn.functional.interpolate(
    sic_tensor.unsqueeze(0).unsqueeze(0), scale_factor=1 / down_sample_scale, mode="nearest"
).squeeze()

# print torch dtype and shape
print("sic_tensor shape: ", sic_tensor.shape)

# convert into numpy array
sic_np = sic_tensor.numpy()

print("sic_np shape: ", sic_np.shape)
print("sic_np range: ", sic_np.min(), "-", sic_np.max())

# get unique values
unique_values = np.unique(sic_np)
print("unique values: ", unique_values)

sic_cfv_num = np.count_nonzero(sic_np == 255)
print(f"number of value 255 in SIC: {sic_cfv_num}, {sic_cfv_num / sic_np.size}%")


def sic_visualization(sic_np, sic_cfv, title="SIC Visualization"):
    """Visualize SIC with a color map"""

    # Get unique values in sic_np
    unique_values = np.unique(sic_np)
    print("unique values: ", unique_values)

    # Count number of sic_cfv values in sic_np
    sic_cfv_num = np.count_nonzero(sic_np == sic_cfv)
    print(f"number of value {sic_cfv} in SIC: {sic_cfv_num}, {sic_cfv_num / sic_np.size}%")

    # mask sic_np here will not change the original sic_np
    sic_np = np.ma.masked_where(sic_np == sic_cfv, sic_np)

    # Plot SIC
    plt.figure(figsize=(15, 10))
    plt.imshow(sic_np, cmap=cmocean.cm.ice)
    plt.colorbar()
    plt.title(title)

    # Interactively show the plot
    # plt.show()


def get_edges(arr_np, ksize, threshold):
    """Get edges using Sobel filter"""
    sobel_x = cv2.Sobel(arr_np, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(arr_np, cv2.CV_64F, 0, 1, ksize=ksize)
    edges = np.hypot(sobel_x, sobel_y)  # Compute gradient magnitude
    edges = edges > threshold

    return edges


def mask_sic_label_edges(SIC, sic_cfv, ksize, threshold):
    """Mask SIC borders"""
    sic_np = SIC.numpy()

    mask = get_edges(sic_np, ksize, threshold)

    # Visualization only in debug mode
    sic_visualization(sic_np=SIC.numpy(), sic_cfv=sic_cfv, title="Original SIC")
    sic_visualization(sic_np=mask, sic_cfv=sic_cfv, title="Edge Mask")

    # update SIC with mask
    SIC[mask] = sic_cfv

    sic_visualization(sic_np=SIC.numpy(), sic_cfv=sic_cfv, title="Masked SIC")


# %% Masking sic_tensor

# mask_sic_label_edges(sic_tensor, sic_cfv=255, ksize=3, threshold=0)


# %%


def create_sic_weight_map(SIC, sic_cfv, ksize, threshold, edge_weights):
    """Mask SIC borders"""
    sic_np = SIC.numpy()

    # set all non-zero values to 1 in sic_np and get ice_water
    ice_water = np.where(sic_np == 0, 0, 1).astype(sic_np.dtype)

    # set all non-sic_cfv values to 1 in sic_np and get ice_cfv
    ice_cfv = np.where(sic_np == sic_cfv, 0, 1).astype(sic_np.dtype)

    edges = get_edges(sic_np, ksize, threshold)
    ice_water_edge = get_edges(ice_water, ksize, threshold)
    ice_cfv_edge = get_edges(ice_cfv, ksize, threshold)

    # inner_edges = edges - ice_water_edge - ice_cfv_edge
    inner_edges = np.where(ice_water_edge == True, False, edges)
    inner_edges = np.where(ice_cfv_edge == True, False, inner_edges)

    # create weight map
    weight_map = np.ones_like(sic_np, dtype=sic_np.dtype)

    # first applying ice_cfv_edges, then ice_water_edges, thus the intersection will be ice_water_edges
    weight_map[inner_edges] = edge_weights["inner_edges"]
    weight_map[ice_cfv_edge] = edge_weights["ice_cfv_edges"]
    weight_map[ice_water_edge] = edge_weights["ice_water_edges"]

    # plot of intersection of ice_water_edges and ice_cfv_edges
    intersection = np.where(ice_water_edge & ice_cfv_edge, 1, 0).astype(sic_np.dtype)
    plt.imshow(intersection, cmap="gray")
    plt.title(f"Intersection: {np.sum(intersection)}")
    plt.axis("off")
    plt.show()

    # set all sic_cfv values to 0 in weight_map
    weight_map = np.where(sic_np == sic_cfv, 0, weight_map).astype(sic_np.dtype)

    plt.figure(figsize=(10, 8))

    plt.subplot(231)
    plt.imshow(edges, cmap="gray")
    plt.title("SIC Edges")
    plt.axis("off")

    plt.subplot(232)
    plt.imshow(ice_water_edge, cmap="gray")
    plt.title("Ice Water Edges")
    plt.axis("off")

    plt.subplot(233)
    plt.imshow(ice_cfv_edge, cmap="gray")
    plt.title("Ice CFV Edges")
    plt.axis("off")

    plt.subplot(234)
    plt.imshow(inner_edges, cmap="gray")
    plt.title("Inner Edges")
    plt.axis("off")

    plt.subplot(235)
    plt.imshow(weight_map, cmap="gray")
    plt.title("Weight Map")
    plt.axis("off")

    plt.subplot(236)
    plt.imshow(np.ma.masked_where(sic_np == sic_cfv, sic_np), cmap=cmocean.cm.ice)
    plt.title("SIC")
    plt.axis("off")

    plt.tight_layout()

    plt.show()

    return weight_map


# weight for inner edges, ice_cfv edges, ice_water edges
edge_weights = {
    "inner_edges": 0.5,
    "ice_cfv_edges": 0.5,
    "ice_water_edges": 1,
}

weight_map = create_sic_weight_map(sic_tensor, sic_cfv=255, ksize=5, threshold=0, edge_weights=edge_weights)


# %%
# Visualization of SAR data

HH = data.nersc_sar_primary.values
HV = data.nersc_sar_secondary.values

# downsample HH and HV
HH = torch.nn.functional.interpolate(
    torch.tensor(HH).unsqueeze(0).unsqueeze(0), scale_factor=1 / down_sample_scale, mode="nearest"
).squeeze()

HV = torch.nn.functional.interpolate(
    torch.tensor(HV).unsqueeze(0).unsqueeze(0), scale_factor=1 / down_sample_scale, mode="nearest"
).squeeze()

plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
plt.imshow(HH, cmap="gray")
plt.title("HH")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(HV, cmap="gray")
plt.title("HV")
plt.axis("off")

plt.show()
