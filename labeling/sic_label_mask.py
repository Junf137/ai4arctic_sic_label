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
hh_tensor = torch.tensor(data.nersc_sar_primary.values, dtype=torch.float32)
hv_tensor = torch.tensor(data.nersc_sar_secondary.values, dtype=torch.float32)


# downsample SIC tensor
down_sample_scale = 10
sic_tensor = torch.nn.functional.interpolate(
    sic_tensor.unsqueeze(0).unsqueeze(0), scale_factor=1 / down_sample_scale, mode="nearest"
).squeeze()
hh_tensor = torch.nn.functional.interpolate(
    hh_tensor.unsqueeze(0).unsqueeze(0), scale_factor=1 / down_sample_scale, mode="nearest"
).squeeze()
hv_tensor = torch.nn.functional.interpolate(
    hv_tensor.unsqueeze(0).unsqueeze(0), scale_factor=1 / down_sample_scale, mode="nearest"
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

# %%


def get_edges(arr_np, ksize, threshold):
    """Get edges using Sobel filter"""
    sobel_x = cv2.Sobel(arr_np, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(arr_np, cv2.CV_64F, 0, 1, ksize=ksize)
    edges = np.hypot(sobel_x, sobel_y)  # Compute gradient magnitude
    edges = edges > threshold

    return edges


def create_sic_weight_map(SIC, sic_cfv, ksize, threshold, edge_weights):
    """Mask SIC borders"""
    sic_np = SIC.numpy()

    # set all non-zero values to 1 in sic_np and get ice_water
    ice_water = np.where(sic_np == 0, 1, 0).astype(sic_np.dtype)

    # set all non-sic_cfv values to 1 in sic_np and get ice_cfv
    ice_cfv = np.where(sic_np == sic_cfv, 1, 0).astype(sic_np.dtype)

    edges = get_edges(sic_np, ksize, threshold)
    ice_water_edge = get_edges(ice_water, ksize, threshold)
    ice_cfv_edge = get_edges(ice_cfv, ksize, threshold)

    # inner_edges = edges - ice_water_edge - ice_cfv_edge
    inner_edges = np.where(ice_water_edge == True, False, edges)
    inner_edges = np.where(ice_cfv_edge == True, False, inner_edges)

    # create weight map
    weight_map = np.ones_like(sic_np, dtype=sic_np.dtype) * edge_weights["center"]

    # first applying ice_water_edges, then ice_cfv_edges, thus the intersection will be ice_cfv_edges
    weight_map[inner_edges] = edge_weights["inner_edges"]
    weight_map[ice_water_edge] = edge_weights["ice_water_edges"]
    weight_map[ice_cfv_edge] = edge_weights["ice_cfv_edges"]

    # set all sic_cfv values to 0 in weight_map
    weight_map = np.where(sic_np == sic_cfv, edge_weights["invalid"], weight_map).astype(sic_np.dtype)

    return edges, ice_water_edge, ice_cfv_edge, inner_edges, weight_map


def plot_weight_map(
    edges: np.ndarray,
    ice_water_edge: np.ndarray,
    ice_cfv_edge: np.ndarray,
    inner_edges: np.ndarray,
    sic_np: np.ndarray,
    sic_cfv: int,
    weight_map: np.ndarray,
    hh_np: np.ndarray,
    hv_np: np.ndarray,
):
    """Plot weight map"""
    plt.figure(figsize=(12, 7))

    plt.subplot(241)
    plt.imshow(edges, cmap="gray")
    plt.title("SIC Edges")
    plt.axis("off")

    plt.subplot(242)
    plt.imshow(ice_water_edge, cmap="gray")
    plt.title("Ice-Water Edges")
    plt.axis("off")

    plt.subplot(243)
    plt.imshow(ice_cfv_edge, cmap="gray")
    plt.title("Ice-CFV Edges")
    plt.axis("off")

    plt.subplot(244)
    plt.imshow(inner_edges, cmap="gray")
    plt.title("Inner Edges")
    plt.axis("off")

    plt.subplot(245)
    plt.imshow(np.ma.masked_where(sic_np == sic_cfv, sic_np), cmap=cmocean.cm.ice)
    plt.title("SIC")
    plt.axis("off")

    plt.subplot(246)
    plt.imshow(weight_map, cmap="gray")
    plt.title("Weight Map")
    plt.axis("off")

    plt.subplot(247)
    plt.imshow(hh_np, cmap="gray")
    plt.title("HH")
    plt.axis("off")

    plt.subplot(248)
    plt.imshow(hv_np, cmap="gray")
    plt.title("HV")
    plt.axis("off")

    plt.tight_layout()

    plt.show()


# weight for inner edges, ice_cfv edges, ice_water edges
edge_weights = {
    "invalid": 0,
    "inner_edges": 0.5,
    "ice_cfv_edges": 0.5,
    "ice_water_edges": 1,
    "center": 1,
}

edges, ice_water_edge, ice_cfv_edge, inner_edges, weight_map = create_sic_weight_map(
    SIC=sic_tensor, sic_cfv=255, ksize=5, threshold=0, edge_weights=edge_weights
)

plot_weight_map(
    edges=edges,
    ice_water_edge=ice_water_edge,
    ice_cfv_edge=ice_cfv_edge,
    inner_edges=inner_edges,
    sic_np=sic_tensor.numpy(),
    sic_cfv=255,
    weight_map=weight_map,
    hh_np=hh_tensor.numpy(),
    hv_np=hv_tensor.numpy(),
)
