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

# close figure and data to avoid memory leak and consumption
data.close()


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

    sic_np = np.ma.masked_where(condition=(sic_np == sic_cfv), a=sic_np, copy=True)

    # Plot SIC
    plt.figure(figsize=(15, 10))
    plt.imshow(sic_np, cmap=cmocean.cm.ice)
    plt.colorbar()
    plt.title(title)

    # Interactively show the plot
    plt.show()


def mask_sic_label_edges(SIC, sic_cfv, ksize, threshold):
    """Mask SIC borders"""
    sic_np = SIC.numpy()

    # Apply Sobel edge detection (to highlight boundaries)
    sobel_x = cv2.Sobel(sic_np, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(sic_np, cv2.CV_64F, 0, 1, ksize=ksize)
    edges = np.hypot(sobel_x, sobel_y)  # Compute gradient magnitude
    mask = edges > threshold

    # Visualization only in debug mode
    sic_visualization(sic_np=SIC.numpy(), sic_cfv=sic_cfv, title="Original SIC")
    sic_visualization(sic_np=mask, sic_cfv=sic_cfv, title="Edge Mask")

    SIC[mask] = sic_cfv

    sic_visualization(sic_np=SIC.numpy(), sic_cfv=sic_cfv, title="Masked SIC")

    return SIC


# %% Masking sic_tensor

sic_tensor = mask_sic_label_edges(sic_tensor, sic_cfv=255, ksize=9, threshold=0)
