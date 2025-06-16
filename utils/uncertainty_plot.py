# %%
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load the netCDF file
path_0 = "../output/models/weight_0_c3u710at.nc"
path_1 = "../output/models/weight_1_h8drr0oi.nc"
path_50 = "../output/models/weight_50_0tcvf3v1.nc"
ds_0 = xr.open_dataset(path_0)
ds_1 = xr.open_dataset(path_1)
ds_50 = xr.open_dataset(path_50)

# List all test files
test_folder = "../data/r2t/test"
test_files = [f for f in os.listdir(test_folder) if f.endswith(".nc")]

# Create output directory for saved plots
output_dir = "../output/uncertainty_plot"

# %%


def plot_test_scene(ds_test, ds_0, ds_1, ds_50, output_dir):
    """
    Plot the test scene with predictions from different models.

    Parameters:
    -----------
    ds_test : xarray.Dataset
        Test dataset containing the input data and reference ice chart
    ds_0 : xarray.Dataset
        Model output for weight 0
    ds_1 : xarray.Dataset
        Model output for weight 1
    ds_50 : xarray.Dataset
        Model output for weight 50
    save_plots : bool, optional
        Whether to save the plots to disk (default: False)
    """

    # Extract the scene ID from the test dataset
    scene_id = os.path.basename(ds_test.encoding["source"]).split(".")[0]
    var_name = scene_id.split("_prep")[0]

    if output_dir:
        scene_dir = os.path.join(output_dir, scene_id)
        os.makedirs(scene_dir, exist_ok=True)
        print(f"Saving plots to: {scene_dir}")

    # Define constants for SIC visualization
    n_classes_SIC = 12
    LABELS_SIC = {0: 0, 1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60, 7: 70, 8: 80, 9: 90, 10: 100}
    CLABEL_SIC = "Sea Ice Concentration [%]"

    # Get SIC data from test dataset (ice chart)
    chart_data = ds_test.SIC.values
    mask = chart_data == 255  # Mask value for invalid data

    chart_data = chart_data.astype(float)
    chart_data[mask] = np.nan

    # Get predictions from different models
    data_0 = ds_0[f"{var_name}_SIC"].values
    data_0 = data_0.astype(float)
    data_0[data_0 >= 11] = 10
    data_0[mask] = np.nan

    data_1 = ds_1[f"{var_name}_SIC"].values
    data_1 = data_1.astype(float)
    data_1[data_1 >= 11] = 10
    data_1[mask] = np.nan

    data_50 = ds_50[f"{var_name}_SIC"].values
    data_50 = data_50.astype(float)
    data_50[data_50 >= 11] = 10
    data_50[mask] = np.nan

    # Get uncertainty metrics
    std_dev_0 = ds_0[f"{var_name}_SIC_std_dev"].values
    std_dev_0 = std_dev_0.astype(float)
    std_dev_0[mask] = np.nan

    std_dev_1 = ds_1[f"{var_name}_SIC_std_dev"].values
    std_dev_1 = std_dev_1.astype(float)
    std_dev_1[mask] = np.nan

    std_dev_50 = ds_50[f"{var_name}_SIC_std_dev"].values
    std_dev_50 = std_dev_50.astype(float)
    std_dev_50[mask] = np.nan

    def cbar_ice_classification(ax, n_classes, LABELS, CBAR_LABEL, cmap="viridis"):
        arranged = np.arange(0, n_classes)
        cmap = plt.get_cmap(cmap, n_classes - 1)
        norm = mpl.colors.BoundaryNorm(arranged - 0.5, cmap.N)
        arranged = arranged[:-1]  # Discount the mask class.
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=arranged, fraction=0.0485, pad=0.049, ax=ax)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label=CBAR_LABEL, fontsize=12)
        cbar.set_ticklabels(list(LABELS.values()))

    def diff_plot(ax, scene1, scene2, title1, title2):
        ax.set_title(f"Diff: {title1} - {title2}")
        diff = np.abs(np.subtract(scene1, scene2))
        im = ax.imshow(diff, cmap="hot_r")
        cbar = plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)
        cbar.set_label(label="Difference", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    def diff_correct_predictions(ax1, ax2, ax3, scene1, scene2, scene3, chart, n_classes, LABELS, CLABEL, title1, title2, title3):
        scene1_copy = np.copy(scene1)
        scene2_copy = np.copy(scene2)
        scene3_copy = np.copy(scene3)

        # Find areas where all three models predict the same value
        diff_scene_mask = (scene1_copy == scene2_copy) & (scene1_copy == scene3_copy) & (scene2_copy == scene3_copy)
        scene1_copy[diff_scene_mask] = np.nan
        scene2_copy[diff_scene_mask] = np.nan
        scene3_copy[diff_scene_mask] = np.nan

        # Find areas where predictions don't match the ground truth
        scene1_chart_mask = scene1_copy != chart
        scene2_chart_mask = scene2_copy != chart
        scene3_chart_mask = scene3_copy != chart

        # Only keep predictions that match the ground truth
        scene1_copy[scene1_chart_mask] = np.nan
        scene2_copy[scene2_chart_mask] = np.nan
        scene3_copy[scene3_chart_mask] = np.nan

        # Plot areas where only one model correctly predicts the value
        ax1.set_title(title1)
        im1 = ax1.imshow(scene1_copy, vmin=0, vmax=n_classes - 2, cmap="jet", interpolation="nearest")
        ax1.set_xticks([])
        ax1.set_yticks([])
        cbar_ice_classification(ax=ax1, n_classes=n_classes, LABELS=LABELS, CBAR_LABEL=CLABEL, cmap="jet")

        ax2.set_title(title2)
        im2 = ax2.imshow(scene2_copy, vmin=0, vmax=n_classes - 2, cmap="jet", interpolation="nearest")
        ax2.set_xticks([])
        ax2.set_yticks([])
        cbar_ice_classification(ax=ax2, n_classes=n_classes, LABELS=LABELS, CBAR_LABEL=CLABEL, cmap="jet")

        ax3.set_title(title3)
        im3 = ax3.imshow(scene3_copy, vmin=0, vmax=n_classes - 2, cmap="jet", interpolation="nearest")
        ax3.set_xticks([])
        ax3.set_yticks([])
        cbar_ice_classification(ax=ax3, n_classes=n_classes, LABELS=LABELS, CBAR_LABEL=CLABEL, cmap="jet")

    # Create plot with SAR and AMSR data
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Scene: {scene_id}", fontsize=16)

    # SAR HH
    sar = ds_test.nersc_sar_primary.values
    sar = sar.astype(float)
    sar[mask] = np.nan
    ax = axes[0, 0]
    ax.set_title("HH SAR")
    im = ax.imshow(sar, cmap="gray")
    cbar = plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)
    cbar.set_label(label="Backscatter Coeff [dB]", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    # SAR HV
    sar = ds_test.nersc_sar_secondary.values
    sar = sar.astype(float)
    sar[mask] = np.nan
    ax = axes[0, 1]
    ax.set_title("HV SAR")
    im = ax.imshow(sar, cmap="gray")
    cbar = plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)
    cbar.set_label(label="Backscatter Coeff [dB]", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    # AMSR 18.7 h
    amsr = ds_test.btemp_18_7h.values
    ax = axes[0, 2]
    ax.set_title("18.7 GHz H")
    im = ax.imshow(amsr)
    cbar = plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)
    cbar.set_label(label="TB [K]", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    # AMSR 18.7 v
    amsr = ds_test.btemp_18_7v.values
    ax = axes[1, 0]
    ax.set_title("18.7 GHz V")
    im = ax.imshow(amsr)
    cbar = plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)
    cbar.set_label(label="TB [K]", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    # AMSR 36.5 h
    amsr = ds_test.btemp_36_5h.values
    ax = axes[1, 1]
    ax.set_title("36.5 GHz H")
    im = ax.imshow(amsr)
    cbar = plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)
    cbar.set_label(label="TB [K]", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    # AMSR 36.5 v
    amsr = ds_test.btemp_36_5v.values
    ax = axes[1, 2]
    ax.set_title("36.5 GHz V")
    im = ax.imshow(amsr)
    cbar = plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)
    cbar.set_label(label="TB [K]", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    if output_dir:
        fig.savefig(os.path.join(scene_dir, "1_scene_data.png"))

    # Create plot for SIC predictions and ice chart
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f"Sea Ice Concentration - Scene: {scene_id}", fontsize=16)

    # Leave blank spaces in unused corners
    axes[0, 0].axis("off")
    axes[0, 2].axis("off")

    # Ice Chart
    ax = axes[0, 1]
    ax.set_title("Ice Chart SIC")
    im = ax.imshow(chart_data, vmin=0, vmax=n_classes_SIC - 2, cmap="jet", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar_ice_classification(ax=ax, n_classes=n_classes_SIC, LABELS=LABELS_SIC, CBAR_LABEL=CLABEL_SIC, cmap="jet")

    # Weight 0 Predictions
    ax = axes[1, 0]
    ax.set_title("Weight 0 SIC")
    im = ax.imshow(data_0, vmin=0, vmax=n_classes_SIC - 2, cmap="jet", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar_ice_classification(ax=ax, n_classes=n_classes_SIC, LABELS=LABELS_SIC, CBAR_LABEL=CLABEL_SIC, cmap="jet")

    # Weight 1 Predictions
    ax = axes[1, 1]
    ax.set_title("Weight 1 SIC")
    im = ax.imshow(data_1, vmin=0, vmax=n_classes_SIC - 2, cmap="jet", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar_ice_classification(ax=ax, n_classes=n_classes_SIC, LABELS=LABELS_SIC, CBAR_LABEL=CLABEL_SIC, cmap="jet")

    # Weight 50 Predictions
    ax = axes[1, 2]
    ax.set_title("Weight 50 SIC")
    im = ax.imshow(data_50, vmin=0, vmax=n_classes_SIC - 2, cmap="jet", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar_ice_classification(ax=ax, n_classes=n_classes_SIC, LABELS=LABELS_SIC, CBAR_LABEL=CLABEL_SIC, cmap="jet")

    # Weight 0 Std Dev
    ax = axes[2, 0]
    ax.set_title("Weight 0 Std Dev")
    im = ax.imshow(std_dev_0, cmap="inferno_r")
    cbar = plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)
    cbar.set_label(label="Std Dev [%]", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    # Weight 1 Std Dev
    ax = axes[2, 1]
    ax.set_title("Weight 1 Std Dev")
    im = ax.imshow(std_dev_1, cmap="inferno_r")
    cbar = plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)
    cbar.set_label(label="Std Dev [%]", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    # Weight 50 Std Dev
    ax = axes[2, 2]
    ax.set_title("Weight 50 Std Dev")
    im = ax.imshow(std_dev_50, cmap="inferno_r")
    cbar = plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)
    cbar.set_label(label="Std Dev [%]", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    if output_dir:
        fig.savefig(os.path.join(scene_dir, "2_sic_predictions.png"))

    # Create plot for difference maps
    fig, axes = plt.subplots(4, 3, figsize=(18, 24))
    fig.suptitle(f"Difference Maps - Scene: {scene_id}", fontsize=16)

    # Correct predictions for each scene (unique correct predictions)
    diff_correct_predictions(
        axes[0, 0],
        axes[0, 1],
        axes[0, 2],
        data_0,
        data_1,
        data_50,
        chart_data,
        n_classes_SIC,
        LABELS_SIC,
        CLABEL_SIC,
        "Only Correct For: Weight 0",
        "Only Correct For: Weight 1",
        "Only Correct For: Weight 50",
    )

    # Differences Ice Chart vs Model predictions
    diff_plot(axes[1, 0], chart_data, data_0, "Ice Chart", "Weight 0")
    diff_plot(axes[1, 1], chart_data, data_1, "Ice Chart", "Weight 1")
    diff_plot(axes[1, 2], chart_data, data_50, "Ice Chart", "Weight 50")

    # Differences between model predictions
    diff_plot(axes[2, 0], data_0, data_1, "Weight 0", "Weight 1")
    diff_plot(axes[2, 1], data_0, data_50, "Weight 0", "Weight 50")
    diff_plot(axes[2, 2], data_1, data_50, "Weight 1", "Weight 50")

    # Differences in standard deviation
    diff_plot(axes[3, 0], std_dev_0, std_dev_1, "Weight 0 Std Dev", "Weight 1 Std Dev")
    diff_plot(axes[3, 1], std_dev_0, std_dev_50, "Weight 0 Std Dev", "Weight 50 Std Dev")
    diff_plot(axes[3, 2], std_dev_1, std_dev_50, "Weight 1 Std Dev", "Weight 50 Std Dev")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    if output_dir:
        fig.savefig(os.path.join(scene_dir, "3_difference_maps.png"))

    # Create plot for ERA5 data
    # Read u10m_rotated, v10m_rotated, t2m, tcwv, tclw from the test dataset
    u10m_rotated = ds_test.u10m_rotated.values
    v10m_rotated = ds_test.v10m_rotated.values
    t2m = ds_test.t2m.values
    tcwv = ds_test.tcwv.values
    tclw = ds_test.tclw.values

    # Create a new figure for ERA5 data
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"ERA5 Data - Scene: {scene_id}", fontsize=16)

    # Plot wind vectors (u10m and v10m)
    ax = axes[0, 0]
    ax.set_title("10m Wind")
    # Skip some points to make the plot clearer
    skip = 8
    y, x = np.mgrid[: u10m_rotated.shape[0], : u10m_rotated.shape[1]]
    ax.quiver(
        x[::skip, ::skip], y[::skip, ::skip], u10m_rotated[::skip, ::skip], v10m_rotated[::skip, ::skip], scale=50, color="blue"
    )
    # Add wind speed as background
    wind_speed = np.sqrt(u10m_rotated**2 + v10m_rotated**2)
    im = ax.imshow(wind_speed, cmap="viridis", alpha=0.5)
    cbar = plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)
    cbar.set_label(label="Wind Speed [m/s]", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot t2m (2m temperature)
    ax = axes[0, 1]
    ax.set_title("2m Temperature")
    # Convert from Kelvin to Celsius
    im = ax.imshow(t2m - 273.15, cmap="RdBu_r")
    cbar = plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)
    cbar.set_label(label="Temperature [°C]", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot tcwv (total column water vapor)
    ax = axes[0, 2]
    ax.set_title("Total Column Water Vapor")
    im = ax.imshow(tcwv, cmap="Blues")
    cbar = plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)
    cbar.set_label(label="TCWV [kg/m²]", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot tclw (total column liquid water)
    ax = axes[1, 0]
    ax.set_title("Total Column Liquid Water")
    im = ax.imshow(tclw, cmap="Blues")
    cbar = plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)
    cbar.set_label(label="TCLW [kg/m²]", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    # Make the other plots empty
    axes[1, 1].axis("off")
    axes[1, 2].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_dir:
        fig.savefig(os.path.join(scene_dir, "4_era5_data.png"))


# %%
# Plot the first test scene
ds_test = xr.open_dataset(os.path.join(test_folder, test_files[0]))
scene_id = os.path.basename(ds_test.encoding["source"]).split(".")[0]
print(f"Plotting scene: {scene_id}")
plot_test_scene(ds_test, ds_0, ds_1, ds_50, output_dir=output_dir)

# %%
# Plot all test scenes
# for test_file in test_files:
#     ds_test_scene = xr.open_dataset(os.path.join(test_folder, test_file))
#     scene_id = os.path.basename(ds_test_scene.encoding["source"]).split(".")[0]
#     print(f"Plotting scene: {scene_id}")
#     plot_test_scene(ds_test_scene, ds_0, ds_1, ds_50, output_dir=output_dir)
#     plt.close("all")
