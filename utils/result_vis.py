# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# 1. Load data
df = pd.read_csv("../output/wandb_export_2025-04-20T16_14_36.944-04_00.csv")

# for the rows where Name is `nomask`, set the values of edges_weight to 1
df.loc[df["Name"] == "nomask", "edges_weight"] = 1

# delete the rows where Name is `masktrain_10`
df = df[df["Name"] != "masktrain_10"]

# sort the rows by edges_weight
df = df.sort_values(by="edges_weight")


# %%

# 2. Set metrics
metrics = {
    "Best Combined Score": ("Test/Best Combined Score", "Test/Best Combined Score Center", "Test/Best Combined Score Edge"),
    "SOD F1": ("Test/SOD F1", "Test/SOD Center F1", "Test/SOD Edge F1"),
    "SIC R2": ("Test/SIC R2", "Test/SIC Center R2", "Test/SIC Edge R2"),
    "FLOE F1": ("Test/FLOE F1", "Test/FLOE Center F1", "Test/FLOE Edge F1"),
}

# Group data by edges_weight and calculate statistics
grouped_df = df.groupby("edges_weight")

# Create a new dataframe with mean and std for each metric
stats_df = pd.DataFrame()

for col in df.columns:
    if col != "edges_weight" and col != "Name":
        # Calculate mean for the column
        stats_df[f"{col}_mean"] = grouped_df[col].mean()
        # Calculate standard deviation for the column
        stats_df[f"{col}_std"] = grouped_df[col].std()

# Reset index to make edges_weight a column
stats_df = stats_df.reset_index()


# Define the boundaries for the three regions
REGION1_BOUNDARY = 1.0  # Boundary between region 1 and 2
REGION2_BOUNDARY = 10.0  # Boundary between region 2 and 3

# Define scale factors for each region
REGION1_WIDTH = 0.13  # Width allocated for region 1 [0, 1) in the transformed space
REGION2_WIDTH = 0.23  # Width allocated for region 2 [1, 10) in the transformed space
REGION3_WIDTH = 0.64  # Width allocated for region 3 [10, 100] in the transformed space

# Create a custom transformation function for the x-axis with three distinct regions
def transform_x(x):
    result = np.zeros_like(x, dtype=float)

    # Region 1: Linear scale for [0, 1)
    mask_r1 = x < REGION1_BOUNDARY
    result[mask_r1] = REGION1_WIDTH * (x[mask_r1] / REGION1_BOUNDARY)

    # Region 2: Log scale for [1, 10)
    mask_r2 = (x >= REGION1_BOUNDARY) & (x < REGION2_BOUNDARY)
    if np.any(mask_r2):
        log_scale = np.log10(x[mask_r2] / REGION1_BOUNDARY) / np.log10(REGION2_BOUNDARY / REGION1_BOUNDARY)
        result[mask_r2] = REGION1_WIDTH + REGION2_WIDTH * log_scale

    # Region 3: Log scale for [10, 100]
    mask_r3 = x >= REGION2_BOUNDARY
    if np.any(mask_r3):
        log_scale = np.log10(x[mask_r3] / REGION2_BOUNDARY) / np.log10(100 / REGION2_BOUNDARY)
        result[mask_r3] = REGION1_WIDTH + REGION2_WIDTH + REGION3_WIDTH * log_scale

    return result

# Custom formatter to display the original values on the axis
def format_x(x, pos):
    if x < REGION1_WIDTH:  # Region 1
        original = (x / REGION1_WIDTH) * REGION1_BOUNDARY
        return f"{original:.1f}"
    elif x < REGION1_WIDTH + REGION2_WIDTH:  # Region 2
        normalized = (x - REGION1_WIDTH) / REGION2_WIDTH
        original = REGION1_BOUNDARY * (10 ** (normalized * np.log10(REGION2_BOUNDARY / REGION1_BOUNDARY)))
        if original < 10:
            return f"{original:.1f}"
        else:
            return f"{int(original)}"
    else:  # Region 3
        normalized = (x - REGION1_WIDTH - REGION2_WIDTH) / REGION3_WIDTH
        original = REGION2_BOUNDARY * (10 ** (normalized * np.log10(100 / REGION2_BOUNDARY)))
        return f"{int(original)}"

# Apply transformation to the x values
stats_df["transformed_edges_weight"] = transform_x(stats_df["edges_weight"])

# %%

# 3. Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()


# Create two y-axes for each subplot (one for edge metrics, one for center/all metrics)
for i, (task_name, (col, col_center, col_edge)) in enumerate(metrics.items()):
    ax = axes[i]

    # Calculate metrics ranges for better visualization
    edge_values = stats_df[f"{col_edge}_mean"].values
    center_all_values = np.concatenate([stats_df[f"{col}_mean"].values, stats_df[f"{col_center}_mean"].values])

    edge_min, edge_max = np.nanmin(edge_values), np.nanmax(edge_values)
    center_all_min, center_all_max = np.nanmin(center_all_values), np.nanmax(center_all_values)

    # Add padding to the ranges (10% of the range)
    edge_padding = (edge_max - edge_min) * 0.1
    center_all_padding = (center_all_max - center_all_min) * 0.1

    # Create y-axis ranges
    y_range_edge = {"min": edge_min - edge_padding, "max": edge_max + edge_padding}

    y_range_center_all = {"min": center_all_min - center_all_padding, "max": center_all_max + center_all_padding}

    # Create a twin axis for the edge metrics
    ax2 = ax.twinx()

    # Plot center and all metrics on the main axis with error bars
    ax.errorbar(
        stats_df["transformed_edges_weight"],
        stats_df[f"{col}_mean"],
        yerr=stats_df[f"{col}_std"],
        marker="*",
        linestyle="-",
        label=f"{task_name} All",
        elinewidth=1,
        capsize=3,
    )

    ax.errorbar(
        stats_df["transformed_edges_weight"],
        stats_df[f"{col_center}_mean"],
        yerr=stats_df[f"{col_center}_std"],
        marker="o",
        linestyle="-",
        label=f"{task_name} Center",
        elinewidth=1,
        capsize=3,
    )

    # Plot edge metrics on the twin axis with error bars
    ax2.errorbar(
        stats_df["transformed_edges_weight"],
        stats_df[f"{col_edge}_mean"],
        yerr=stats_df[f"{col_edge}_std"],
        marker="s",
        linestyle="--",
        color="red",
        label=f"{task_name} Edge",
        elinewidth=1,
        capsize=3,
    )

    # Set labels and limits for main axis (center/all)
    ax.set_title(task_name)
    ax.set_xlabel("edges_weight")
    ax.set_ylabel(f"{task_name} (Center/All)", color="blue")
    ax.set_ylim(y_range_center_all["min"], y_range_center_all["max"])
    ax.tick_params(axis="y", labelcolor="blue")

    # Set labels and limits for twin axis (edge)
    ax2.set_ylabel(f"{task_name} (Edge)", color="red")
    ax2.set_ylim(y_range_edge["min"], y_range_edge["max"])
    ax2.tick_params(axis="y", labelcolor="red")

    # Create combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    # Set custom formatter to show original values
    ax.xaxis.set_major_formatter(FuncFormatter(format_x))

    # Add vertical lines to show the region transitions
    ax.axvline(x=REGION1_WIDTH, color="gray", linestyle="--", alpha=0.5)

    # Add custom ticks for key values
    region_ticks = [
        0,                                  # 0
        REGION1_WIDTH * 0.5,                # 0.5
        REGION1_WIDTH,                      # 1
        REGION1_WIDTH + REGION2_WIDTH * 0.5, # ~3
        REGION1_WIDTH + REGION2_WIDTH,      # 10
        REGION1_WIDTH + REGION2_WIDTH + REGION3_WIDTH * 0.25, # ~18
        REGION1_WIDTH + REGION2_WIDTH + REGION3_WIDTH * 0.5,  # ~32
        REGION1_WIDTH + REGION2_WIDTH + REGION3_WIDTH * 0.75, # ~56
        1.0                                 # 100
    ]
    ax.set_xticks(region_ticks)

    # reduce the dot size
    for line in ax.lines + ax2.lines:
        line.set_markersize(3)  # Set the size of the markers
        line.set_alpha(0.7)  # Set the transparency of the lines
        line.set_linewidth(1)

plt.tight_layout()
plt.savefig("../output/metrics_with_error_bars.png", dpi=300)
plt.show()

# 4. Print the statistical summary
print("Statistical Summary for Each Edge Weight:")
for edge_weight in sorted(stats_df["edges_weight"].unique()):
    print(f"\nEdges Weight = {edge_weight}")
    for metric in ["Test/Best Combined Score", "Test/SOD F1", "Test/SIC R2", "Test/FLOE F1"]:
        mean_val = stats_df.loc[stats_df["edges_weight"] == edge_weight, f"{metric}_mean"].values[0]
        std_val = stats_df.loc[stats_df["edges_weight"] == edge_weight, f"{metric}_std"].values[0]
        print(f"  {metric}: {mean_val:.2f} ± {std_val:.2f}")
