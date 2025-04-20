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

# Scale factor for the x-axis transformation between 1 and 100
X_AXIS_SCALE_FACTOR_1 = 2
X_AXIS_SCALE_FACTOR_2 = 10

Y_RANGE_1 = {
    "min": 0.0,
    "max": 1.0,
}
Y_RANGE_2 = {
    "min": 0.0,
    "max": 1.0,
}


# Create a custom transformation function for the x-axis
def transform_x(x):
    # For values between 0 and 1, return as is (linear scale)
    # For values between 1 and 100, compress using logarithmic scale
    return np.where(x <= 1, x, 1 + np.log10(x) / np.log10(X_AXIS_SCALE_FACTOR_1))


# Create a formatter to display the original values on the axis
def format_x(x, pos):
    # For values between 0 and 1, return as is (linear scale)
    # For values between 1 and 100, return the transformed value
    return f"{x:.1f}" if x <= 1 else f"{int(X_AXIS_SCALE_FACTOR_1 ** (x - 1))}"


# Apply transformation to the x values
df["transformed_edges_weight"] = transform_x(df["edges_weight"])

# %%

# 3. Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()


# Create two y-axes for each subplot (one for edge metrics, one for center/all metrics)
for i, (task_name, (col, col_center, col_edge)) in enumerate(metrics.items()):
    ax = axes[i]

    # Calculate metrics ranges for better visualization
    edge_values = df[col_edge].values
    center_all_values = np.concatenate([df[col].values, df[col_center].values])

    edge_min, edge_max = edge_values.min(), edge_values.max()
    center_all_min, center_all_max = center_all_values.min(), center_all_values.max()

    # Add padding to the ranges (5% of the range)
    edge_padding = (edge_max - edge_min) * 0.05
    center_all_padding = (center_all_max - center_all_min) * 0.05

    # Update Y ranges
    Y_RANGE_1["min"] = edge_min - edge_padding
    Y_RANGE_1["max"] = edge_max + edge_padding
    Y_RANGE_2["min"] = center_all_min - center_all_padding
    Y_RANGE_2["max"] = center_all_max + center_all_padding

    # Create a twin axis for the edge metrics
    ax2 = ax.twinx()

    # Plot center and all metrics on the main axis
    ax.plot(df["transformed_edges_weight"], df[col], marker="*", linestyle="-", label=f"{task_name} All")
    ax.plot(df["transformed_edges_weight"], df[col_center], marker="o", linestyle="-", label=f"{task_name} Center")

    # Plot edge metrics on the twin axis
    ax2.plot(df["transformed_edges_weight"], df[col_edge], marker="s", linestyle="--", color="red", label=f"{task_name} Edge")

    # Set labels and limits for main axis (center/all)
    ax.set_title(task_name)
    ax.set_xlabel("edges_weight")
    ax.set_ylabel(f"{task_name} (Center/All)", color="blue")
    ax.set_ylim(Y_RANGE_2["min"], Y_RANGE_2["max"])
    ax.tick_params(axis="y", labelcolor="blue")

    # Set labels and limits for twin axis (edge)
    ax2.set_ylabel(f"{task_name} (Edge)", color="red")
    ax2.set_ylim(Y_RANGE_1["min"], Y_RANGE_1["max"])
    ax2.tick_params(axis="y", labelcolor="red")

    # Create combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    # Set custom formatter to show original values
    ax.xaxis.set_major_formatter(FuncFormatter(format_x))

    # Add vertical line at x=1 to show the transition between scales
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.5)

    # reduce the dot size
    for line in ax.lines + ax2.lines:
        line.set_markersize(3)  # Set the size of the markers
        line.set_alpha(0.7)  # Set the transparency of the lines
        line.set_linewidth(1)

plt.tight_layout()
plt.show()
