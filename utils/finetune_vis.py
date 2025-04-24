import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

path = "../output/result_vis"

# Read the CSV file
df = pd.read_csv(f"{path}/finetune_result.csv")

# Original best values
original_best = {
    "Best Combined Score": [86.600, 89.125, 64.498],
    "SOD F1": [88.858, 91.036, 67.272],
    "SIC R2": [90.922, 93.801, 68.312],
    "FLOE F1": [73.440, 75.951, 51.324],
}

# Dictionary of metrics to visualize
metrics = {
    "Best Combined Score": ("Test/Best Combined Score", "Test/Best Combined Score Center", "Test/Best Combined Score Edge"),
    "SOD F1": ("Test/SOD F1", "Test/SOD Center F1", "Test/SOD Edge F1"),
    "SIC R2": ("Test/SIC R2", "Test/SIC Center R2", "Test/SIC Edge R2"),
    "FLOE F1": ("Test/FLOE F1", "Test/FLOE Center F1", "Test/FLOE Edge F1"),
}

# Group data by ksize
grouped = df.groupby("ksize")

# Create a new dataframe with mean and std for each metric
stats_df = pd.DataFrame()

for col in df.columns:
    if col != "ksize" and col != "Name" and col != "Tags":
        # Calculate mean for the column
        stats_df[f"{col}_mean"] = grouped[col].mean()
        # Calculate standard deviation for the column
        stats_df[f"{col}_std"] = grouped[col].std()

# Reset index to make ksize a column
stats_df = stats_df.reset_index()

# Sort by ksize
stats_df = stats_df.sort_values(by="ksize")

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

color_scheme = {
    "All": "blue",
    "Center": "green",
    "Edge": "purple",
}

# Plot each metric
for i, (task_name, (col, col_center, col_edge)) in enumerate(metrics.items()):
    ax = axes[i]

    ax.errorbar(
        stats_df["ksize"],
        stats_df[f"{col}_mean"],
        yerr=stats_df[f"{col}_std"],
        marker="o",
        linestyle="-",
        color=f"{color_scheme['All']}",
        label=f"All",
        capsize=3,
        linewidth=1,
        markersize=3,
        alpha=0.7,
    )

    ax.errorbar(
        stats_df["ksize"],
        stats_df[f"{col_center}_mean"],
        yerr=stats_df[f"{col}_std"],
        marker="s",
        linestyle="--",
        color=f"{color_scheme['Center']}",
        label=f"Center",
        capsize=3,
        linewidth=1,
        markersize=3,
        alpha=0.7,
    )

    ax.errorbar(
        stats_df["ksize"],
        stats_df[f"{col_edge}_mean"],
        yerr=stats_df[f"{col}_std"],
        marker="^",
        linestyle="--",
        color=f"{color_scheme['Edge']}",
        label=f"Edge",
        capsize=3,
        linewidth=1,
        markersize=3,
        alpha=0.7,
    )

    # Add horizontal lines for original best values
    ax.axhline(
        y=original_best[task_name][0],
        color=color_scheme["All"],
        linestyle=":",
        alpha=0.5,
        label=f"Original All",
        linewidth=1,
    )
    ax.axhline(
        y=original_best[task_name][1],
        color=color_scheme["Center"],
        linestyle=":",
        alpha=0.3,
        label=f"Original Center",
        linewidth=1,
    )
    ax.axhline(
        y=original_best[task_name][2],
        color=color_scheme["Edge"],
        linestyle=":",
        alpha=0.3,
        label=f"Original Edge",
        linewidth=1,
    )

    ax.set_title(task_name)
    ax.set_xlabel("Kernel Size") if (i >= 2) else None
    ax.set_ylabel("Score") if (i % 2 == 0) else None
    if i == 0:
        ax.legend(loc="center right")

    # Set x-ticks to be ksizes
    ax.set_xticks(sorted(df["ksize"].unique()))

plt.tight_layout()
plt.savefig(f"{path}/finetune_vis.png", dpi=300, bbox_inches="tight")
plt.show()

# Print statistical summary
print("Statistical Summary for Each Kernel Size:")
for ksize in sorted(stats_df["ksize"].unique()):
    print(f"\nKernel Size = {ksize}")
    for metric_name, metric_cols in metrics.items():
        for col in metric_cols:
            mean_val = stats_df.loc[stats_df["ksize"] == ksize, f"{col}_mean"].values[0]
            std_val = stats_df.loc[stats_df["ksize"] == ksize, f"{col}_std"].values[0]
            print(f"  {col}: {mean_val:.2f} ± {std_val:.2f}")
