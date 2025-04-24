# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from matplotlib.ticker import StrMethodFormatter
from scipy import interpolate

path = "../output/result_vis"

# Load data
df = pd.read_csv(f"{path}/weight_experiments_result.csv")

# change column name "weight_map.weights.SIC.inner_edges" to "edges_weight"
df.rename(columns={"weight_map.weights.SIC.inner_edges": "edges_weight"}, inplace=True)

# for the rows where Name is `nomask`, set the values of edges_weight to 1
df.loc[df["Name"] == "nomask", "edges_weight"] = 1

# delete the rows where Name is `masktrain_10`
df = df[df["Name"] != "masktrain_10"]

# sort the rows by edges_weight
df = df.sort_values(by="edges_weight")


# %%

# Set metrics
metrics = {
    "Best Combined Score": ("Test/Best Combined Score", "Test/Best Combined Score Center", "Test/Best Combined Score Edge"),
    "SOD F1": ("Test/SOD F1", "Test/SOD Center F1", "Test/SOD Edge F1"),
    "SIC R2": ("Test/SIC R2", "Test/SIC Center R2", "Test/SIC Edge R2"),
    "FLOE F1": ("Test/FLOE F1", "Test/FLOE Center F1", "Test/FLOE Edge F1"),
}

color_scheme = {
    "All": "blue",
    "Center": "green",
    "Edge": "purple",
}

# Group data by edges_weight and calculate statistics
grouped_df = df.groupby("edges_weight")

stats = pd.DataFrame()
for col in df.columns:
    if col != "edges_weight" and col != "Name":
        stats[f"{col}_mean"] = grouped_df[col].mean()
        stats[f"{col}_std"] = grouped_df[col].std()

# Reset index to make edges_weight a column
stats = stats.reset_index()

x_mapping = {
    "before": [-1, -0.5, 0, 0.5, 1, 3, 5, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 105, 110],
    "after": [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
}


# Create a custom transformation function for the x-axis with three distinct regions
def transform_x(x):
    return [x_mapping["after"][x_mapping["before"].index(val)] for val in x]


stats["tx"] = transform_x(stats["edges_weight"])

edge_weights = sorted(df["edges_weight"].unique())


def save_stats():
    # save stats to csv in the format of "mean ± std"
    stats_formatted = pd.DataFrame()
    stats_formatted["edges_weight"] = stats["edges_weight"]

    # Format all metric columns as "mean ± std"
    for metric in ["Test/Best Combined Score", "Test/SOD F1", "Test/SIC R2", "Test/FLOE F1"]:
        for suffix in ["", " Center", " Edge"]:
            column = f"{metric}{suffix}"
            mean_col = f"{column}_mean"
            std_col = f"{column}_std"

            if mean_col in stats.columns and std_col in stats.columns:
                stats_formatted[column] = stats.apply(lambda row: f"{row[mean_col]:.3f} ± {row[std_col]:.3f}", axis=1)

    # Save the formatted stats to CSV
    stats_formatted.to_csv(f"{path}/weight_experiments_stats.csv", index=False)


def smooth_curve(x, y, smoothing_factor=0.3, num_points=100):
    """
    Apply a spline smoothing to create a smoother trend line

    Args:
        x: x-coordinates
        y: y-coordinates
        smoothing_factor: Controls the amount of smoothing (0 to 1)

    Returns:
        Tuple of (x_smooth, y_smooth) for plotting
    """
    # Create more fine-grained points for smoother curves
    x_smooth = np.linspace(min(x), max(x), num_points)

    # Create the spline representation
    # s parameter controls smoothing (higher = more smoothing)
    spline = interpolate.splrep(x, y, s=smoothing_factor)

    # Evaluate the smoothed curve at the new x points
    y_smooth = interpolate.splev(x_smooth, spline)

    return x_smooth, y_smooth


save_stats()

# %%

fig = plt.figure(figsize=(12, 8))
outer_gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)

# Define categories for visualization
categories = ["All", "Center", "Edge"]
colors = [color_scheme["All"], color_scheme["Center"], color_scheme["Edge"]]

for i, (task_name, (col_all, col_center, col_edge)) in enumerate(metrics.items()):
    # locate subplot
    row, col = divmod(i, 2)
    cell = outer_gs[row, col]

    # compute y-limits for broken axis:
    edge_vals = df[col_edge].values
    ac_vals = np.concatenate([df[col_all].values, df[col_center].values])
    pad_e = (edge_vals.max() - edge_vals.min()) * 0.1
    pad_a = (ac_vals.max() - ac_vals.min()) * 0.1
    y_edge = (edge_vals.min() - pad_e, edge_vals.max() + pad_e)
    y_ac = (ac_vals.min() - pad_a, ac_vals.max() + pad_a)

    # make a 2-row gridspec inside this cell
    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=cell, height_ratios=[2, 4], hspace=0.05)
    ax_top = fig.add_subplot(inner_gs[0])
    ax_bottom = fig.add_subplot(inner_gs[1], sharex=ax_top)

    for ax in (ax_top, ax_bottom):
        x_smooth, y_smooth = smooth_curve(stats["tx"], stats[f"{col_edge}_mean"], smoothing_factor=0.3, num_points=10)
        ax.plot(
            x_smooth,
            y_smooth,
            linestyle="--",
            color=f"{color_scheme['Edge']}",
            label=f"Edge",
            linewidth=1,
            alpha=0.5,
        )
        x_smooth, y_smooth = smooth_curve(stats["tx"], stats[f"{col_all}_mean"], smoothing_factor=1, num_points=10)
        ax.plot(
            x_smooth,
            y_smooth,
            linestyle="--",
            color=f"{color_scheme['All']}",
            label=f"All",
            linewidth=1,
            alpha=0.5,
        )
        x_smooth, y_smooth = smooth_curve(stats["tx"], stats[f"{col_center}_mean"], smoothing_factor=1, num_points=10)
        ax.plot(
            x_smooth,
            y_smooth,
            linestyle="--",
            color=f"{color_scheme['Center']}",
            label=f"Center",
            linewidth=1,
            alpha=0.5,
        )
    # Generate positions for box plot groups
    group_width = 0.8
    box_width = group_width / 3  # 3 categories

    for j, w in enumerate(edge_weights):
        positions = [j - group_width / 3 + box_width / 2, j, j + group_width / 3 - box_width / 2]

        group = [
            df.loc[df["edges_weight"] == w, col_all].values,
            df.loc[df["edges_weight"] == w, col_center].values,
            df.loc[df["edges_weight"] == w, col_edge].values,
        ]

        for ax in (ax_top, ax_bottom):
            bp = ax.boxplot(
                group,
                positions=positions,
                widths=box_width * 0.8,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black"),
                boxprops=dict(alpha=0.7),
            )

            # Set box colors
            for box, color in zip(bp["boxes"], colors):
                box.set(facecolor=color)

    # Set the y-limits for the two axes
    ax_top.set_ylim(*y_ac)  # All + Center region
    ax_bottom.set_ylim(*y_edge)  # Edge region

    # Hide the spines between them
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, top=False)

    # Add “break” marks
    d = 0.015  # size of diagonal lines in axes coords
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10, linestyle="none", color="k", clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

    ax_top.set_title(task_name)
    ax_bottom.set_xlabel("Edges Weight") if i >= 2 else None
    fig.text(
        0.07,  # x position in figure coords
        0.28,  # y position in figure coords
        "Score",
        va="center",
        rotation="vertical",
    )
    fig.text(
        0.07,  # x position in figure coords
        0.72,  # y position in figure coords
        "Score",
        va="center",
        rotation="vertical",
    )

    ax_top.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax_bottom.grid(True, axis="y", linestyle="--", alpha=0.5)

    if i == 0:
        handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7) for color in colors]
        ax_bottom.legend(handles, categories, loc="lower right", fontsize=8)

    ax_top.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    ax_bottom.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))

    ax_top.tick_params(axis="y", labelsize=8)
    ax_bottom.tick_params(axis="y", labelsize=8)
    ax_bottom.set_xticks(range(len(edge_weights)))
    ax_bottom.set_xticklabels([str(w) for w in x_mapping["before"][2:-2]], rotation=45, fontsize=8)

plt.tight_layout()
plt.savefig(f"{path}/weight_experiments_boxplot_broken.png", dpi=300)
plt.show()
