# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

path = "../output/result_vis"

# Load data
df = pd.read_csv(f"{path}/weight_experiments_result.csv")

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

x_mapping = {
    "before": [-1, -0.5, 0, 0.5, 1, 3, 5, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 105, 110],
    "after": [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
}


# Create a custom transformation function for the x-axis with three distinct regions
def transform_x(x):
    return [x_mapping["after"][x_mapping["before"].index(val)] for val in x]


# Apply transformation to the x values
stats_df["transformed_edges_weight"] = transform_x(stats_df["edges_weight"])

# Print the statistical summary
print("Statistical Summary for Each Edge Weight:")
for edge_weight in sorted(stats_df["edges_weight"].unique()):
    print(f"\nEdges Weight = {edge_weight}")
    for metric in ["Test/Best Combined Score", "Test/SOD F1", "Test/SIC R2", "Test/FLOE F1"]:
        mean_val = stats_df.loc[stats_df["edges_weight"] == edge_weight, f"{metric}_mean"].values[0]
        std_val = stats_df.loc[stats_df["edges_weight"] == edge_weight, f"{metric}_std"].values[0]
        print(f"  {metric}: {mean_val:.2f} ± {std_val:.2f}")

# %%

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Define categories for visualization
categories = ["All", "Center", "Edge"]
colors = [color_scheme["All"], color_scheme["Center"], color_scheme["Edge"]]

for i, (task_name, (col_all, col_center, col_edge)) in enumerate(metrics.items()):
    ax = axes[i]

    ax.plot(
        stats_df["transformed_edges_weight"],
        stats_df[f"{col_edge}_mean"],
        linestyle="--",
        color=f"{color_scheme['Edge']}",
        label=f"Edge",
        linewidth=1,
        alpha=0.5,
    )

    edge_weights = sorted(df["edges_weight"].unique())

    box_data = []
    # Prepare data for the box plot
    for edge_weight in edge_weights:
        box_data.append(
            [
                df.loc[df["edges_weight"] == edge_weight, col_all].values,
                df.loc[df["edges_weight"] == edge_weight, col_center].values,
                df.loc[df["edges_weight"] == edge_weight, col_edge].values,
            ]
        )

    # Generate positions for box plot groups
    group_width = 0.8
    box_width = group_width / 3  # 3 categories

    for j in range(len(edge_weights)):
        positions = [j - group_width / 3 + box_width / 2, j, j + group_width / 3 - box_width / 2]

        bp = ax.boxplot(
            box_data[j],
            positions=positions,
            widths=box_width * 0.8,
            patch_artist=True,
            showfliers=False,  # Don't show outliers
            medianprops=dict(color="black"),
            boxprops=dict(alpha=0.7),
        )

        # Set box colors
        for box, color in zip(bp["boxes"], colors):
            box.set(facecolor=color)

    # Set x-tick labels with original edge weight values
    ax.set_xticks(range(len(edge_weights)))
    ax.set_xticklabels([str(w) for w in x_mapping["before"][2:-2]])

    # Add labels and title
    ax.set_title(task_name)
    ax.set_xlabel("Edges Weight") if i >= 2 else None
    ax.set_ylabel("Score") if i % 2 == 0 else None

    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    # Add a legend
    if i == 0:
        handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7) for color in colors]
        ax.legend(handles, categories, loc="center right")

plt.tight_layout()
plt.savefig(f"{path}/weight_experiments_boxplot.png", dpi=300)
plt.show()
