import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import json

sys.path.append("..")
from functions import get_edges

# Define the folder path containing NC files
# folder_path = "./data/r2t/train/"
folder_path = "../data/r2t/test/"

output_folder = "../output/"

# Target data list
# datalist_path = "./datalists/valset2.json"
datalist_path = "../datalists/testset.json"

ksize = 5
threshold = 0

# Define class ranges
chart_info = {
    "SIC": {
        "classes": range(11),  # 0-10
        "invalid_value": 255,
    },
    "SOD": {
        "classes": range(6),  # 0-5
        "invalid_value": 255,
    },
    "FLOE": {
        "classes": range(7),  # 0-6
        "invalid_value": 255,
    },
}


def get_chart_info(data: np.array, chart: str, chart_info: dict):

    edges = get_edges(arr_np=data, ksize=ksize, threshold=threshold)
    edge_pixel = np.sum(edges)

    invalid = np.sum(data == chart_info[chart]["invalid_value"])
    valid = data.size - invalid
    class_counts = {cls: np.sum(data == cls) for cls in chart_info[chart]["classes"]}
    class_ratios = {cls: count / valid if valid > 0 else 0 for cls, count in class_counts.items()}

    return invalid, valid, class_counts, class_ratios, edge_pixel


# Variables to track
invalid_value = 255


def process_filename(file):
    return f"{file[17:32]}_{file[77:80]}_prep.nc"


# Function to analyze a single NC file
def analyze_nc_file(file_path):
    try:
        # Open the dataset
        data = xr.open_dataset(file_path, engine="h5netcdf")

        # Get the variables as np arrays
        sic = data.SIC.values
        sod = data.SOD.values
        floe = data.FLOE.values

        # Get dimensions
        height, width = sic.shape
        total_pixels = height * width

        # Calculate statistics for SIC
        sic_invalid, sic_valid, sic_class_counts, sic_class_ratios, sic_edge_pixel = get_chart_info(sic, "SIC", chart_info)

        # Calculate statistics for SOD
        sod_invalid, sod_valid, sod_class_counts, sod_class_ratios, sod_edge_pixel = get_chart_info(sod, "SOD", chart_info)

        # Calculate statistics for FLOE
        floe_invalid, floe_valid, floe_class_counts, floe_class_ratios, floe_edge_pixel = get_chart_info(floe, "FLOE", chart_info)

        # Create a dictionary of statistics
        stats = {
            "file_name": file_path.name,
            "dimensions": f"{height}x{width}",
            "total_pixels": total_pixels,
            "sic_invalid_pixels": sic_invalid,
            "sic_valid_pixels": sic_valid,
            "sic_valid_ratio": sic_valid / total_pixels,
            "sic_edge_ratio": sic_edge_pixel / total_pixels,
            "sod_invalid_pixels": sod_invalid,
            "sod_valid_pixels": sod_valid,
            "sod_valid_ratio": sod_valid / total_pixels,
            "sod_edge_ratio": sod_edge_pixel / total_pixels,
            "floe_invalid_pixels": floe_invalid,
            "floe_valid_pixels": floe_valid,
            "floe_valid_ratio": floe_valid / total_pixels,
            "floe_edge_ratio": floe_edge_pixel / total_pixels,
        }

        # Add class counts and ratios to stats
        for cls in chart_info["SIC"]["classes"]:
            stats[f"sic_class_{cls}_count"] = sic_class_counts[cls]
            stats[f"sic_class_{cls}_ratio"] = sic_class_ratios[cls]

        for cls in chart_info["SOD"]["classes"]:
            stats[f"sod_class_{cls}_count"] = sod_class_counts[cls]
            stats[f"sod_class_{cls}_ratio"] = sod_class_ratios[cls]

        for cls in chart_info["FLOE"]["classes"]:
            stats[f"floe_class_{cls}_count"] = floe_class_counts[cls]
            stats[f"floe_class_{cls}_ratio"] = floe_class_ratios[cls]

        return stats

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_folder(folder_path: str, datalist_pah: str):
    folder = Path(folder_path)

    with open(datalist_pah, "r") as f:
        datalist = json.load(f)

    nc_files = [folder / process_filename(file_name) for file_name in datalist]
    print(f"Found {len(nc_files)} NC files in {folder}")

    all_stats = []
    for file_path in nc_files:
        print(f"Processing {file_path.name}...")
        stats = analyze_nc_file(file_path)
        if stats:
            all_stats.append(stats)

    return all_stats


# Calculate average statistics across all scenes
def calculate_average_stats(all_stats):
    if not all_stats:
        return None

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_stats)

    # Calculate averages for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    avg_stats = df[numeric_cols].mean().to_dict()

    # Count total files
    avg_stats["total_files"] = len(all_stats)

    # Add average dimensions (can't directly average string dimensions)
    heights = []
    widths = []
    for dim_str in df["dimensions"]:
        h, w = map(int, dim_str.split("x"))
        heights.append(h)
        widths.append(w)

    avg_stats["avg_height"] = np.mean(heights)
    avg_stats["avg_width"] = np.mean(widths)

    return avg_stats


def main():
    print(f"Analyzing NetCDF files in {folder_path}")

    # Process all NC files
    all_stats = process_folder(folder_path, datalist_path)

    # Calculate average statistics
    if all_stats:
        avg_stats = calculate_average_stats(all_stats)

        # Create detailed report
        print("\n" + "=" * 50)
        print(f"ANALYSIS SUMMARY FOR {len(all_stats)} NC FILES")
        print("=" * 50)

        print(f"\nAverage Dimensions: {avg_stats['avg_height']:.1f} x {avg_stats['avg_width']:.1f}")
        print(f"Average Total Pixels: {avg_stats['total_pixels']:.1f}")

        # Summary for SIC
        print("\nSEA ICE CONCENTRATION (SIC):")
        print(f"  Valid Pixels: {avg_stats['sic_valid_pixels']:.1f} ({avg_stats['sic_valid_ratio']*100:.2f}%)")
        print(f"  Invalid Pixels: {avg_stats['sic_invalid_pixels']:.1f} ({(1-avg_stats['sic_valid_ratio'])*100:.2f}%)")
        print(f"  Edge Ratio: {avg_stats['sic_edge_ratio']*100:.2f}%")
        print("  Class Distribution (Valid Pixels):")
        for cls in chart_info["SIC"]["classes"]:
            cls_count = avg_stats[f"sic_class_{cls}_count"]
            cls_ratio = avg_stats[f"sic_class_{cls}_ratio"]
            print(f"    Class {cls}: {cls_count:.1f} pixels ({cls_ratio*100:.2f}%)")

        # Summary for SOD
        print("\nSTAGE OF DEVELOPMENT (SOD):")
        print(f"  Valid Pixels: {avg_stats['sod_valid_pixels']:.1f} ({avg_stats['sod_valid_ratio']*100:.2f}%)")
        print(f"  Invalid Pixels: {avg_stats['sod_invalid_pixels']:.1f} ({(1-avg_stats['sod_valid_ratio'])*100:.2f}%)")
        print(f"  Edge Ratio: {avg_stats['sod_edge_ratio']*100:.2f}%")
        print("  Class Distribution (Valid Pixels):")
        for cls in chart_info["SOD"]["classes"]:
            cls_count = avg_stats[f"sod_class_{cls}_count"]
            cls_ratio = avg_stats[f"sod_class_{cls}_ratio"]
            print(f"    Class {cls}: {cls_count:.1f} pixels ({cls_ratio*100:.2f}%)")

        # Summary for FLOE
        print("\nFLOE SIZE (FLOE):")
        print(f"  Valid Pixels: {avg_stats['floe_valid_pixels']:.1f} ({avg_stats['floe_valid_ratio']*100:.2f}%)")
        print(f"  Invalid Pixels: {avg_stats['floe_invalid_pixels']:.1f} ({(1-avg_stats['floe_valid_ratio'])*100:.2f}%)")
        print(f"  Edge Ratio: {avg_stats['floe_edge_ratio']*100:.2f}%")
        print("  Class Distribution (Valid Pixels):")
        for cls in chart_info["FLOE"]["classes"]:
            cls_count = avg_stats[f"floe_class_{cls}_count"]
            cls_ratio = avg_stats[f"floe_class_{cls}_ratio"]
            print(f"    Class {cls}: {cls_count:.1f} pixels ({cls_ratio*100:.2f}%)")

        # Save to CSV
        results_df = pd.DataFrame(all_stats)
        results_df.to_csv(f"{output_folder}nc_file_analysis_results.csv", index=False)
        print(f"\nDetailed results saved to '{output_folder}nc_file_analysis_results.csv'")

        # Save average stats
        avg_df = pd.DataFrame([avg_stats])
        avg_df.to_csv(f"{output_folder}nc_file_analysis_average.csv", index=False)
        print(f"Average statistics saved to '{output_folder}nc_file_analysis_average.csv'")
    else:
        print("No valid NC files were processed.")


if __name__ == "__main__":
    main()
