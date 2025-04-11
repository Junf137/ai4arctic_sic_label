import os
import xarray as xr


old_path = "./data/gt_embed_old/"
new_path = "./output/test_gt_embed/"

idx = 0
# for each file in the old_path
for file in os.listdir(old_path):
    print(f"{idx} Comparing {file}...")

    scene_old = xr.open_dataset(old_path + file, engine="h5netcdf")
    scene_new = xr.open_dataset(new_path + file, engine="h5netcdf")

    print(f"scene_old equal to scene_new: {scene_old.equals(scene_new)}")
    idx += 1
