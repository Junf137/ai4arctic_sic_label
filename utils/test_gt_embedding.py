import os

import xarray as xr


input_path = "./data/ori_test/"
output_path = "./output/test_gt_embed/"

if not os.path.exists(output_path):
    os.makedirs(output_path)


for file in os.listdir(input_path):
    if file.endswith("prep.nc"):
        scene = xr.open_dataset(input_path + file, engine="h5netcdf")
        ref_scene = xr.open_dataset(input_path + file[:-3] + "_reference.nc", engine="h5netcdf")

        # combine the scene and the reference scene
        new_scene = xr.merge([scene, ref_scene])

        new_scene.to_netcdf(output_path + file, mode="a", format="netcdf4", engine="h5netcdf")
