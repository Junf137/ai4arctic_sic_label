# AI4ArcticSeaIceChallenge

## Dependencies
The following packages and versions were used to develop and test the code along with the dependancies installed with pip:
- python==3.9.11
- jupyterlab==3.4.5
- xarray==2022.10.0
- netCDF4==1.6.1
- numpy==1.23.2
- matplotlib==3.6.1
- torch==1.12.1+cu116
- tqdm==4.64.1
- sklearn==0.0
- ipywidgets==8.0.2


## Data visualization
### Dependencies
- cmocean

### Usage
All visualization code in this repository (see also `vip_ai4arctic/visualization` repo) is in the `data_visualization` directory.

The `vis_single_scene.ipynb` notebook provides an example visualization and link to the plotting function.

#### Visualize imagery & charts for a single scene (from NetCDF):
`python r2t_vis.py {filepath.nc}`

#### Visualize imagery & charts for all scenes in a directory (from NetCDF):
`python vis_all_train.py {dir}`

#### Export imagery & charts from NetCDF to file:
`python export_data.py {in_dir} {out_dir}`