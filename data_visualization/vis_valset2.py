from r2t_vis import run_vis
import json
from tqdm import tqdm


def process_filename(file):
    return f"{file[17:32]}_{file[77:80]}_prep.nc"


# read json file as datalist
with open("../datalists/valset2.json") as f:
    datalist = json.load(f)


for file in tqdm(datalist):
    # combine file name with path ../data/r2t/train/
    file = "../data/r2t/train/" + process_filename(file)
    # run vis
    run_vis(file, "../output/valset2/")
