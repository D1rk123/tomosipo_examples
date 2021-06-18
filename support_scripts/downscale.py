import tifffile
from pathlib import Path
import torch
import tomosipo as ts
import tomosipo.torch_support
import ts_algorithms as tsa
import numpy as np
from tqdm import tqdm
from tiff_handling import load_stack, save_stack
from skimage.transform import rescale, resize, downscale_local_mean

data_in_path = Path("/home/dirkschut/Scandata/paprikas/2021-04-21_day3/F3")
data_out_path = Path("pepper_projections")

sino = load_stack(data_in_path, prefix="scan", stack_axis=1, skip_last=True)
sino = np.round(downscale_local_mean(sino, (4, 8, 4))).astype(np.uint16)
save_stack(data_out_path, sino, prefix="scan", stack_axis=1)

other_imgs = data_in_path.glob("*i*.tif")

for img_path in other_imgs:
    img = tifffile.imread(str(img_path))
    img = np.round(downscale_local_mean(img, (4, 4))).astype(np.uint16)
    tifffile.imsave(str(data_out_path / img_path.name), img)
