import tifffile
from pathlib import Path
import torch
import tomosipo as ts
import tomosipo.torch_support
import ts_algorithms as tsa
import numpy as np
from tqdm import tqdm
from tiff_handling import load_stack, save_stack
from tomosipo.qt import animate

# Reads the scanner settings file into a dictionary
def parse_scan_settings(path):
    contents = {}
    with open(path, "r") as file:
        for line in file:
            split_point =  line.find(":")
            if split_point == -1:
                continue
            else:
                contents[line[:split_point].strip()] = line[split_point+1:].strip()
    return contents


# Preprocesses the projection data without making copies
def preprocess_in_place(y, dark, flat):
    dark = dark[:, None, :]
    flat = flat[:, None, :]
    y -= dark
    y /= (flat - dark)
    torch.log_(y)
    y *= -1


# Loads a tiff file and converts it to a float32 torch tensor
def load_tiff_to_torch(path):
    return torch.from_numpy(tifffile.imread(str(path)).astype(np.float32))


if __name__ == "__main__":
    data_path = Path("pepper_projections")
    save_path = Path("reconstruction")
    dark_image = load_tiff_to_torch(data_path / "di000000.tif")
    light_image = load_tiff_to_torch(data_path / "io000000.tif")
    scan_settings = parse_scan_settings(data_path / "scan settings.txt")
    src_det_dist = float(scan_settings["SDD"])
    src_obj_dist = float(scan_settings["SOD"])
    pixel_size = float(scan_settings["Binned pixel size"])
    pixel_width = dark_image.shape[1]
    pixel_height = dark_image.shape[0]

    y = torch.from_numpy(load_stack(data_path, prefix="scan", dtype=np.float32, stack_axis=1, range_stop=-1))
    print("finished loading")
    preprocess_in_place(y, dark_image, light_image)
    print("finished preprocessing")

    vg = ts.volume(
        shape=(pixel_height, pixel_width, pixel_width),
        size=np.array((pixel_height, pixel_width, pixel_width))*pixel_size,
        pos=0
    )
    pg = ts.cone(
        angles=y.shape[1],
        shape=(pixel_height, pixel_width),
        size=np.array((pixel_height, pixel_width))*pixel_size,
        src_det_dist = src_det_dist,
        src_orig_dist = src_obj_dist
    )
    op = ts.operator(vg, pg)
    
    s = ts.scale(1/100)
    anim = animate(s*vg, s*pg)
    anim.save("flexray.mp4")

    #dev = torch.device("cuda")
    #y.cuda()
    reconstruction = tsa.fdk(A=op, y=y, overwrite_y=True)

    save_stack(save_path, reconstruction.numpy(), exist_ok=True)

