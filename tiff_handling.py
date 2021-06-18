# This file is copied from the ct_experiment_utils repository.
# For the most up to date version check https://github.com/D1rk123/ct_experiment_utils
# That repository can be included in a project as a git submodule in a project,
# but to make the tutorial simpler it was just copied here.

from pathlib import Path
import tifffile
from tqdm import tqdm
import numpy as np
import torch


def load_stack(path, *, prefix="", dtype=None, stack_axis=0, range_start=0, range_stop=None, range_step=1):
    """Load a stack of tiff files into a contiguous numpy array

    Make sure that the tiff files are sorted *alphabetically*,
    otherwise it is not going to look pretty..

    :param path: path to directory containing tiff files
    :param prefix: only images starting with this prefix are loaded
    :param dtype: sets the type of the resulting array. If provided all images will be cast to this type
    :param stack_axis: the dimension in the output where the image in the stack will be indexed
    :param range_start: the start of the range of included images, default is the first image
    :param range_stop: the end of the range of included images, default is the last image
    :param range_step: every range_step image between range_start and range_stop is included
    
    :returns: an np.array containing the values in the tiff files
    :rtype: np.array

    """
    path = Path(path).expanduser().resolve()

    img_paths = sorted(path.glob(prefix+"*.tif"))
    if range_stop is None:
        range_stop = len(img_paths)
    img_paths = img_paths[range_start:range_stop:range_step]
    
    img0 = tifffile.imread(str(img_paths[0]))
    if dtype is None:
        dtype = img0.dtype
    
    result_shape = np.insert(np.array(img0.shape), stack_axis, len(img_paths))
    result = np.empty(result_shape, dtype=dtype)
    for i, p in enumerate(tqdm(img_paths)):
        read_image = tifffile.imread(str(p)).astype(dtype=dtype, copy=False)
        if stack_axis == 0:
            result[i, ...] = read_image
        elif stack_axis == 1:
            result[:, i, ...] = read_image
        else:
            result[:, :, i] = read_image
    
    return result
    

def save_stack(path, data, *, prefix="output", exist_ok=False, parents=False, stack_axis=0):
    path = Path(path).expanduser().resolve()
    path.mkdir(exist_ok=exist_ok, parents=parents)
    
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    for i in tqdm(range(data.shape[stack_axis])):
        output_path = path / f"{prefix}_{i:05d}.tif"
        tifffile.imsave(str(output_path), data.take(indices=i, axis=stack_axis))
