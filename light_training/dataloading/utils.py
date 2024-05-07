import numpy as np 
import os 
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
import multiprocessing

def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False) -> None:
    # try:
    a = np.load(npz_file)  # inexpensive, no compression is done here. This just reads metadata
    if overwrite_existing or not isfile(npz_file[:-3] + "npy"):
        np.save(npz_file[:-3] + "npy", a['data'])
        
    if unpack_segmentation and (overwrite_existing or not isfile(npz_file[:-4] + "_seg.npy")):
        np.save(npz_file[:-4] + "_seg.npy", a['seg'])

def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = 8):
    """
    all npz files in this folder belong to the dataset, unpack them all
    """
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        npz_files = subfiles(folder, True, None, ".npz", True)
        p.starmap(_convert_to_npy, zip(npz_files,
                                       [unpack_segmentation] * len(npz_files),
                                       [overwrite_existing] * len(npz_files))
                  )
