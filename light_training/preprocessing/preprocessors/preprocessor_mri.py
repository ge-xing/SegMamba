#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import multiprocessing
import shutil
from time import sleep
from typing import Union, Tuple
import glob
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from light_training.preprocessing.cropping.cropping import crop_to_nonzero
# from .default_resampling import resample_data_or_seg_to_spacing, resample_img
from light_training.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape, compute_new_shape
from tqdm import tqdm
from light_training.preprocessing.normalization.default_normalization_schemes import CTNormalization, ZScoreNormalization
import SimpleITK as sitk 
from tqdm import tqdm 
from copy import deepcopy
import json 
from .default_preprocessor import DefaultPreprocessor

class MultiModalityPreprocessor(DefaultPreprocessor):
    def __init__(self, 
                 base_dir,
                 image_dir,
                 data_filenames=[],
                 seg_filename="",
                 ):
        self.base_dir = base_dir
        self.image_dir = image_dir
        self.data_filenames = data_filenames
        self.seg_filename = seg_filename

    def get_iterable_list(self):
        all_cases = os.listdir(os.path.join(self.base_dir, self.image_dir))
        return all_cases

    def _normalize(self, data: np.ndarray, seg: np.ndarray,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            normalizer_class = ZScoreNormalization
            normalizer = normalizer_class(use_mask_for_norm=False,
                                          intensityproperties=foreground_intensity_properties_per_channel)
            data[c] = normalizer.run(data[c], seg[0])
        return data
    
    # need to modify
    def read_data(self, case_name):
        ## only for CT dataset
        assert len(self.data_filenames) != 0
        data = []
        for dfname in self.data_filenames:
            d = sitk.ReadImage(os.path.join(self.base_dir, self.image_dir, case_name, dfname))
            spacing = d.GetSpacing()
            data.append(sitk.GetArrayFromImage(d).astype(np.float32)[None,])
        
        data = np.concatenate(data, axis=0)

        seg_arr = None
        ## 一定要是float32！！！！
    
        if self.seg_filename != "":
            seg = sitk.ReadImage(os.path.join(self.base_dir, self.image_dir, case_name, self.seg_filename))
            ## 读出来以后一定转float32!!!
            seg_arr = sitk.GetArrayFromImage(seg).astype(np.float32)
            seg_arr = seg_arr[None]
            intensities_per_channel, intensity_statistics_per_channel = self.collect_foreground_intensities(seg_arr, data)

        else :
            intensities_per_channel = []
            intensity_statistics_per_channel = []

        properties = {"spacing": spacing, 
                      "raw_size": data.shape[1:], 
                      "name": case_name.split(".")[0],
                      "intensities_per_channel": intensities_per_channel,
                      "intensity_statistics_per_channel": intensity_statistics_per_channel}

        return data, seg_arr, properties
    
    def run(self, 
            output_spacing, 
            output_dir, 
            all_labels,
            num_processes=8):
        self.out_spacing = output_spacing
        self.all_labels = all_labels
        self.output_dir = output_dir
        self.foreground_intensity_properties_per_channel = {}

        all_iter = self.get_iterable_list()
        
        maybe_mkdir_p(self.output_dir)

        # test_run 
        for case_name in all_iter:
            self.run_case_save(case_name)
            break

        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            for case_name in all_iter:
                r.append(p.starmap_async(self.run_case_save,
                                         ((case_name, ),)))
            remaining = list(range(len(all_iter)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]
            with tqdm(desc=None, total=len(all_iter)) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                           'OK jokes aside.\n'
                                           'One of your background processes is missing. This could be because of '
                                           'an error (look for an error message) or because it was killed '
                                           'by your OS due to running out of RAM. If you don\'t see '
                                           'an error message, out of RAM is likely the problem. In that case '
                                           'reducing the number of workers might help')
                    done = [i for i in remaining if r[i].ready()]
                    for _ in done:
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)