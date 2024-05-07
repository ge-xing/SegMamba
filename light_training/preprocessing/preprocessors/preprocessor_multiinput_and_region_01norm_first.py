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
from light_training.preprocessing.normalization.default_normalization_schemes import CTNormalization, ZScoreNormalization, CTNormStandard
import SimpleITK as sitk 
from tqdm import tqdm 
from copy import deepcopy
import json 
from .default_preprocessor import DefaultPreprocessor

class MultiInputAndRegionPreprocessor(DefaultPreprocessor):
    def __init__(self, 
                 base_dir,
                 image_dir,
                 data_filenames=[],
                 seg_filename="",
                 norm_clip_min=-175,
                 norm_clip_max=250,
                 ):
        
        self.base_dir = base_dir
        self.image_dir = image_dir
        self.data_filenames = data_filenames
        self.seg_filename = seg_filename
        self.norm_clip_min = norm_clip_min
        self.norm_clip_max = norm_clip_max

    def get_iterable_list(self):
        all_cases = os.listdir(os.path.join(self.base_dir, self.image_dir))
        return all_cases

    def _normalize(self, data: np.ndarray, seg: np.ndarray,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        # for c in range(data.shape[0]):
        normalizer = CTNormStandard(a_min=self.norm_clip_min, 
                                    a_max=self.norm_clip_max, 
                                    b_min=0.0,
                                    b_max=1.0, clip=True)
        
        data = normalizer(data)
        return data

    # def convert_labels_to_region(self, labels):
    #     patch_size = labels.shape[1:]
    #     one_hot_labels = np.zeros([self.all_labels_num, 
    #                                  patch_size[0], 
    #                                  patch_size[1], 
    #                                  patch_size[2]])
        
    #     for k, v in self.all_labels_dict.items():
    #         if isinstance(v, list):
    #             for vv in v:
    #                 one_hot_labels[vv-1] = (labels == vv)[0]

    #     return one_hot_labels
    
    def run_case_npy(self, data: np.ndarray, seg, properties: dict):
        # let's not mess up the inputs!
        data = np.copy(data)
        old_shape = data.shape
        original_spacing = list(properties['spacing'])
        ## 由于old spacing读出来是反的，因此这里需要转置一下

        original_spacing_trans = original_spacing[::-1]
        properties["original_spacing_trans"] = original_spacing_trans
        properties["target_spacing_trans"] = self.out_spacing

        ### norm first 
        need_to_check = False
        if seg is None :
            seg_norm = np.zeros_like(data)
        else :
            seg_norm = seg
            before_crop_seg_sum = np.sum(seg.astype(np.uint8))
            need_to_check = True
        data = self._normalize(data, seg_norm,
                               self.foreground_intensity_properties_per_channel)
        
        shape_before_cropping = data.shape[1:]
        ## crop
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg)

        if need_to_check:
            seg_temp = np.copy(seg)
            seg_temp[seg_temp==-1] = 0
            after_crop_seg_sum = np.sum(seg_temp.astype(np.uint8))
            print(f"before crop seg sum is {before_crop_seg_sum}, after is {after_crop_seg_sum}")
            
        properties['bbox_used_for_cropping'] = bbox

        # crop, remember to store size before cropping!
        shape_before_resample = data.shape[1:]
        properties['shape_after_cropping_before_resample'] = shape_before_resample

        new_shape = compute_new_shape(data.shape[1:], original_spacing_trans, self.out_spacing)

        assert len(data.shape) == 4

        data = resample_data_or_seg_to_shape(data, new_shape, 
                                             original_spacing, 
                                             self.out_spacing,
                                             order=3,
                                             order_z=0)
        properties['shape_after_resample'] = new_shape
        
        if seg is not None :
            assert len(seg.shape) == 4
            seg = resample_data_or_seg_to_shape(seg, new_shape, 
                                                original_spacing, 
                                                self.out_spacing,
                                                is_seg=True,
                                                order=1,
                                                order_z=0)

            properties['class_locations'] = self._sample_foreground_locations(seg, 
                                                                              self.all_labels,
                                                                              True)
            
            ## convert to one-hot
            # seg = self.convert_labels_to_region(seg)

            if np.max(seg) > 127:
                seg = seg.astype(np.int16)
            else:
                seg = seg.astype(np.int8)
            
        print(f'old shape: {old_shape}, shape_after_cropping_before_resample is {shape_before_resample}, new_shape after crop and resample: {new_shape}, old_spacing: {original_spacing}, '
                f'new_spacing: {self.out_spacing}, boxes is {bbox}')
           
        return data, seg
    
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
            all_labels_dict,
            num_processes=8):
        self.out_spacing = output_spacing
        # all_labels 必须为region格式，例如[[0, 1, 2, 3], [4, 5], [6, 7, 8], 9, 10]
        
        self.all_labels_dict = all_labels_dict
        self.all_labels = []
        
        for k, v in all_labels_dict.items():
            self.all_labels.append(v)

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