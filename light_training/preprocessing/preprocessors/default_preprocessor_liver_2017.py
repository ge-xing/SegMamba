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

def create_image(image_arr, spacing):
    image = sitk.GetImageFromArray(image_arr)
    image.SetSpacing(spacing)
    return image 

def get_shape_must_be_divisible_by(net_numpool_per_axis):
    return 2 ** np.array(net_numpool_per_axis)

def pad_shape(shape, must_be_divisible_by):
    """
    pads shape so that it is divisible by must_be_divisible_by
    :param shape:
    :param must_be_divisible_by:
    :return:
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shp = [shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i] for i in range(len(shape))]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    new_shp = np.array(new_shp).astype(int)
    return new_shp

def get_pool_and_conv_props(spacing, patch_size, min_feature_map_size, max_numpool):
    """
    this is the same as get_pool_and_conv_props_v2 from old nnunet

    :param spacing:
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck
    :param max_numpool:
    :return:
    """
    # todo review this code
    dim = len(spacing)

    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    pool_op_kernel_sizes = [[1] * len(spacing)]
    conv_kernel_sizes = []

    num_pool_per_axis = [0] * dim
    kernel_size = [1] * dim

    while True:
        # exclude axes that we cannot pool further because of min_feature_map_size constraint
        valid_axes_for_pool = [i for i in range(dim) if current_size[i] >= 2*min_feature_map_size]
        if len(valid_axes_for_pool) < 1:
            break

        spacings_of_axes = [current_spacing[i] for i in valid_axes_for_pool]

        # find axis that are within factor of 2 within smallest spacing
        min_spacing_of_valid = min(spacings_of_axes)
        valid_axes_for_pool = [i for i in valid_axes_for_pool if current_spacing[i] / min_spacing_of_valid < 2]

        # max_numpool constraint
        valid_axes_for_pool = [i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool]

        if len(valid_axes_for_pool) == 1:
            if current_size[valid_axes_for_pool[0]] >= 3 * min_feature_map_size:
                pass
            else:
                break
        if len(valid_axes_for_pool) < 1:
            break

        # now we need to find kernel sizes
        # kernel sizes are initialized to 1. They are successively set to 3 when their associated axis becomes within
        # factor 2 of min_spacing. Once they are 3 they remain 3
        for d in range(dim):
            if kernel_size[d] == 3:
                continue
            else:
                if spacings_of_axes[d] / min(current_spacing) < 2:
                    kernel_size[d] = 3

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]

        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = np.ceil(current_size[v] / 2)
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(deepcopy(kernel_size))
        #print(conv_kernel_sizes)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    conv_kernel_sizes.append([3]*dim)
    return num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, must_be_divisible_by


class DefaultPreprocessor(object):
    def __init__(self, 
                 base_dir,
                 ):
        """
        Everything we need is in the plans. Those are given when run() is called
        """
        self.base_dir = base_dir

    def run_case_npy(self, data: np.ndarray, seg, properties: dict):
        # let's not mess up the inputs!
        data = np.copy(data)
        old_shape = data.shape
        original_spacing = list(properties['spacing'])
        ## 由于old spacing读出来是反的，因此这里需要转置一下

        original_spacing_trans = original_spacing[::-1]
        properties["original_spacing_trans"] = original_spacing_trans
        properties["target_spacing_trans"] = self.out_spacing

        shape_before_cropping = data.shape[1:]
        ## crop
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox

        # crop, remember to store size before cropping!
        shape_before_resample = data.shape[1:]
        properties['shape_after_cropping_before_resample'] = shape_before_resample

        new_shape = compute_new_shape(data.shape[1:], original_spacing_trans, self.out_spacing)

        if seg is None :
            seg_norm = np.zeros_like(data)
        else :
            seg_norm = seg 
        data = self._normalize(data, seg_norm,
                               self.foreground_intensity_properties_per_channel)

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
                                                                              )
            
            if np.max(seg) > 127:
                seg = seg.astype(np.int16)
            else:
                seg = seg.astype(np.int8)
            
        print(f'old shape: {old_shape}, new_shape after crop and resample: {new_shape}, old_spacing: {original_spacing}, '
                f'new_spacing: {self.out_spacing}, boxes is {bbox}')
           
        return data, seg

    # need to modify
    def get_iterable_list(self):
        all_cases = os.listdir(self.base_dir)

        all_cases_2 = []
        for c in all_cases:
            if "volume" in c and ".nii" in c:
                ## get data id 
                all_cases_2.append(c.split("-")[-1].split(".")[0])

        return all_cases_2

    def _normalize(self, data: np.ndarray, seg: np.ndarray,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            normalizer_class = CTNormalization
            normalizer = normalizer_class(use_mask_for_norm=False,
                                          intensityproperties=foreground_intensity_properties_per_channel[str(c)])
            data[c] = normalizer.run(data[c], seg[0])
        return data

    # need to modify
    def read_data(self, case_name):
        ## only for CT dataset
        try:
            data = sitk.ReadImage(os.path.join(self.base_dir, f"volume-{case_name}.nii"))
        except:
            print(f"data read error: {self.base_dir, case_name}")
            return None, None, None
        seg_arr = None
        ## 一定要是float32！！！！
        data_arr = sitk.GetArrayFromImage(data).astype(np.float32)
        data_arr = data_arr[None]

        if os.path.exists(os.path.join(self.base_dir, f"segmentation-{case_name}.nii")):
            seg = sitk.ReadImage(os.path.join(self.base_dir, f"segmentation-{case_name}.nii"))
            ## 读出来以后一定转float32!!!
            seg_arr = sitk.GetArrayFromImage(seg).astype(np.float32)[None,]

            intensities_per_channel, intensity_statistics_per_channel = self.collect_foreground_intensities(seg_arr, data_arr)
        else :
            intensities_per_channel = []
            intensity_statistics_per_channel = []

        properties = {"spacing": data.GetSpacing(), 
                      "raw_size": data_arr.shape[1:], 
                      "name": case_name.split(".")[0],
                      "intensities_per_channel": intensities_per_channel,
                      "intensity_statistics_per_channel": intensity_statistics_per_channel}

        return data_arr, seg_arr, properties
    
    def run_case(self, case_name):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        data, seg, properties = self.read_data(case_name)
        if data is not None:
            data, seg = self.run_case_npy(data, seg, properties)
            return data, seg, properties 
        else :
            return None, None, None 

    def run_case_save(self, case_name):
        print(case_name + "~~~~~~~~" * 10)
        data, seg, properties = self.run_case(case_name)
        if data is not None:
            # print('dtypes', data.dtype, seg.dtype)
            case_name = case_name.split(".")[0]
            np.savez_compressed(os.path.join(self.output_dir, case_name) + '.npz', data=data, seg=seg)
            write_pickle(properties, os.path.join(self.output_dir, case_name) + '.pkl')
            print(f"data is saved at: {os.path.join(self.output_dir, case_name) + '.npz'}")
    
    def experiment_plan(self, case_name):

        data, seg, properties = self.read_data(case_name)
        if data is None:
            return None, None, None
        
        print(f"labels is {np.unique(seg)}")
        spacing = properties["spacing"]
        raw_size = properties["raw_size"]
        intensities_per_channel = properties["intensities_per_channel"]

        return spacing, raw_size, intensities_per_channel

    def determine_fullres_target_spacing(self, spacings, sizes) -> np.ndarray:
        # if self.overwrite_target_spacing is not None:
        #     return np.array(self.overwrite_target_spacing)

        # spacings = self.dataset_fingerprint['spacings']
        # sizes = self.dataset_fingerprint['shapes_after_crop']

        target = np.percentile(np.vstack(spacings), 50, 0)
        target_size = np.percentile(np.vstack(sizes), 50, 0)
        # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
        # the following properties:
        # - one axis which much lower resolution than the others
        # - the lowres axis has much less voxels than the others
        # - (the size in mm of the lowres axis is also reduced)
        worst_spacing_axis = np.argmax(target)
        other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        other_spacings = [target[i] for i in other_axes]
        other_sizes = [target_size[i] for i in other_axes]

        has_aniso_spacing = target[worst_spacing_axis] > (3 * max(other_spacings))
        has_aniso_voxels = target_size[worst_spacing_axis] * 3 < min(other_sizes)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
            # don't let the spacing of that axis get higher than the other axes
            if target_spacing_of_that_axis < max(other_spacings):
                target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
            target[worst_spacing_axis] = target_spacing_of_that_axis
        return target

    def compute_new_shape(self, old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                      old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
        ## spacing need to be transposed
        old_spacing = list(old_spacing)[::-1]
        new_spacing = list(new_spacing)[::-1]

        assert len(old_spacing) == len(old_shape)
        assert len(old_shape) == len(new_spacing)
        new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
        return new_shape

    def run_plan(self):
        all_iter = self.get_iterable_list()
        spacings = []
        sizes = []
        intensities_per_channels = []
        print(f"analysing data......")
        for case in tqdm(all_iter, total=len(all_iter)):
            spacing, size, intensities_per_channel = self.experiment_plan(case)
            if spacing is None:
                continue

            spacings.append(spacing)
            sizes.append(size)
            intensities_per_channels.append(intensities_per_channel)
        
        print(f"all spacing is {spacings}")
        print(f"all sizes is {sizes}")
        foreground_intensities_per_channel = [np.concatenate([r[i] for r in intensities_per_channels]) for i in
                                                  range(len(intensities_per_channels[0]))]
        
        num_channels = len(intensities_per_channels[0])

        intensity_statistics_per_channel = {}
        for i in range(num_channels):
            intensity_statistics_per_channel[i] = {
                'mean': float(np.mean(foreground_intensities_per_channel[i])),
                'median': float(np.median(foreground_intensities_per_channel[i])),
                'std': float(np.std(foreground_intensities_per_channel[i])),
                'min': float(np.min(foreground_intensities_per_channel[i])),
                'max': float(np.max(foreground_intensities_per_channel[i])),
                'percentile_99_5': float(np.percentile(foreground_intensities_per_channel[i], 99.5)),
                'percentile_00_5': float(np.percentile(foreground_intensities_per_channel[i], 0.5)),
            }

        print(f"intensity_statistics_per_channel is {intensity_statistics_per_channel}")

        fullres_spacing = self.determine_fullres_target_spacing(spacings, sizes)
        print(f"fullres spacing is {fullres_spacing[::-1]}")

        # get transposed new median shape (what we would have after resampling)
        new_shapes = [self.compute_new_shape(j, i, fullres_spacing) for i, j in
                      zip(spacings, sizes)]
        new_median_shape = np.median(new_shapes, 0)
        print(f"median_shape is {new_median_shape}")

        tmp = 1 / np.array(fullres_spacing)
        initial_patch_size = [round(i) for i in tmp * (256 ** 3 / np.prod(tmp)) ** (1 / 3)]

        print(f"initial_patch_size is {initial_patch_size[::-1]}")

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(fullres_spacing, initial_patch_size,
                                                             4,
                                                             999999)
        print(f"target medium patch size is {patch_size[::-1]}")

        analysis_path = "./data_analysis_result.txt"
        with open(analysis_path, "w") as f:

            f.write(json.dumps({
                "intensity_statistics_per_channel": intensity_statistics_per_channel,
                "fullres spacing": fullres_spacing.tolist(),
                "median_shape": new_median_shape.tolist(),
                "initial_patch_size": initial_patch_size,
                "target medium patch size": patch_size[::-1].tolist()
            }))
        print(f"Analysis done, save to {analysis_path}")
        

    def collect_foreground_intensities(self, segmentation: np.ndarray, images: np.ndarray, seed: int = 1234,
                                       num_samples: int = 10000):
        """
        images=image with multiple channels = shape (c, x, y(, z))
        """
        assert len(images.shape) == 4
        assert len(segmentation.shape) == 4

        assert not np.any(np.isnan(segmentation)), "Segmentation contains NaN values. grrrr.... :-("
        assert not np.any(np.isnan(images)), "Images contains NaN values. grrrr.... :-("

        rs = np.random.RandomState(seed)

        intensities_per_channel = []
        # we don't use the intensity_statistics_per_channel at all, it's just something that might be nice to have
        intensity_statistics_per_channel = []

        # segmentation is 4d: 1,x,y,z. We need to remove the empty dimension for the following code to work
        foreground_mask = segmentation[0] > 0

        for i in range(len(images)):
            foreground_pixels = images[i][foreground_mask]
            num_fg = len(foreground_pixels)
            # sample with replacement so that we don't get issues with cases that have less than num_samples
            # foreground_pixels. We could also just sample less in those cases but that would than cause these
            # training cases to be underrepresented
            intensities_per_channel.append(
                rs.choice(foreground_pixels, num_samples, replace=True) if num_fg > 0 else [])
            intensity_statistics_per_channel.append({
                'mean': np.mean(foreground_pixels) if num_fg > 0 else np.nan,
                'median': np.median(foreground_pixels) if num_fg > 0 else np.nan,
                'min': np.min(foreground_pixels) if num_fg > 0 else np.nan,
                'max': np.max(foreground_pixels) if num_fg > 0 else np.nan,
                'percentile_99_5': np.percentile(foreground_pixels, 99.5) if num_fg > 0 else np.nan,
                'percentile_00_5': np.percentile(foreground_pixels, 0.5) if num_fg > 0 else np.nan,

            })

        return intensities_per_channel, intensity_statistics_per_channel

    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
                                     seed: int = 1234, verbose: bool = False):
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)
            if isinstance(c, (tuple, list)):
                ## region
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = np.argwhere(mask)
            else:
                all_locs = np.argwhere(seg == c)
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[k] = selected
            if verbose:
                print(c, target_num_samples)

        return class_locs
    
    def run(self, output_spacing, 
            output_dir, 
            all_labels,
            foreground_intensity_properties_per_channel=None, 
            num_processes=8):
        self.out_spacing = output_spacing
        self.all_labels = all_labels
        self.output_dir = output_dir
        self.foreground_intensity_properties_per_channel = foreground_intensity_properties_per_channel

        all_iter = self.get_iterable_list()
        
        maybe_mkdir_p(self.output_dir)

        # test_run 
        for case_name in all_iter:
            self.run_case_save(case_name)
            break

        # multiprocessing magic.
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