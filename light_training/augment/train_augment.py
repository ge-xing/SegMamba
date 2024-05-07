import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor


def get_train_transforms(patch_size, mirror_axes=None):
    tr_transforms = []
    patch_size_spatial = patch_size
    ignore_axes = None
    angle = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

    tr_transforms.append(SpatialTransform(
    patch_size_spatial, patch_center_dist_from_border=None,
    do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
    do_rotation=True, angle_x=angle, angle_y=angle, angle_z=angle,
    p_rot_per_axis=1,  # todo experiment with this
    do_scale=True, scale=(0.7, 1.4),
    border_mode_data="constant", border_cval_data=0, order_data=3,
    border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
    random_crop=False,  # random cropping is part of our dataloaders
    p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
    independent_scale_for_each_axis=False  # todo experiment with this
    ))

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                            p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                    p_per_channel=0.5,
                                                    order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                    ignore_axes=ignore_axes))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes))

    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))

    tr_transforms = Compose(tr_transforms)

    return tr_transforms

def get_train_transforms_nomirror(patch_size, mirror_axes=None):
    tr_transforms = []
    patch_size_spatial = patch_size
    ignore_axes = None
    angle = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

    tr_transforms.append(SpatialTransform(
    patch_size_spatial, patch_center_dist_from_border=None,
    do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
    do_rotation=True, angle_x=angle, angle_y=angle, angle_z=angle,
    p_rot_per_axis=1,  # todo experiment with this
    do_scale=True, scale=(0.7, 1.4),
    border_mode_data="constant", border_cval_data=0, order_data=3,
    border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
    random_crop=False,  # random cropping is part of our dataloaders
    p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
    independent_scale_for_each_axis=False  # todo experiment with this
    ))

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                            p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                    p_per_channel=0.5,
                                                    order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                    ignore_axes=ignore_axes))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    # if mirror_axes is not None and len(mirror_axes) > 0:
    #     tr_transforms.append(MirrorTransform(mirror_axes))

    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))

    tr_transforms = Compose(tr_transforms)

    return tr_transforms

def get_train_transforms_onlymirror(patch_size, mirror_axes=None):
    tr_transforms = []
    patch_size_spatial = patch_size
    ignore_axes = None
    angle = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

    # tr_transforms.append(SpatialTransform(
    # patch_size_spatial, patch_center_dist_from_border=None,
    # do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
    # do_rotation=True, angle_x=angle, angle_y=angle, angle_z=angle,
    # p_rot_per_axis=1,  # todo experiment with this
    # do_scale=True, scale=(0.7, 1.4),
    # border_mode_data="constant", border_cval_data=0, order_data=3,
    # border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
    # random_crop=False,  # random cropping is part of our dataloaders
    # p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
    # independent_scale_for_each_axis=False  # todo experiment with this
    # ))

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                            p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                    p_per_channel=0.5,
                                                    order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                    ignore_axes=ignore_axes))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes))

    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))

    tr_transforms = Compose(tr_transforms)

    return tr_transforms

def get_train_transforms_onlyspatial(patch_size, mirror_axes=None):
    tr_transforms = []
    patch_size_spatial = patch_size
    ignore_axes = None
    angle = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

    tr_transforms.append(SpatialTransform(
    patch_size_spatial, patch_center_dist_from_border=None,
    do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
    do_rotation=True, angle_x=angle, angle_y=angle, angle_z=angle,
    p_rot_per_axis=1,  # todo experiment with this
    do_scale=True, scale=(0.7, 1.4),
    border_mode_data="constant", border_cval_data=0, order_data=3,
    border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
    random_crop=False,  # random cropping is part of our dataloaders
    p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
    independent_scale_for_each_axis=False  # todo experiment with this
    ))

    # tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    # tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
    #                                         p_per_channel=0.5))
    # tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    # tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    # tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
    #                                                 p_per_channel=0.5,
    #                                                 order_downsample=0, order_upsample=3, p_per_sample=0.25,
    #                                                 ignore_axes=ignore_axes))
    # tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    # tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes))

    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))

    tr_transforms = Compose(tr_transforms)

    return tr_transforms

def get_train_transforms_noaug(patch_size, mirror_axes=None):
    tr_transforms = []
    # patch_size_spatial = patch_size
    # ignore_axes = None
    # angle = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

    # tr_transforms.append(SpatialTransform(
    # patch_size_spatial, patch_center_dist_from_border=None,
    # do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
    # do_rotation=True, angle_x=angle, angle_y=angle, angle_z=angle,
    # p_rot_per_axis=1,  # todo experiment with this
    # do_scale=True, scale=(0.7, 1.4),
    # border_mode_data="constant", border_cval_data=0, order_data=3,
    # border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
    # random_crop=False,  # random cropping is part of our dataloaders
    # p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
    # independent_scale_for_each_axis=False  # todo experiment with this
    # ))

    # tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    # tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
    #                                         p_per_channel=0.5))
    # tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    # tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    # tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
    #                                                 p_per_channel=0.5,
    #                                                 order_downsample=0, order_upsample=3, p_per_sample=0.25,
    #                                                 ignore_axes=ignore_axes))
    # tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    # tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    # if mirror_axes is not None and len(mirror_axes) > 0:
    #     tr_transforms.append(MirrorTransform(mirror_axes))

    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))

    tr_transforms = Compose(tr_transforms)

    return tr_transforms

def get_validation_transforms() -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))

        # val_transforms.append(RenameTransform('seg', 'target', True))

        val_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms

# import SimpleITK as sitk 
# import matplotlib.pyplot as plt 

# image = sitk.ReadImage("/Users/xingzhaohu/Documents/工作/code/medical_image_processing/SSL/BraTS20_Training_365/BraTS20_Training_365_flair.nii.gz")
# label = sitk.ReadImage("/Users/xingzhaohu/Documents/工作/code/medical_image_processing/SSL/BraTS20_Training_365/BraTS20_Training_365_seg.nii.gz")

# # image = sitk.ReadImage("./AIIB/image/AIIB23_171.nii.gz")
# # label = sitk.ReadImage("./AIIB/gt/AIIB23_171.nii.gz")

# image_arr = sitk.GetArrayFromImage(image)
# label_arr = sitk.GetArrayFromImage(label)
# intensityproperties = {}

# norm = RescaleTo01Normalization(intensityproperties=intensityproperties)
# image_arr = image_arr[0:128, 0:128, 0:128][None, None]
# label_arr = label_arr[0:128, 0:128, 0:128][None, None]


# image_arr = norm.run(image_arr, label_arr)

# print(image_arr.shape, label_arr.shape)

# tr_transforms = Compose(tr_transforms)

# trans_out = tr_transforms(data=image_arr, seg=label_arr)

# image_arr_aug = trans_out["data"]
# label_arr_aug = trans_out["seg"]

# print(image_arr_aug.shape, label_arr_aug.shape)


# for i in range(40, 128):
#     plt.subplot(1, 4, 1)
#     plt.imshow(image_arr[0, 0, i], cmap="gray")
#     plt.subplot(1, 4, 2)
#     plt.imshow(label_arr[0, 0, i], cmap="gray")
#     plt.subplot(1, 4, 3)
#     plt.imshow(image_arr_aug[0, 0, i], cmap="gray")
#     plt.subplot(1, 4, 4)
#     plt.imshow(label_arr_aug[0, 0, i], cmap="gray")
#     plt.show()