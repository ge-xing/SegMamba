        
import torch
import numpy as np
import SimpleITK as sitk 
import os
from light_training.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from scipy import ndimage
import skimage.measure as measure

class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def large_connected_domain(label):
    cd, num = measure.label(label, return_num=True, connectivity=1)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    # print(volume_sort)
    label = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    label = ndimage.binary_fill_holes(label)
    label = label.astype(np.uint8)
    return label

class Predictor:
    def __init__(self, window_infer, mirror_axes=None) -> None:
        self.window_infer = window_infer
        self.mirror_axes = mirror_axes

    @staticmethod
    def predict_raw_probability(model_output, properties):
        if len(model_output.shape) == 5:
            model_output = model_output[0]

        shape_before_resample = model_output.shape
        if isinstance(model_output, torch.Tensor):
            model_output = model_output.cpu().numpy()

        spacing = properties["spacing"]
        new_spacing = [spacing[0].item(), spacing[1].item(), spacing[2].item()]
        new_spacing_trans = new_spacing[::-1]

        print(f"current spacing is {[0.5, 0.70410156, 0.70410156]}, new_spacing is {new_spacing_trans}")
        shape_after_cropping_before_resample = properties["shape_after_cropping_before_resample"]
        d, w, h = shape_after_cropping_before_resample[0].item(), shape_after_cropping_before_resample[1].item(), shape_after_cropping_before_resample[2].item()
        # model_output = torch.nn.functional.interpolate(model_output, mode="nearest", size=(d, w, h))
        model_output = resample_data_or_seg_to_shape(model_output,
                                                     new_shape=(d, w, h),
                                                     current_spacing=[0.5, 0.70410156, 0.70410156],
                                                     new_spacing=new_spacing_trans,
                                                     is_seg=False,
                                                     order=1,
                                                     order_z=0)
        shape_after_resample = model_output.shape
        print(f"before resample shape: {shape_before_resample}, after resample shape: {shape_after_resample}")
        
        return model_output 

    @staticmethod
    def apply_nonlinear(model_output, nonlinear_type="softmax"):
        if isinstance(model_output, np.ndarray):
            model_output = torch.from_numpy(model_output)
        assert len(model_output.shape) == 4

        assert nonlinear_type in ["softmax", "sigmoid"]

        if nonlinear_type == "softmax":
            model_output = torch.softmax(model_output, dim=0)
            model_output = model_output.argmax(dim=0)
        else :
            model_output = torch.sigmoid(model_output)
        
        return model_output.numpy()
    

    @staticmethod
    def predict_noncrop_probability(model_output, properties):
        assert len(model_output.shape) == 3

        shape_before_cropping = properties["shape_before_cropping"]
        none_crop_pred = np.zeros([shape_before_cropping[0], shape_before_cropping[1], shape_before_cropping[2]], dtype=np.uint8)
        bbox_used_for_cropping = properties["bbox_used_for_cropping"]

        none_crop_pred[
                    bbox_used_for_cropping[0][0]: bbox_used_for_cropping[0][1], 
                    bbox_used_for_cropping[1][0]: bbox_used_for_cropping[1][1], 
                    bbox_used_for_cropping[2][0]: bbox_used_for_cropping[2][1]] = model_output

        return model_output
    
    def maybe_mirror_and_predict(self, x, model, **kwargs) -> torch.Tensor:
        # mirror_axes = [0, 1, 2]
        window_infer = self.window_infer
        device = next(model.parameters()).device
        
        with torch.no_grad():
            prediction = window_infer(x, model, **kwargs)
            mirror_axes = self.mirror_axes

            if mirror_axes is not None:
                # check for invalid numbers in mirror_axes
                # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
                assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

                num_predictons = 2 ** len(mirror_axes)
                if 0 in mirror_axes:
                    prediction += torch.flip(window_infer(torch.flip(x, (2,)), model, **kwargs), (2,))
                if 1 in mirror_axes:
                    prediction += torch.flip(window_infer(torch.flip(x, (3,)), model, **kwargs), (3,))
                if 2 in mirror_axes:
                    prediction += torch.flip(window_infer(torch.flip(x, (4,)), model, **kwargs), (4,))
                if 0 in mirror_axes and 1 in mirror_axes:
                    prediction += torch.flip(window_infer(torch.flip(x, (2, 3)), model, **kwargs), (2, 3))
                if 0 in mirror_axes and 2 in mirror_axes:
                    prediction += torch.flip(window_infer(torch.flip(x, (2, 4)), model, **kwargs), (2, 4))
                if 1 in mirror_axes and 2 in mirror_axes:
                    prediction += torch.flip(window_infer(torch.flip(x, (3, 4)), model, **kwargs), (3, 4))
                if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
                    prediction += torch.flip(window_infer(torch.flip(x, (2, 3, 4)), model, **kwargs), (2, 3, 4))
                prediction /= num_predictons
            
            return prediction
    
    def save_to_nii(self, return_output,
                    raw_spacing,
                    save_dir,
                    case_name,
                    postprocess=False):
        return_output = return_output.astype(np.uint8)

        # # postprocessing
        if postprocess:
            return_output = large_connected_domain(return_output)
        
        return_output = sitk.GetImageFromArray(return_output)
        return_output.SetSpacing((raw_spacing[0].item(), raw_spacing[1].item(), raw_spacing[2].item()))

        sitk.WriteImage(return_output, os.path.join(save_dir, f"{case_name}.nii.gz"))