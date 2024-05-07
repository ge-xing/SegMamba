        
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

        device = model_output.device
        shape_after_cropping_before_resample = properties["shape_after_cropping_before_resample"]
        d, w, h = shape_after_cropping_before_resample[0], shape_after_cropping_before_resample[1], shape_after_cropping_before_resample[2]
        print(f"resample....")
        channel = model_output.shape[0]

        try:
            with torch.no_grad():
                resample_output = torch.zeros((channel, d, w, h), dtype=torch.half, device=device)
                for c in range(channel):  
                    resample_output[c] = torch.nn.functional.interpolate(model_output[c][None, None], mode="trilinear", size=(d, w, h))[0, 0]

                del model_output

        except RuntimeError:
            with torch.no_grad():
                model_output = model_output.to("cpu")
                resample_output = torch.zeros((channel, d, w, h))
                for c in range(channel):  
                    resample_output[c] = torch.nn.functional.interpolate(model_output[c][None, None], mode="trilinear", size=(d, w, h))[0, 0]
                del model_output
        
        torch.cuda.empty_cache()

        return resample_output

    @staticmethod
    def predict_noncrop_probability(model_output, properties):
        
        print(f"restoring noncrop region......")
        if isinstance(model_output, torch.Tensor):
            model_output = model_output.cpu().numpy()
        
        torch.cuda.empty_cache()

        if len(model_output.shape) == 3:
            shape_before_cropping = properties["shape_before_cropping"]
            if isinstance(shape_before_cropping[0], torch.Tensor):
                shape_before_cropping = [shape_before_cropping[0].item(), shape_before_cropping[1].item(), shape_before_cropping[2].item()]
            
            none_crop_pred = np.zeros([shape_before_cropping[0], shape_before_cropping[1], shape_before_cropping[2]], dtype=np.uint8)            
            bbox_used_for_cropping = properties["bbox_used_for_cropping"]

            none_crop_pred[
                        bbox_used_for_cropping[0][0]: bbox_used_for_cropping[0][1], 
                        bbox_used_for_cropping[1][0]: bbox_used_for_cropping[1][1], 
                        bbox_used_for_cropping[2][0]: bbox_used_for_cropping[2][1]] = model_output
            del model_output
            return none_crop_pred
        
        elif len(model_output.shape) == 4:
            shape_before_cropping = properties["shape_before_cropping"]
            if isinstance(shape_before_cropping[0], torch.Tensor):
                shape_before_cropping = [shape_before_cropping[0].item(), shape_before_cropping[1].item(), shape_before_cropping[2].item()]
            
            none_crop_pred = np.zeros([model_output.shape[0], shape_before_cropping[0], shape_before_cropping[1], shape_before_cropping[2]], dtype=np.uint8)            
            bbox_used_for_cropping = properties["bbox_used_for_cropping"]

            none_crop_pred[
                        :,
                        bbox_used_for_cropping[0][0]: bbox_used_for_cropping[0][1], 
                        bbox_used_for_cropping[1][0]: bbox_used_for_cropping[1][1], 
                        bbox_used_for_cropping[2][0]: bbox_used_for_cropping[2][1]] = model_output
            del model_output

            return none_crop_pred

        else:
            print(f"restore crop error")
            exit(0)
    
    def maybe_mirror_and_predict(self, x, model, device=torch.device("cpu"), **kwargs) -> torch.Tensor:
        # mirror_axes = [0, 1, 2]
        window_infer = self.window_infer
        if type(device) is str:
            device = torch.device(device)

        model.to(device)
        # if type(x) is list:
        #     for i in range(len(x)):
        #         x[i] = x[i].to(device)
        # else :
        x = x.to(device)
        with torch.no_grad():
            print(f"predicting....")
            with torch.autocast("cuda", enabled=True) if device.type == "cuda" else dummy_context():
                prediction = window_infer(x, model, **kwargs).cpu()
                mirror_axes = self.mirror_axes

                if mirror_axes is not None:
                    # check for invalid numbers in mirror_axes
                    # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
                    assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

                    num_predictons = 2 ** len(mirror_axes)
                    if 0 in mirror_axes:
                        prediction += torch.flip(window_infer(torch.flip(x, (2,)), model, **kwargs), (2,)).cpu()
                        torch.cuda.empty_cache()
                    if 1 in mirror_axes:
                        prediction += torch.flip(window_infer(torch.flip(x, (3,)), model, **kwargs), (3,)).cpu()
                        torch.cuda.empty_cache()
                    if 2 in mirror_axes:
                        prediction += torch.flip(window_infer(torch.flip(x, (4,)), model, **kwargs), (4,)).cpu()
                        torch.cuda.empty_cache()
                    if 0 in mirror_axes and 1 in mirror_axes:
                        prediction += torch.flip(window_infer(torch.flip(x, (2, 3)), model, **kwargs), (2, 3)).cpu()
                        torch.cuda.empty_cache()
                    if 0 in mirror_axes and 2 in mirror_axes:
                        prediction += torch.flip(window_infer(torch.flip(x, (2, 4)), model, **kwargs), (2, 4)).cpu()
                        torch.cuda.empty_cache()
                    if 1 in mirror_axes and 2 in mirror_axes:
                        prediction += torch.flip(window_infer(torch.flip(x, (3, 4)), model, **kwargs), (3, 4)).cpu()
                        torch.cuda.empty_cache()
                    if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
                        prediction += torch.flip(window_infer(torch.flip(x, (2, 3, 4)), model, **kwargs), (2, 3, 4)).cpu()
                        torch.cuda.empty_cache()
                    prediction /= num_predictons

                torch.cuda.empty_cache()
                del x 
                return prediction
    
    def maybe_mirror_and_predict_cuda(self, x, model, device=torch.device("cpu"), **kwargs) -> torch.Tensor:
        # mirror_axes = [0, 1, 2]
        window_infer = self.window_infer
        if type(device) is str:
            device = torch.device(device)

        model.to(device)
        x = x.to(device)
        with torch.no_grad():
            print(f"predicting....")
            with torch.autocast("cuda", enabled=True) if device.type == "cuda" else dummy_context():
                prediction = window_infer(x, model, **kwargs)
                mirror_axes = self.mirror_axes

                if mirror_axes is not None:
                    # check for invalid numbers in mirror_axes
                    # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
                    assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

                    num_predictons = 2 ** len(mirror_axes)
                    if 0 in mirror_axes:
                        prediction += torch.flip(window_infer(torch.flip(x, (2,)), model, **kwargs), (2,))
                        torch.cuda.empty_cache()
                    if 1 in mirror_axes:
                        prediction += torch.flip(window_infer(torch.flip(x, (3,)), model, **kwargs), (3,))
                        torch.cuda.empty_cache()
                    if 2 in mirror_axes:
                        prediction += torch.flip(window_infer(torch.flip(x, (4,)), model, **kwargs), (4,))
                        torch.cuda.empty_cache()
                    if 0 in mirror_axes and 1 in mirror_axes:
                        prediction += torch.flip(window_infer(torch.flip(x, (2, 3)), model, **kwargs), (2, 3))
                        torch.cuda.empty_cache()
                    if 0 in mirror_axes and 2 in mirror_axes:
                        prediction += torch.flip(window_infer(torch.flip(x, (2, 4)), model, **kwargs), (2, 4))
                        torch.cuda.empty_cache()
                    if 1 in mirror_axes and 2 in mirror_axes:
                        prediction += torch.flip(window_infer(torch.flip(x, (3, 4)), model, **kwargs), (3, 4))
                        torch.cuda.empty_cache()
                    if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
                        prediction += torch.flip(window_infer(torch.flip(x, (2, 3, 4)), model, **kwargs), (2, 3, 4)).cpu()
                        torch.cuda.empty_cache()
                    prediction /= num_predictons

                torch.cuda.empty_cache()
                del x 
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
        if isinstance(raw_spacing[0], torch.Tensor):
            raw_spacing = [raw_spacing[0].item(), raw_spacing[1].item(), raw_spacing[2].item()]

        return_output.SetSpacing((raw_spacing[0], raw_spacing[1], raw_spacing[2]))

        sitk.WriteImage(return_output, os.path.join(save_dir, f"{case_name}.nii.gz"))

        print(f"{os.path.join(save_dir, f'{case_name}.nii.gz')} is saved successfully")