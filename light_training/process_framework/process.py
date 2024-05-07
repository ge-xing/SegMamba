import torch
import numpy as np
import SimpleITK
import os
import sys
from monai.inferers import SlidingWindowInferer

class Customalgorithm():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):
        """
        Do not modify the `self.input_dir` and `self.output_dir`. 
        (Check https://grand-challenge.org/algorithms/interfaces/)
        """
        self.input_dir = "/input/"
        self.output_dir = "/output/images/head-neck-segmentation/"

        # self.out_spacing = [3.0, 0.54199219, 0.54199219] 
        self.out_spacing = [3.0, 1.0, 1.0] 

        # self.device = "cpu"

        self.device = torch.device("cuda")

        self.patch_size = [64, 128, 128]

    def filte_state_dict(self, sd):
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 
        del sd 
        return new_sd

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        print(img.GetSize())
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def read(self, mha_path):
        img = SimpleITK.ReadImage(mha_path)
        spacing = img.GetSpacing()
        raw_size = SimpleITK.GetArrayFromImage(img).shape
        img = SimpleITK.GetArrayFromImage(img)[None,].astype(np.float32)
        properties = {
            "spacing": spacing,
            "raw_size": raw_size
        }
        return img, properties

    def check_gpu(self):
        """
        Check if GPU is available. Note that the Grand Challenge only has one available GPU.
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' +
                  str(torch.cuda.get_device_properties(0).total_memory))

    def load_inputs(self):      # use two modalities input data
        """
        Read input data (two modalities) from `self.input_dir` (/input/). 
        Please do not modify the path for CT and contrast-CT images.
        """
        ct_mha = os.listdir(os.path.join(self.input_dir, 'images/head-neck-ct/'))[0]
        ctc_mha = os.listdir(os.path.join(self.input_dir, 'images/head-neck-contrast-enhanced-ct/'))[0]
        uuid = os.path.splitext(ct_mha)[0]

        img, properties = self.read(os.path.join(self.input_dir, 'images/head-neck-ct/', ct_mha))
        img_c, _ = self.read(os.path.join(self.input_dir, 'images/head-neck-contrast-enhanced-ct/', ctc_mha))

        data = np.concatenate([img, img_c], axis=0)
        del img
        del img_c
        # data is (2, d, w, h)
        return uuid, data, properties

    def crop(self, data, properties):
        from light_training.preprocessing.cropping.cropping import crop_to_nonzero

        seg = np.zeros_like(data)

        shape_before_cropping = data.shape[1:]
        ## crop
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg)
        del seg 

        properties['bbox_used_for_cropping'] = bbox

        return data, properties

    def resample(self, data, properties):
        from light_training.preprocessing.resampling.default_resampling import compute_new_shape, resample_data_or_seg_to_shape
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

        return data, properties
    
    def preprocess(self, data, properties, crop_first=True):        
        from light_training.process_framework.norm import norm_func

        original_spacing = list(properties['spacing'])
        ## 由于old spacing读出来是反的，因此这里需要转置一下
        original_spacing_trans = original_spacing[::-1]
        properties["original_spacing_trans"] = original_spacing_trans
        properties["target_spacing_trans"] = self.out_spacing

        if crop_first:
            data, properties = self.crop(data, properties)

        data = norm_func(data)
        
        if not crop_first:
            data, properties = self.crop(data, properties)

        
        data, properties = self.resample(data, properties)

        data = data[None,]

        data = torch.from_numpy(data)

        return data, properties

    def predict(self, data, properties, uid):
        torch.cuda.empty_cache()

        from models.nnunet3d import NNUNetWrapper
        model = NNUNetWrapper(norm="ins")

        new_sd = self.filte_state_dict(torch.load("./weight/unet3d_0_addaug_bs2_ep1000_ds_gpu4/final_model_0.8552.pt", map_location="cpu"))
        model.load_state_dict(new_sd)

        del new_sd
        torch.cuda.empty_cache()
        # data = data.to(self.deivce)
        # model.to(self.device)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=self.patch_size,
                                                sw_batch_size=1,
                                                overlap=0.5,
                                                progress=True,
                                                mode="gaussian")
        
        predictor = Predictor(window_infer, mirror_axes=None)
        try:
            ensemble_output = predictor.maybe_mirror_and_predict(data, model, self.device)

        except RuntimeError:
            ensemble_output = predictor.maybe_mirror_and_predict(data, model, torch.device("cpu"))
        torch.cuda.empty_cache()
        del model
        del data

        print(f"prediction done")
        ensemble_output = predictor.predict_raw_probability(ensemble_output, properties)
        print(f"non linear....")
        # ensemble_output = predictor.apply_nonlinear(ensemble_output, nonlinear_type="sigmoid")
        ensemble_output = ensemble_output > 0

        print(f"restore crop...")
        ensemble_output = predictor.predict_noncrop_probability(ensemble_output, properties)

        raw_spacing = properties["spacing"]
        case_name = uid
        print(f"uuid is {uid}")
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)

        print(f"saving....")
        predictor.save_to_nii_multi_organ(ensemble_output,
                              raw_spacing,
                              save_dir=self.output_dir,
                              case_name=case_name,
                              postprocess=False)
        
        # """
        # load the model and checkpoint, and generate the predictions. You can replace this part with your own model.
        # """
        # predict_from_folder_segrap2023(self.weight, self.nii_path, self.result_path, 0, 0, 1)
        # print("nnUNet segmentation done!")
        # if not os.path.exists(os.path.join(self.result_path, self.nii_seg_file)):
        #     print('waiting for nnUNet segmentation to be created')

        # while not os.path.exists(os.path.join(self.result_path, self.nii_seg_file)):
        #     import time
        #     print('.', end='')
        #     time.sleep(5)
        # # print(cproc)  # since nnUNet_predict call is split into prediction and postprocess, a pre-mature exit code is received but segmentation file not yet written. This hack ensures that all spawned subprocesses are finished before being printed.
        # print('Prediction finished !')

    def post_process(self):
        self.check_gpu()
        print('Start processing')
        uuid, data, properties = self.load_inputs()

        data, properties = self.preprocess(data, properties)
        print(properties)
        print('Start prediction')
        self.predict(data, properties, uuid)
        # print('Start output writing')
        # self.write_outputs(uuid)

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        self.post_process()


if __name__ == "__main__":
    Customalgorithm().process()
