import numpy as np
from light_training.dataloading.dataset import get_test_loader_from_test
import torch 
import torch.nn as nn 
from monai.networks.nets.basic_unet import BasicUNet
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from models.uent3d import UNet3D
from monai.networks.nets.segresnet import SegResNet
from models.transbts.TransBTS_downsample8x_skipconnection import TransBTS
from einops import rearrange
from models.modelgenesis.unet3d import UNet3DModelGen
from models.transvw.models.ynet3d import UNet3DTransVW
from monai.networks.nets.basic_unet import BasicUNet
from monai.networks.nets.attentionunet import AttentionUnet
from light_training.loss.compound_losses import DC_and_CE_loss
from light_training.loss.dice import MemoryEfficientSoftDiceLoss
from light_training.evaluation.metric import dice
set_determinism(123)
from light_training.loss.compound_losses import DC_and_CE_loss
import os
from medpy import metric
from light_training.prediction import Predictor


data_dir = "./data/fullres/test"
env = "pytorch"
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:2"
patch_size = [128, 128, 128]

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        self.patch_size = patch_size       

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]
        # label = self.convert_labels(label)
        del batch 
        return image, label, properties 
    
    def define_model_diffunet(self):
        from models.nnunet_denoise_ddp_infer.get_unet3d_denoise_uncer_edge import DiffUNet
        model = DiffUNet(1, 10, 3, 1, bta=True)

        model_path = "/home/xingzhaohu/zongweizhou/logs_gpu4/diffunet/model/final_model_0.8384.pt"
        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.3,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0,1,2])
        save_path = "./prediction_results/diffunet_ep1000_test"
        
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path

    def validation_step(self, batch):
        image, label, properties = self.get_input(batch)
        print(properties['spacing'])

        ddim = True
        model, predictor, save_path = self.define_model_diffunet()
    
        if ddim:
            model_output = predictor.maybe_mirror_and_predict(image, model, device=device, ddim=True)
        else :
            model_output = predictor.maybe_mirror_and_predict(image, model, device=device)

        model_output = predictor.predict_raw_probability(model_output, 
                                                         properties=properties).cpu()
        

        model_output = model_output.argmax(dim=0)

        model_output = predictor.predict_noncrop_probability(model_output, properties)
        print(f"save shape is {model_output.shape}")


        seg_list = ["aorta", "gall_bladder", "kidney_left", 
                         "kidney_right", "liver", "pancreas", 
                         "postcava", "spleen", "stomach"]
        
        save_path = os.path.join(save_path, properties['name'][0], "predictions")
        # print(f"save_path is {save_path}")
        os.makedirs(save_path, exist_ok=True)
        for i in range(1, len(seg_list) + 1):
            model_output_c = model_output == i 
            predictor.save_to_nii(model_output_c, 
                                raw_spacing=properties['spacing'],
                                case_name=seg_list[i-1],
                                save_dir=save_path)
        
        return 0
    

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
    
if __name__ == "__main__":

    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir="",
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17751,
                            training_script=__file__)
    
    test_ds = get_test_loader_from_test(data_dir=data_dir)

    trainer.validation_single_gpu(test_ds)


