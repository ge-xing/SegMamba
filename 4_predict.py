import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.evaluation.metric import dice
set_determinism(123)
import os
from light_training.prediction import Predictor

data_dir = "./data/fullres/train"
env = "pytorch"
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
patch_size = [128, 128, 128]

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        self.patch_size = patch_size
        self.augmentation = False
    
    def convert_labels(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
        
        return torch.cat(result, dim=1).float()

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]
        label = self.convert_labels(label)

        return image, label, properties 

    def define_model_segmamba(self):
        from model_segmamba.segmamba import SegMamba
        model = SegMamba(in_chans=4,
                        out_chans=4,
                        depths=[2,2,2,2],
                        feat_size=[48, 96, 192, 384])
        
        model_path = "/home/xingzhaohu/dev/jiuding_code/brats23/logs/segmamba/model/final_model_0.9038.pt"
        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.5,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0,1,2])

        save_path = "./prediction_results/segmamba"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path
    
    def validation_step(self, batch):
        image, label, properties = self.get_input(batch)
        ddim = False
      
        model, predictor, save_path = self.define_model_segmamba()

        model_output = predictor.maybe_mirror_and_predict(image, model, device=device)

        model_output = predictor.predict_raw_probability(model_output, 
                                                         properties=properties)
        

        model_output = model_output.argmax(dim=0)[None]
        model_output = self.convert_labels_dim0(model_output)

        label = label[0]
        c = 3
        dices = []
        for i in range(0, c):
            output_i = model_output[i].cpu().numpy()
            label_i = label[i].cpu().numpy()
            d = dice(output_i, label_i)
            dices.append(d)

        print(dices)

        model_output = predictor.predict_noncrop_probability(model_output, properties)
        predictor.save_to_nii(model_output, 
                              raw_spacing=[1,1,1],
                              case_name = properties['name'][0],
                              save_dir=save_path)
        
        return 0

    def convert_labels_dim0(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
        
        return torch.cat(result, dim=0).float()
    

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
    
    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)

    trainer.validation_single_gpu(test_ds)

    # print(f"result is {v_mean}")


