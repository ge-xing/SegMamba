import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
# from dataset.brats_data_utils_resample128 import get_loader_brats
import torch 
import torch.nn as nn 
# from ddim_seg.basic_unet import BasicUNet
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from models.uent2d import UNet2D
from models.uent3d import UNet3D
from monai.networks.nets.segresnet import SegResNet
# from ddim_seg.unet3d import DiffusionUNet
# from ddim_seg.ddim import DDIM
# from ddim_seg.nnunet3d_raw import Generic_UNet
# from ddim_seg.basic_unet_denose import BasicUNetDe
# from ddim_seg.basic_unet import BasicUNetEncoder
from models.transbts.TransBTS_downsample8x_skipconnection import TransBTS
import argparse
from monai.losses.dice import DiceLoss
# from light_training.model.bit_diffusion import decimal_to_bits, bits_to_decimal

# from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
# from guided_diffusion.respace import SpacedDiffusion, space_timesteps
# from guided_diffusion.resample import UniformSampler
set_determinism(123)
import os
from scipy import ndimage


os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
data_dir = "./data/fullres/train"

logdir = f"./logs_gpu4/diffunet_ep2000"

model_save_path = os.path.join(logdir, "model")
# augmentation = "nomirror"
augmentation = True

env = "pytorch"
max_epoch = 2000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
roi_size = [128, 128, 128]

def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if (dim == 2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1) 
    ero = ndimage.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge

def edge_3d(image_3d):
    # image_3d = torch.from_numpy(image_3d)
    b, c, d, h, w = image_3d.shape

    image_3d = image_3d[:, 0] > 0

    return_edge = []

    for i in range(image_3d.shape[0]):
        return_edge.append(get_edge_points(image_3d[i])[None,])
    
    return_edge = np.concatenate(return_edge, axis=0)

    return return_edge

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=roi_size,
                                        sw_batch_size=1,
                                        overlap=0.5)
        self.augmentation = augmentation

        from models.nnunet_denoise_ddp_infer.get_unet3d_denoise_uncer_edge import DiffUNet
        self.model = DiffUNet(1, 10, 3, 1, bta=True)

        self.patch_size = roi_size
        self.best_mean_dice = 0.0
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.train_process = 20
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                    momentum=0.99, nesterov=True)
        
        self.scheduler_type = "poly"
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)
        self.cross = nn.CrossEntropyLoss()

    def training_step(self, batch):
        image, label = self.get_input(batch)
        
        pred, pred_edge = self.model(image, label)

        loss_edge = self.cross(pred_edge, label)
        loss_seg = self.cross(pred, label)

        self.log("loss_seg", loss_seg, step=self.global_step)
        self.log("loss_edge", loss_edge, step=self.global_step)

        loss = loss_edge + loss_seg
        return loss
    
    
    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        # label = self.convert_labels(label)

        # label = label.float()
        label = label[:, 0].long()
        return image, label 

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            # hd95 = metric.binary.hd95(pred, gt)
            return np.array([d, 50])
        
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        
        else:
            return np.array([0.0, 50])
    
    def validation_step(self, batch):
        image, label = self.get_input(batch)
       
        output = self.model(image, ddim=True)

        # output = output > 0
        output = output.argmax(dim=1)        

        output = output.cpu().numpy()
        target = label.cpu().numpy()
        
        dices = []

        c = 10
        for i in range(1, c):
            pred_c = output == i
            target_c = target == i

            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)
        
        return dices
    
    def validation_end(self, val_outputs):
        dices = val_outputs

        dices_mean = []
        c = 9
        for i in range(0, c):
            dices_mean.append(dices[i].mean())

        mean_dice = sum(dices_mean) / len(dices_mean)
        
        self.log("0", dices_mean[0], step=self.epoch)
        self.log("1", dices_mean[1], step=self.epoch)
        self.log("2", dices_mean[2], step=self.epoch)
        self.log("3", dices_mean[3], step=self.epoch)
        self.log("4", dices_mean[4], step=self.epoch)
        self.log("5", dices_mean[5], step=self.epoch)
        self.log("6", dices_mean[6], step=self.epoch)
        self.log("7", dices_mean[7], step=self.epoch)
        self.log("8", dices_mean[8], step=self.epoch)

        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")


        print(f"mean_dice is {mean_dice}")

if __name__ == "__main__":

    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17759,
                            training_script=__file__)

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
