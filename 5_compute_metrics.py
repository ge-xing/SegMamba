from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from monai.utils import set_determinism
import torch 
import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
import argparse
from tqdm import tqdm 

import numpy as np

set_determinism(123)

parser = argparse.ArgumentParser()

parser.add_argument("--pred_name", required=True, type=str)

results_root = "prediction_results"
args = parser.parse_args()

pred_name = args.pred_name

def cal_metric(gt, pred, voxel_spacing):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        return np.array([dice, hd95])
    else:
        return np.array([0.0, 50])

def each_cases_metric(gt, pred, voxel_spacing):
    classes_num = 3
    class_wise_metric = np.zeros((classes_num, 2))
    for cls in range(0, classes_num):
        class_wise_metric[cls, ...] = cal_metric(pred[cls], gt[cls], voxel_spacing)
    print(class_wise_metric)
    return class_wise_metric

def convert_labels(labels):
    ## TC, WT and ET
    labels = labels.unsqueeze(dim=0)

    result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
    
    return torch.cat(result, dim=0).float()


if __name__ == "__main__":
    data_dir = "./data/fullres/train"
    raw_data_dir = "./data/raw_data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)
    print(len(test_ds))
    all_results = np.zeros((250,3,2))

    ind = 0
    for batch in tqdm(test_ds, total=len(test_ds)):
        properties = batch["properties"]
        case_name = properties["name"]
        gt_itk = os.path.join(raw_data_dir, case_name, f"seg.nii.gz")
        voxel_spacing = [1, 1, 1]
        gt_itk = sitk.ReadImage(gt_itk)
        gt_array = sitk.GetArrayFromImage(gt_itk).astype(np.int32)
        gt_array = torch.from_numpy(gt_array)
        gt_array = convert_labels(gt_array).numpy()
        pred_itk = sitk.ReadImage(f"./{results_root}/{pred_name}/{case_name}.nii.gz")
        pred_array = sitk.GetArrayFromImage(pred_itk)

        m = each_cases_metric(gt_array, pred_array, voxel_spacing)

        all_results[ind, ...] = m
    
        ind += 1

    os.makedirs(f"./{results_root}/result_metrics/", exist_ok=True)
    np.save(f"./{results_root}/result_metrics/{pred_name}.npy", all_results) 
    
    result = np.load(f"./{results_root}/result_metrics/{pred_name}.npy")
    print(result.shape)
    print(result.mean(axis=0))
    print(result.std(axis=0))



