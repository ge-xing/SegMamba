
from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor 
import numpy as np 
import pickle 
import json 

data_filename = ["t2w.nii.gz",
                 "t2f.nii.gz",
                 "t1n.nii.gz",
                 "t1c.nii.gz"]
seg_filename = "seg.nii.gz"

def process_train():
    # fullres spacing is [0.5        0.70410156 0.70410156]
    # median_shape is [602.5 516.5 516.5]
    base_dir = "./data/raw_data/BraTS2023/"
    image_dir = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )

    out_spacing = [1.0, 1.0, 1.0]
    output_dir = "./data/fullres/train/"
    
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1, 2, 3],
    )

def process_val():
    base_dir = "./data/raw_data/BraTS2023/"
    image_dir = "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=""
                                   )

    out_spacing = [1.0, 1.0, 1.0]
    output_dir = "./data/fullres/val/"
    
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1, 2, 3],
    )

def process_test():
    # fullres spacing is [0.5        0.70410156 0.70410156]
    # median_shape is [602.5 516.5 516.5]
    base_dir = "/home/xingzhaohu/sharefs/datasets/WORD-V0.1.0/"
    image_dir = "imagesTs"
    label_dir = "labelsTs"
    preprocessor = DefaultPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    label_dir=label_dir,
                                   )

    out_spacing = [3.0, 0.9765625, 0.9765625]

    output_dir = "./data/fullres/test/"
    with open("./data_analysis_result.txt", "r") as f:
        content = f.read().strip("\n")
        print(content)
    content = json.loads(content)
    foreground_intensity_properties_per_channel = content["intensity_statistics_per_channel"]
    
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                     foreground_intensity_properties_per_channel=foreground_intensity_properties_per_channel)


def plan():
    base_dir = "./data/raw_data/BraTS2023/"
    image_dir = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )
    
    preprocessor.run_plan()


if __name__ == "__main__":
# 
    # plan()

    process_train()
    # process_val()
    # process_test()
    
